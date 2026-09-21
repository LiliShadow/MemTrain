import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin
from typing_extensions import override

import verl.utils.torch_functional as verl_F
from recurrent.interface import RAgent, RConfig, RDataset, RRegister
from recurrent.utils import TokenTemplate, chat_template, now, unpad
from verl.protocol import DataProto

logger = logging.getLogger(__file__)
logger.setLevel('INFO')

@dataclass
class MemoryConfig(RConfig):
    context_key: str
    max_prompt_length: int  #
    chunk_size: int  # size of each context chunk in number of tokens3
    max_memorization_length: int  # max number of tokens to memorize
    # max_input_length = max_prompt_length + chunk_size + max_memorization_length + template_length
    max_chunks: int  # max number of chunks to process
    max_final_response_length: int
    # max_output_length = max_final_response_length if final else max_memorization_length
    enable_memory_recall: bool = False
    memory_recall_n: int = 1
    memory_recall_coeff: float = 1.0

    @property
    def max_raw_input_length(self):
        return self.max_prompt_length + self.chunk_size + self.max_memorization_length

    # use property incase we want to adapt soft punishment to length.
    @property
    def gen_max_tokens_memorization(self):
        return self.max_memorization_length

    @property
    def gen_max_tokens_final_response(self):
        return self.max_final_response_length

    @property
    def gen_pad_to(self):
        return max(self.max_prompt_length, self.max_final_response_length)

class MemoryDataset(RDataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """
    def __init__(
        self,
        recurrent_config: MemoryConfig,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        data_config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if data_config.truncation != 'center':
            raise ValueError('MemoryDataset only support center truncation')
        data_config.max_prompt_length=recurrent_config.max_chunks * recurrent_config.chunk_size
        self.context_key = recurrent_config.context_key
        super().__init__(
            recurrent_config=recurrent_config,
            data_files=data_files,
            tokenizer=tokenizer,
            data_config=data_config,
            processor=processor,
        )

    @override
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]

        chat = row_dict.pop(self.prompt_key)
        context = row_dict.pop(self.context_key)

        model_inputs = self.tokenizer(context, return_tensors="pt", add_special_tokens=False)

        context_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        context_ids, attention_mask = verl_F.postprocess_data(
            input_ids=context_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id, # pyright: ignore
            left_pad=False,
            truncation=self.truncation,
        )

        row_dict["context_ids"] = context_ids[0]
        lengths = attention_mask.sum(dim=-1)
        row_dict["context_length"] = lengths[0]
        row_dict["prompt_ids"] = self.tokenizer.encode(
            chat[0]["content"], add_special_tokens=False
        )
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        row_dict["sample_uuid"] = str(uuid4())

        return row_dict

    @override
    def get_bactch_keys(self) -> Tuple[List[str], List[str]]:
         # tensor can use 2-deminsional index for chunking.
         # while prompt_ids will not be indexed, so keep it as list.
        return ["context_ids", "context_length"], ["prompt_ids"]

TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

TEMPLATE_FINAL_BOXED = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem>
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""

TEMPLATE_RECALL = """You are presented with a memory from previously read sections, and a section containing [TARGET]. Based on the memory, please predict the entity marked by [TARGET] and put the answer in \\boxed{{}}.

<memory>
{memory}
</memory>

<section>
{maskedsection}
</section>

Your answer:
"""


class MemoryAgent(RAgent):
    def __init__(self, tokenizer:PreTrainedTokenizer, config: MemoryConfig):
        self.config = config
        self.tokenizer = tokenizer
        # A trick to get a simple chat_template for any tokenizer
        # the output text looks like:
        # '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n'
        # This is a format string itself, '{message}' will be replaced by the actual message.
        self.chat_template = chat_template(tokenizer)
        self.token_message_template = TokenTemplate(self.chat_template.format(message=TEMPLATE), tokenizer)
        self.token_final_message_template = TokenTemplate(self.chat_template.format(message=TEMPLATE_FINAL_BOXED), tokenizer)
        self.token_recall_message_template = TokenTemplate(self.chat_template.format(message=TEMPLATE_RECALL), tokenizer)
        # we assume that final_message template is difinately shorter than message_template
        self.max_input_length = self.config.max_raw_input_length + self.token_message_template.length
        logger.info(f'\n[RECURRENT] max_input_length: {self.config.max_raw_input_length}(raw) '
              f'+ {self.token_message_template.length}(message_template) = {self.max_input_length}\n')
        self.NO_MEMORY_TOKENS = tokenizer.encode("No previous memory", add_special_tokens=False)
        # Rollout data directory for logging
        self.rollout_data_dir = None
        self.global_step = 0
    
    @override
    def start(self, gen_batch: DataProto, timing_raw: dict):
        self.gen_batch = gen_batch
        self.step = 0
        self.final_mask_list = [] # only the final turn will be verified, used for reward compute
        self.sample_index_list = [] # map each turn in final to the sample id in the original batch

        self.ctx_length = gen_batch.batch['context_length'] # if all context is used, then the sample will no more be active
        self.bsz = len(self.ctx_length)
        self.memory = np.empty(self.bsz, dtype=object)
        self.is_final = False
        # Per-step storage for memory recall
        self.step_input_memories = []  # memory available at each step's input
        self.step_chunks = []  # unpadded chunk tokens at each step
        # Get rollout_data_dir and global_step from meta_info
        self.rollout_data_dir = gen_batch.meta_info.get('rollout_data_dir', None)
        self.global_step = gen_batch.meta_info.get('global_step', 0)

        # Clean up old rollout files when starting a new experiment
        if self.rollout_data_dir:
            main_file = os.path.join(self.rollout_data_dir, f"main_{self.global_step}.txt")
            recall_file = os.path.join(self.rollout_data_dir, f"recall_{self.global_step}.txt")
            if os.path.exists(main_file):
                os.remove(main_file)
            if os.path.exists(recall_file):
                os.remove(recall_file)
    
    @override
    def action(self) -> Tuple[List[torch.Tensor], dict]:
        # suppose 0 is pad_token_id
        # max_chunks = 3, chunk_sieze = 2
        # pi is token in prompt, ti is token in chat template, 
        # [1,2] [3,4] [5,0] | p0 string
        # [1,2] [3,0] [0,0] | p1,p1 string
        # [1,0] [0,0] [0,0] | p2,p2,p2 string
        # -------- round 1 ---------
        # [1,2]            [t0,p0,t1, m,t2, 1, 2,t3]                           [ 0, 0, 0,t0,p0,t1, m,t2, 1, 2,t3]
        # [1,2]  -format-> [t0,p1,p1,t1, m,t2, 1, 2,t3] -pad2Dlist2Tendors->   [ 0, 0,t0,p1,p1,t1, m,t2, 1, 2,t3]
        # [1,0]            [t0,p2,p2,p3,t1, m,t2, 1,t3]                        [ 0, 0,t0,p2,p2,p3,t1, m,t2, 1,t3]
        # get mask & positionids
        active_mask = self.ctx_length > self.step * self.config.chunk_size
        self.active_mask = active_mask
        gen_batch = self.gen_batch
        # if all context is used, and its not done, then it will be the final turn for this batch
        if active_mask.sum().item() == 0:
            self.is_final = True
            self.messages = [
                self.token_final_message_template.format(
                    prompt=prompt,
                    memory=memory if memory is not None else self.NO_MEMORY_TOKENS,
                )
                for prompt, memory in zip(gen_batch.non_tensor_batch['prompt_ids'], self.memory)
            ]
            sample_index = torch.arange(self.bsz, dtype=torch.int)
            final_mask = torch.full(sample_index.shape, True, dtype=torch.bool) # all False
            self.meta_info = {'input_pad_to': self.max_input_length,
                         'pad_to': self.config.gen_pad_to,
                         'generation_kwargs': {
                          'max_tokens': self.config.gen_max_tokens_memorization,
                          'n': 1 # note that we have already repeat n times in ray_trainer
                        }}
            logger.info(f'FINAL TURN: MemoryAgent.next() done')
        else:
            # 1. no need to pad prompt
            # 2. context padded for 2D indexing, elegant engineering
            # 3. no need to pad memory
            prompt_i = gen_batch.non_tensor_batch['prompt_ids'][active_mask]
            chunk_i = gen_batch.batch['context_ids'][active_mask, self.config.chunk_size * self.step: self.config.chunk_size * (self.step+1)] # bs * chunk_size
            memory_i = self.memory[active_mask]

            # Store per-step data for memory recall
            unpadded_chunks = [chunk[chunk != self.tokenizer.pad_token_id] for chunk in chunk_i]
            chunk_storage = np.empty(self.bsz, dtype=object)
            active_indices = active_mask.nonzero(as_tuple=True)[0]
            for j, idx in enumerate(active_indices):
                chunk_storage[idx.item()] = unpadded_chunks[j]
            self.step_chunks.append(chunk_storage)

            input_mem_storage = np.empty(self.bsz, dtype=object)
            for j, idx in enumerate(active_indices):
                input_mem_storage[idx.item()] = memory_i[j] if memory_i[j] is not None else self.NO_MEMORY_TOKENS
            self.step_input_memories.append(input_mem_storage)
            
            # format: we use our token_template to avoid decoding & formatting with str function & encoding back.
            self.messages = [
                self.token_message_template.format(
                        prompt=prompt,
                        memory=memory if memory is not None else self.NO_MEMORY_TOKENS, # use pre-tokenized "No previous memory" for first round
                        chunk=chunk[chunk != self.tokenizer.pad_token_id], # unpadding needed here
                )
                for prompt, memory, chunk in zip(prompt_i, memory_i, chunk_i)
            ]
            sample_index = torch.arange(self.bsz, dtype=torch.long)[active_mask] # map active sample to original batch
            final_mask = torch.full(sample_index.shape, False, dtype=torch.bool) # all False
            self.meta_info = {'input_pad_to': self.max_input_length,
                         'pad_to': self.config.gen_pad_to,
                         'generation_kwargs': {
                          'max_tokens': self.config.gen_max_tokens_memorization,
                          'n': 1 # note that we have already repeat n times in ray_trainer
                        }}
            logger.info(f'MemoryAgent.action() done')
        self.final_mask_list.append(final_mask)
        self.sample_index_list.append(sample_index)
        return self.messages, self.meta_info

    @override
    def update(self, gen_output: DataProto) -> DataProto:
        if not self.is_final:
            self.memory[self.active_mask] = unpad(self.tokenizer, gen_output.batch['responses'], remove_eos=True)
        self.log_step(gen_output)
        self.step += 1
        return gen_output
    
    @override
    def done(self):
        return self.is_final
    
    @override
    def end(self):
        del self.gen_batch
        del self.ctx_length
        del self.meta_info
        del self.memory
        del self.messages
        sample_index = torch.cat(self.sample_index_list)
        final_mask = torch.cat(self.final_mask_list)
        del self.final_mask_list
        del self.sample_index_list
        return final_mask, sample_index

    def cleanup(self):
        """Clean up per-step data after recall is complete."""
        if hasattr(self, 'step_input_memories'):
            del self.step_input_memories
        if hasattr(self, 'step_chunks'):
            del self.step_chunks

    def recall(self, sample_indices: List[int]) -> Tuple[List[torch.Tensor], dict]:
        """Construct memory recall prompts for random steps.

        Args:
            sample_indices: list of original sample indices that survived filtering

        Returns:
            messages: list of token-ID tensors for each recall prompt
            meta_info: dict with generation kwargs, recall ground truths, etc.
        """
        num_steps = len(self.step_chunks)
        if num_steps == 0:
            return [], {}

        all_messages = []
        all_sample_indices = []
        all_ground_truths = []
        all_data_sources = []

        for i in sample_indices:
            # Find valid non-final steps for this sample
            valid_steps = [s for s in range(1, num_steps) if self.step_chunks[s][i] is not None and self.step_input_memories[s][i] != self.NO_MEMORY_TOKENS]
            if not valid_steps:
                continue

            # Randomly select one step k
            k = valid_steps[np.random.randint(len(valid_steps))]

            # Get input memory at step k
            memory = self.step_input_memories[k][i]

            # Randomly select section j where j <= k
            valid_js = [s for s in range(1, k + 1) if self.step_chunks[s][i] is not None]
            if not valid_js:
                continue

            j = valid_js[np.random.randint(len(valid_js))]

            # Get chunk at step j, decode to string, mask entity, re-encode
            chunk_tokens = self.step_chunks[j][i]
            chunk_str = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            # Use spaCy NER to extract and mask entity
            from recurrent.utils import get_random_entity_spacy
            selected_entity = get_random_entity_spacy(chunk_str)
            if not selected_entity:
                continue

            masked_str = re.sub(r'\b' + re.escape(selected_entity) + r'\b', '[TARGET]', chunk_str)
            masked_section_tokens = self.tokenizer.encode(masked_str, add_special_tokens=False)

            message = self.token_recall_message_template.format(
                memory=memory,
                maskedsection=masked_section_tokens,
            )
            all_messages.append(message)
            all_sample_indices.append(i)
            all_ground_truths.append([selected_entity])
            # Get data_source from gen_batch if still available, else default
            all_data_sources.append('hotpotqa')

        if not all_messages:
            return [], {}

        # Store recall messages for later logging
        self.recall_messages_decoded = []
        for msg in all_messages:
            self.recall_messages_decoded.append(self.tokenizer.decode(msg))

        meta_info = {
            'input_pad_to': self.max_input_length,
            'pad_to': self.config.gen_pad_to,
            'generation_kwargs': {
                'max_tokens': self.config.gen_max_tokens_final_response,
                'n': 1
            },
            'recall_sample_indices': all_sample_indices,
            'recall_ground_truths': all_ground_truths,
            'recall_data_sources': all_data_sources,
            'recall_messages_decoded': self.recall_messages_decoded,
        }

        return all_messages, meta_info
        

    def log_step(self, gen_output):
        """Log multi-turn conversation details for the first sample.
        Logs to both screen and file if rollout_data_dir is set.
        """
        def clip_long_string(string, max_length=2000):
            """Clip long string to a maximum length."""
            if not len(string) > max_length:
                return string
            return string[:max_length//2] + '\n\n...(ignored)\n\n' + string[-max_length//2:]

        # Header with dynamic step number
        step = self.step if not self.is_final else "FINAL"
        log_header = f"\n{'='*30}[RECURRENT] STEP{step}{'='*30}"
        logger.info(log_header)

        # Message and Response section (for first sample, index 0)
        # For final turn, active_mask is all False but we still want to log
        if self.is_final or self.active_mask[0]:
            decoded_message = self.tokenizer.decode(self.messages[0])
            rsp0 = gen_output.batch['responses'][0]
            decoded_response = self.tokenizer.decode(rsp0[rsp0!=self.tokenizer.pad_token_id])
            logger.info(f"[MESSAGE] {clip_long_string(decoded_message)}")
            logger.info(f"{' '*10}{'-'*20}prompt end{'-'*20}{' '*10}")
            logger.info(f"[RESPONSE] {decoded_response}")
            logger.info(f"{' '*10}{'-'*20}response end{'-'*20}{' '*10}")

            # Save to file if rollout_data_dir is set
            if self.rollout_data_dir:
                self._save_main_rollout(decoded_message, decoded_response, step)
        else:
            logger.info("MESSAGE and RESPONSE is empty since it is not active.")

    def _save_main_rollout(self, message: str, response: str, step):
        """Save main trajectory rollout for sample 0 to file."""
        main_file = os.path.join(self.rollout_data_dir, f"main_{self.global_step}.txt")
        os.makedirs(os.path.dirname(main_file), exist_ok=True)
        with open(main_file, "a") as f:
            f.write(f"[TURN {step}]\n")
            f.write(f"{'='*50}\n")
            f.write(f"[MESSAGE]\n{message}\n\n")
            f.write(f"[RESPONSE]\n{response}\n\n")

    def log_recall(self, message: str, response: str, ground_truth: str, sample_idx: int):
        """Log recall trajectory.
        Logs to both screen and file if rollout_data_dir is set.
        """
        def clip_long_string(string, max_length=2000):
            if not len(string) > max_length:
                return string
            return string[:max_length//2] + '\n\n...(ignored)\n\n' + string[-max_length//2:]

        log_header = f"\n{'='*30}[RECALL] Sample {sample_idx}{'='*30}"
        logger.info(log_header)
        logger.info(f"[MESSAGE] {clip_long_string(message)}")
        logger.info(f"{' '*10}{'-'*20}prompt end{'-'*20}{' '*10}")
        logger.info(f"[RESPONSE] {response}")
        logger.info(f"[GROUND_TRUTH] {ground_truth}")
        logger.info(f"{' '*10}{'-'*20}response end{'-'*20}{' '*10}")

        # Save to file if rollout_data_dir is set
        if self.rollout_data_dir:
            recall_file = os.path.join(self.rollout_data_dir, f"recall_{self.global_step}.txt")
            os.makedirs(os.path.dirname(recall_file), exist_ok=True)
            with open(recall_file, "w") as f:
                f.write(f"{'='*50}\n")
                f.write(f"Step: {self.global_step}, Sample: {sample_idx}\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"[MESSAGE]\n{message}\n\n")
                f.write(f"[RESPONSE]\n{response}\n\n")
                f.write(f"[GROUND_TRUTH]\n{ground_truth}\n")


# Important, we will import `REGISTER` from this file to get all registered classes.
# specified by recurrent.path / recurrent.name(defaults to REGISTER)
REGISTER = RRegister(config_cls=MemoryConfig, dataset_cls=MemoryDataset, agent_cls=MemoryAgent)
