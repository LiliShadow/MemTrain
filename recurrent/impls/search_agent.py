# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Search Agent Implementation for Multi-Turn Think-Search with GRPO

This module implements a multi-turn search agent that:
1. Generates thoughts in ``` blocks
2. Executes search actions via external API
3. Receives retrieved information
4. Iterates until providing final answer
5. Supports recall mechanism for entity prediction

Adapted from MEM1's generation_think.py for MemAgent's framework.
"""

import logging
import os
import re
import requests
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin
from typing_extensions import override

from recurrent.interface import RAgent, RConfig, RDataset, RRegister
from recurrent.utils import TokenTemplate, chat_template, get_random_entity_spacy
from verl.protocol import DataProto

logger = logging.getLogger(__file__)
logger.setLevel('INFO')


@dataclass
class SearchAgentConfig(RConfig):
    """Configuration for multi-turn search agent - matching MEM1's GenerationConfig"""
    max_turns: int = 3
    max_start_length: int = 2048  # Initial question/input length
    max_prompt_length: int = 4096  # Maximum context length
    max_response_length: int = 1000  # Total response length (think + action)
    max_obs_length: int = 1000  # Retrieved information length
    search_url: str = "http://localhost:8000/search"
    topk: int = 3

    @property
    def gen_max_tokens_response(self):
        return self.max_response_length

    @property
    def gen_pad_to(self):
        return self.max_response_length


class SearchAgentDataset(RDataset):
    """Dataset for search agent tasks"""

    def __init__(
        self,
        recurrent_config: SearchAgentConfig,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        data_config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        super().__init__(
            recurrent_config=recurrent_config,
            data_files=data_files,
            tokenizer=tokenizer,
            data_config=data_config,
            processor=processor
        )
        self.recurrent_config = recurrent_config

    @override
    def __getitem__(self, item):
        row_dict = super().__getitem__(item)
        # Add UUID for GRPO grouping (shared across all turns from same question)
        row_dict["sample_uuid"] = str(uuid4())
        return row_dict


# Template for recall mechanism (following memory.py pattern)
TEMPLATE_RECALL = """You are presented with a thinking process and a section containing [TARGET]. Based on the thinking, please predict the entity marked by [TARGET] and put the answer in \\boxed{{}}.

<thinking>
{thinking}
</thinking>

<section>
{masked_source}
</section>

Your answer:
"""


class SearchAgent(RAgent):
    """
    Multi-turn search agent implementing think-search-answer loop.

    This agent:
    1. Receives pre-formatted prompts from dataset (MEM1 pattern)
    2. Generates thoughts in ``` blocks
    3. Executes search or answer actions
    4. Processes retrieved information
    5. Splits trajectories into separate samples for GRPO
    6. Masks information tokens from loss computation
    7. Supports recall mechanism for entity prediction

    CRITICAL: Does NOT modify gen_batch.batch (like memory agent).
    Stores context externally and constructs messages fresh each turn.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, config: SearchAgentConfig):
        self.config = config
        self.tokenizer = tokenizer
        # Chat template for recall mechanism
        self.chat_template = chat_template(tokenizer)

        # Recall template - will be initialized if recall is enabled globally
        self.recall_template = None

        logger.info(f'[SEARCH_AGENT] Initialized with max_turns={config.max_turns}, '
                   f'search_url={config.search_url}')

    @override
    def start(self, gen_batch: DataProto, timing_raw: dict):
        """Initialize agent state for new generation batch"""
        self.gen_batch = gen_batch
        self.timing_raw = timing_raw
        self.step = 0

        # Batch size
        self.bsz = gen_batch.batch['input_ids'].shape[0]

        # UID for GRPO grouping (all turns from same question share UID)
        if 'sample_uuid' in gen_batch.non_tensor_batch:
            self.uids = gen_batch.non_tensor_batch['sample_uuid']
        else:
            # Generate UIDs if not present
            self.uids = [str(uuid4()) for _ in range(self.bsz)]

        # Track which samples are still active
        self.active_mask = torch.ones(self.bsz, dtype=torch.bool)

        # Track turn indices
        self.turn_indices = torch.zeros(self.bsz, dtype=torch.long)

        # Storage for trajectory segments (for splitting into separate samples)
        self.trajectory_segments = []

        # Storage for recall mechanism: list[turn][sample] = tokens
        self.turn_thinkings = []  # Per-turn thinking tokens
        self.turn_infos = []  # Per-turn information tokens

        # Store original questions (immutable) - EXTERNAL storage
        self.original_questions = [None] * self.bsz
        for i in range(self.bsz):
            input_ids = gen_batch.batch['input_ids'][i]
            attention_mask = gen_batch.batch['attention_mask'][i]
            valid_length = attention_mask.sum().item()
            self.original_questions[i] = input_ids[-valid_length:].clone()

        # Accumulated context for each sample - EXTERNAL storage (like memory agent)
        self.accumulated_context = [None] * self.bsz

        # Initialize recall template if recall is enabled globally
        # Check meta_info for algorithm config (passed from ray_trainer)
        enable_recall = gen_batch.meta_info.get('enable_memory_recall', False)
        if enable_recall and self.recall_template is None:
            self.recall_template = TokenTemplate(
                self.chat_template.format(message=TEMPLATE_RECALL),
                self.tokenizer
            )
            logger.info(f'[SEARCH_AGENT] Recall enabled')

        logger.info(f'[SEARCH_AGENT] Started batch with size {self.bsz}')

    @override
    def action(self) -> Tuple[List[torch.Tensor], dict]:
        """
        Return prompts for current turn.

        CRITICAL: Constructs messages FRESH each turn from external storage.
        Does NOT read from gen_batch.batch (which is read-only).
        """
        # Get active samples
        active_indices = self.active_mask.nonzero(as_tuple=True)[0]

        # For each active sample, construct message from original question + accumulated context
        messages = []
        for idx in active_indices:
            idx_val = idx.item()
            question = self.original_questions[idx_val]
            context = self.accumulated_context[idx_val]

            if context is None:
                # First turn: just the question
                input_ids = question
            else:
                # Subsequent turns: question + accumulated context (responses + info from previous turns)
                input_ids = torch.cat([question, context])

            # CRITICAL: Truncate to max_prompt_length to avoid exceeding model context limit
            if len(input_ids) > self.config.max_prompt_length:
                logger.warning(f'[SEARCH_AGENT] Truncating prompt from {len(input_ids)} to {self.config.max_prompt_length}')
                input_ids = input_ids[-self.config.max_prompt_length:]

            messages.append(input_ids)

        # Store messages for logging
        self.messages = messages

        # Meta info for generation
        meta_info = {
            'input_pad_to': self.config.max_prompt_length,  # Pad to max context length
            'pad_to': self.config.gen_pad_to,
            'generation_kwargs': {
                'max_tokens': self.config.max_response_length,
                'n': 1
            }
        }

        return messages, meta_info

    def log_step(self, gen_output: DataProto):
        """Log multi-turn conversation details for the first sample.
        Shows prompt structure clearly for search agent.
        """
        # Header with dynamic step number
        log_header = f"\n{'='*30}[RECURRENT] STEP{self.step}{'='*30}"
        logger.info(log_header)

        # Message and Response section (for first sample, index 0)
        if self.active_mask[0]:
            idx_val = 0
            question = self.original_questions[idx_val]
            context = self.accumulated_context[idx_val]

            # Decode question (show first 500 chars)
            question_str = self.tokenizer.decode(question)
            if len(question_str) > 500:
                question_str = question_str[:250] + "...\n" + question_str[-250:]

            # Decode context (previous response + info) - show MORE for context
            if context is not None:
                context_str = self.tokenizer.decode(context, skip_special_tokens=True)
                # Don't clip context - show full thinking and action
                logger.info(f"[MESSAGE]\n{question_str}\n{context_str}")
            else:
                logger.info(f"[MESSAGE]\n{question_str}")

            logger.info(f"{' '*10}{'-'*20}prompt end{'-'*20}{' '*10}")

            # Decode response
            rsp0 = gen_output.batch['responses'][0]
            decoded_response = self.tokenizer.decode(rsp0[rsp0 != self.tokenizer.pad_token_id])
            logger.info(f"[RESPONSE] {decoded_response}")
            logger.info(f"{' '*10}{'-'*20}response end{'-'*20}{' '*10}")
        else:
            logger.info("MESSAGE and RESPONSE is empty since sample 0 is not active.")

    @override
    def update(self, gen_output: DataProto) -> DataProto:
        """
        Process generated responses, execute search if needed, prepare next turn.

        CRITICAL: Does NOT modify gen_output or gen_batch.
        Updates EXTERNAL storage (accumulated_context) for next turn.
        """
        # Get responses for active samples
        responses_ids = gen_output.batch['responses']

        # Decode responses
        responses_str = self.tokenizer.batch_decode(responses_ids, skip_special_tokens=True)

        # Parse responses and execute actions
        active_indices = self.active_mask.nonzero(as_tuple=True)[0]

        # Track which samples are done
        new_done_mask = torch.zeros(self.bsz, dtype=torch.bool)

        # Initialize storage for this turn (indexed by sample)
        turn_thinkings_per_sample = [None] * self.bsz
        turn_infos_per_sample = [None] * self.bsz

        # STEP 1: Parse all actions first (CRITICAL - batch search pattern)
        actions = []
        search_queries = []

        for i, idx in enumerate(active_indices):
            idx_val = idx.item()
            response_str = responses_str[i]
            response_ids = responses_ids[i]

            # Extract think and response/action
            think_ids, action_ids = self._extract_think_and_response(response_ids)
            action_type, action_content = self._parse_action(response_str)

            actions.append({
                'idx': idx_val,
                'think_ids': think_ids,
                'action_ids': action_ids,
                'action_type': action_type,
                'action_content': action_content,
                'response_ids': response_ids
            })

            # Collect search queries for BATCH execution
            if action_type == 'search':
                search_queries.append(action_content)

        # STEP 2: Execute ALL searches in ONE batch call (MEM1 pattern)
        if search_queries:
            search_results = self._batch_search(search_queries)
        else:
            search_results = []

        # STEP 3: Process each response and distribute search results
        search_result_idx = 0
        for i, action_data in enumerate(actions):
            idx_val = action_data['idx']
            think_ids = action_data['think_ids']
            action_ids = action_data['action_ids']
            action_type = action_data['action_type']
            action_content = action_data['action_content']
            response_ids = action_data['response_ids']

            # Determine if done (sample terminates if not search OR reached max turns)
            is_done = (action_type != 'search') or (self.step >= self.config.max_turns - 1)

            # Store turn data for trajectory splitting
            turn_data = {
                'question': self.original_questions[idx_val],
                'think': think_ids,
                'action': action_ids,
                'uid': self.uids[idx_val],
                'turn_idx': self.turn_indices[idx_val].item(),
                'is_final': is_done
            }

            # Execute search if needed (only if search action AND continuing)
            if action_type == 'search' and not is_done:
                # Get search result (distributed from batch)
                search_result = search_results[search_result_idx]
                search_result_idx += 1

                # Generate hint (MEM1 pattern)
                hint = self._get_hint(self.turn_indices[idx_val].item())

                # Insert hint INSIDE <information> tags (MEM1 line 607)
                info_str = f"\n\n<information>\n{hint}\n\n{search_result.strip()}\n</information>\n\n"
                info_ids = self.tokenizer.encode(info_str, add_special_tokens=False)
                info_ids_tensor = torch.tensor(info_ids, dtype=torch.long)

                # Truncate info if too long (MEM1 pattern)
                info_ids_tensor = self._process_next_obs(info_ids_tensor)

                turn_data['info'] = info_ids_tensor

                # Store for recall: turn_thinkings[turn][sample] = tokens
                turn_thinkings_per_sample[idx_val] = think_ids
                turn_infos_per_sample[idx_val] = info_ids_tensor

                # Update accumulated context for next turn (EXTERNAL storage)
                # CRITICAL: Only keep the MOST RECENT turn, not all previous turns
                # Prompt structure: [question] + [thinking t-1] [response t-1] + [info t-1]
                self.accumulated_context[idx_val] = torch.cat([response_ids, info_ids_tensor])
            else:
                turn_data['info'] = None
                # Store thinking even if no info (for recall)
                turn_thinkings_per_sample[idx_val] = think_ids
                new_done_mask[idx_val] = True

            self.trajectory_segments.append(turn_data)

        # STEP 4: Store this turn's data (indexed by sample) to turn history
        self.turn_thinkings.append(turn_thinkings_per_sample)
        self.turn_infos.append(turn_infos_per_sample)

        # Log step BEFORE updating active_mask (so samples are still marked active for current step)
        self.log_step(gen_output)

        # Update active mask
        self.active_mask = self.active_mask & ~new_done_mask

        # Increment turn indices for active samples
        self.turn_indices[self.active_mask] += 1

        # Increment step
        self.step += 1

        logger.info(f'[SEARCH_AGENT] Updated: {new_done_mask.sum().item()} samples done, '
                   f'{self.active_mask.sum().item()} still active')

        # Return gen_output UNCHANGED - it will be concatenated later
        return gen_output

    @override
    def done(self):
        """Check if all samples have terminated"""
        return not self.active_mask.any()

    @override
    def end(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return final_mask and sample_index for GRPO.

        Each turn becomes a separate sample in the batch.
        All turns from same question share same sample_index for GRPO grouping.
        """
        # Map string UIDs to integer indices for GRPO
        unique_uids = list(dict.fromkeys(segment['uid'] for segment in self.trajectory_segments))
        uid_to_index = {uid: idx for idx, uid in enumerate(unique_uids)}

        final_mask_list = []
        sample_index_list = []

        for segment in self.trajectory_segments:
            sample_index_list.append(uid_to_index[segment['uid']])
            final_mask_list.append(segment['is_final'])

        final_mask = torch.tensor(final_mask_list, dtype=torch.bool)
        sample_index = torch.tensor(sample_index_list, dtype=torch.long)

        # Cleanup
        del self.gen_batch
        del self.active_mask
        del self.turn_indices
        del self.original_questions
        del self.accumulated_context

        logger.info(f'[SEARCH_AGENT] Ended: {len(self.trajectory_segments)} total turns from {self.bsz} questions')

        return final_mask, sample_index

    def recall(self, sample_indices: List[int]) -> Tuple[List[torch.Tensor], dict]:
        """
        Construct recall prompts for entity prediction.

        Given thinking at turn t, randomly select thinking or info from before t,
        mask an entity with [TARGET], and let model predict.

        Following memory.py pattern exactly.
        """
        if self.recall_template is None:
            return [], {}

        num_turns = len(self.turn_thinkings)
        if num_turns == 0:
            return [], {}

        all_messages = []
        all_sample_indices = []
        all_ground_truths = []

        for i in sample_indices:
            # Find valid turns with thinking (turn 1 onwards)
            valid_turns = [t for t in range(1, num_turns)
                          if self.turn_thinkings[t][i] is not None]
            if not valid_turns:
                continue

            # Randomly select turn t
            t = valid_turns[np.random.randint(len(valid_turns))]

            # Get thinking at turn t
            thinking = self.turn_thinkings[t][i]

            # Randomly select thinking or info from before t
            valid_sources = []
            for s in range(t):
                if self.turn_thinkings[s][i] is not None:
                    valid_sources.append(('thinking', s))
                if self.turn_infos[s][i] is not None:
                    valid_sources.append(('info', s))

            if not valid_sources:
                continue

            source_type, j = valid_sources[np.random.randint(len(valid_sources))]

            if source_type == 'thinking':
                source_tokens = self.turn_thinkings[j][i]
            else:
                source_tokens = self.turn_infos[j][i]

            # Decode to string
            source_str = self.tokenizer.decode(source_tokens, skip_special_tokens=True)

            # Use spaCy NER to extract and mask entity
            selected_entity = get_random_entity_spacy(source_str)

            if not selected_entity:
                continue

            # Mask entity with [TARGET]
            masked_str = re.sub(r'\b' + re.escape(selected_entity) + r'\b', '[TARGET]', source_str)
            masked_tokens = self.tokenizer.encode(masked_str, add_special_tokens=False)

            # Format recall message using template
            message = self.recall_template.format(
                thinking=thinking,
                masked_source=masked_tokens,
            )

            all_messages.append(message)
            all_sample_indices.append(i)
            all_ground_truths.append([selected_entity])

        if not all_messages:
            return [], {}

        meta_info = {
            'input_pad_to': self.config.max_obs_length,
            'pad_to': self.config.gen_pad_to,
            'generation_kwargs': {
                'max_tokens': self.config.max_response_length,
                'n': 1
            },
            'recall_sample_indices': all_sample_indices,
            'recall_ground_truths': all_ground_truths,
        }

        logger.info(f'[SEARCH_AGENT] Recall: generated {len(all_messages)} recall prompts')

        return all_messages, meta_info

    # Helper methods

    def _extract_think_and_response(self, response_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract think and response from response ids - MEM1 pattern"""
        response_str = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Think block is in ```...``` markers
        if '```' in response_str:
            parts = response_str.split('```')
            think_str = parts[0] + '```'
            response_str = parts[1] if len(parts) > 1 else ''
        else:
            think_str = ''
            # response_str remains unchanged

        think_ids = self.tokenizer(think_str, add_special_tokens=False, return_tensors='pt')['input_ids'].squeeze(0)
        response_ids = self.tokenizer(response_str, add_special_tokens=False, return_tensors='pt')['input_ids'].squeeze(0)

        return think_ids, response_ids

    def _parse_action(self, response_str: str) -> Tuple[Optional[str], str]:
        """Parse action from response - MEM1 pattern"""
        # Extract <search> or <answer> tags
        pattern = r'<(answer|search)>(.*?)</\1>'
        match = re.search(pattern, response_str, re.DOTALL)

        if match:
            action_type = match.group(1).strip()  # 'search' or 'answer'
            action_content = match.group(2).strip()
            return action_type, action_content
        else:
            return None, ""  # Invalid format

    def _get_hint(self, cur_step: int) -> str:
        """Generate hint based on turns remaining - MEM1 pattern (lines 300-306)"""
        turns_left = self.config.max_turns - cur_step - 1
        if turns_left > 1:
            return f"[HINT]You have {turns_left} turns left.[/HINT]"
        elif turns_left == 1:
            return f"[HINT]You have 1 turn left. You must answer the question in the next turn.[/HINT]"
        else:
            return ""

    def _batch_search(self, queries: List[str]) -> List[str]:
        """Batch search using MEM1's retrieval API"""
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        try:
            response = requests.post(self.config.search_url, json=payload, timeout=30)
            results = response.json()['result']

            # Format results
            formatted_results = []
            for result in results:
                formatted = self._format_search_result(result)
                formatted_results.append(formatted)

            return formatted_results
        except Exception as e:
            logger.warning(f'[SEARCH_AGENT] Search API error: {e}')
            return ["<information>Search unavailable</information>"] * len(queries)

    def _format_search_result(self, result: List[dict]) -> str:
        """Format search result as information block (matching MEM1 pattern)"""
        format_reference = ''
        for idx, doc_item in enumerate(result):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference

    def _process_next_obs(self, next_obs_ids: torch.Tensor) -> torch.Tensor:
        """Truncate observation if too long (matching MEM1 pattern)"""
        if next_obs_ids.shape[0] > self.config.max_obs_length:
            next_obs_ids = next_obs_ids[:self.config.max_obs_length]

        return next_obs_ids


# Register implementation
REGISTER = RRegister(
    config_cls=SearchAgentConfig,
    dataset_cls=SearchAgentDataset,
    agent_cls=SearchAgent
)
