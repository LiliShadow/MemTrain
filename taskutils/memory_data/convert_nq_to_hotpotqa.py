#!/usr/bin/env python3
"""
Convert nq_hotpotqa_train_multi_2 format to hotpotqa_train_32k format.

Source format (nq_hotpotqa_train_multi_2):
- Columns: data_source, prompt, ability, reward_model, extra_info
- prompt: Contains multi-question prompts with system instructions
- reward_model: Has ground_truth with multiple targets

Target format (hotpotqa_train_32k):
- Columns: data_source, prompt, context, ability, reward_model, extra_info,
           score_boxed_pretrain, score_contains_pretrain, score_boxed_instruct, score_contains_instruct
- prompt: Simple single question
- context: Document text
- reward_model: Single ground_truth
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import re

def extract_question_from_prompt(prompt_list):
    """Extract the actual question from the complex multi-question prompt."""
    if isinstance(prompt_list, list) and len(prompt_list) > 0:
        # Get the user message content
        for msg in prompt_list:
            if isinstance(msg, dict) and msg.get('role') == 'user':
                content = msg.get('content', '')
                # Extract the question after "Answer the following questions:"
                match = re.search(r'Answer the following questions?:\s*(.+?)(?:<\|im_end\|>|$)', content, re.DOTALL)
                if match:
                    questions = match.group(1).strip()
                    # If multiple questions separated by semicolon, take first one
                    if ';' in questions:
                        return questions.split(';')[0].strip()
                    return questions
    return None

def extract_single_target(ground_truth, index=0):
    """Extract a single target from multi-target ground truth."""
    if isinstance(ground_truth, dict) and 'target' in ground_truth:
        targets = ground_truth['target']
        if isinstance(targets, list) and len(targets) > index:
            target = targets[index]
            # If target is a list, take the first element
            if isinstance(target, list):
                return target[0] if len(target) > 0 else None
            return target
    return None

def create_context_from_prompt(prompt_list):
    """Create placeholder context - in real scenario would retrieve documents."""
    # For now, return empty string - context would normally be retrieved
    return ""

def convert_row(row):
    """Convert a single row from source to target format."""
    # Extract question from complex prompt
    question = extract_question_from_prompt(row['prompt'])

    # If we can't extract question, use a placeholder
    if not question:
        question = "What is the answer?"

    # Create simple prompt format
    new_prompt = [{'content': question, 'role': 'user'}]

    # Extract single target (use first one)
    target = extract_single_target(row['reward_model'], index=0)

    # Create new reward_model format
    new_reward_model = {
        'ground_truth': np.array([target] if target else [], dtype=object),
        'style': 'rule'
    }

    # Create extra_info
    new_extra_info = {
        'index': row['extra_info'].get('indices', [0])[0] if isinstance(row['extra_info'], dict) else 0,
        'num_docs': 0,
        'question': question
    }

    return {
        'data_source': 'nq',  # or could be 'hotpotqa' depending on source
        'prompt': new_prompt,
        'context': create_context_from_prompt(row['prompt']),
        'ability': 'memory',  # Changed from 'fact-reasoning' to 'memory'
        'reward_model': new_reward_model,
        'extra_info': new_extra_info,
        'score_boxed_pretrain': [0.0, 0.0],
        'score_contains_pretrain': [False, False],
        'score_boxed_instruct': [0.0, 0.0],
        'score_contains_instruct': [False, False]
    }

def main():
    # Read source file
    print("Loading source file...")
    source_df = pd.read_parquet('taskutils/memory_data/nq_hotpotqa_train_multi_2/train.parquet')
    print(f"Source shape: {source_df.shape}")

    # Convert each row
    print("Converting rows...")
    converted_rows = []
    for idx, row in tqdm(source_df.iterrows(), total=len(source_df)):
        converted_rows.append(convert_row(row))

    # Create new dataframe
    print("Creating target dataframe...")
    target_df = pd.DataFrame(converted_rows)

    # Save to target location
    output_path = 'taskutils/memory_data/hotpotqa/hotpotqa_train_from_nq.parquet'
    print(f"Saving to {output_path}...")
    target_df.to_parquet(output_path, index=False)

    print(f"Done! Output shape: {target_df.shape}")
    print(f"\nSample output:")
    print(f"data_source: {target_df.iloc[0]['data_source']}")
    print(f"prompt: {target_df.iloc[0]['prompt']}")
    print(f"context length: {len(target_df.iloc[0]['context'])}")
    print(f"ability: {target_df.iloc[0]['ability']}")
    print(f"reward_model: {target_df.iloc[0]['reward_model']}")
    print(f"extra_info: {target_df.iloc[0]['extra_info']}")

if __name__ == '__main__':
    main()
