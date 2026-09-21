"""
Semantic-similarity-based preprocessing for mask dataset.
Instead of randomly selecting 150 paragraphs, uses pre-built FAISS index
to retrieve the 149 most semantically similar paragraphs to a random anchor.

Requires: pre-built FAISS index from build_embedding_index.py

Usage:
    python preprocess_mask_dataset_semantic.py \
        --input_file data/wikipedia_en_qwen_keys_30w_filtered_v2.jsonl \
        --index_dir data/embedding_index \
        --output_file data/mask_pretrain_150docs_v2.parquet \
        --num_samples 30000 \
        --num_docs 150
"""

import argparse
import json
import os
import random
import re
import sys
from typing import List

import numpy as np
import faiss
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer

# Path to an external repo providing recipe.dapo.utils (set PTQ_ROOT to override)
sys.path.insert(0, os.environ.get("PTQ_ROOT", "ptq"))
from recipe.dapo.utils import get_random_entity_spacy

MASK = "[MASK]"

# QUESTION_TEMPLATE = (
#     f"What is the word replaced by {MASK} in the article?"
# )

QUESTION_TEMPLATE = (
    f"Based on the entire article, which entity is represented by the {MASK} placeholder?"
)


def mask_in_context(context: str, mask: str) -> str:
    pattern = r"\b" + re.escape(mask) + r"\b"
    return re.sub(pattern, MASK, context)


def build_context_from_paragraphs(paragraphs: List[str]) -> str:
    parts = []
    for i, para in enumerate(paragraphs, 1):
        parts.append(f"Document {i}:\n{para}")
    return "\n\n".join(parts)


def batch_faiss_search(
    num_samples: int,
    embeddings: np.ndarray,
    faiss_index,
    num_docs: int,
    seed: int,
    num_retrieved: int = 30,
):
    """Run all FAISS searches in the main process and return retrieval results.

    Strategy: Mix top-`num_retrieved` similar documents with randomly selected documents.

    Returns a list of (anchor_idx, selected_indices) tuples, picklable for workers.
    """
    rng = random.Random(seed)
    total_docs = len(embeddings)
    anchor_indices = [rng.randint(0, total_docs - 1) for _ in range(num_samples)]

    # Batch search: collect all anchor embeddings, search at once
    # Retrieve top num_retrieved + 1 to account for anchor potentially being in results
    anchor_embs = np.array(
        [embeddings[aidx] for aidx in anchor_indices], dtype=np.float32
    )
    distances, indices = faiss_index.search(anchor_embs, num_retrieved + 1)

    results = []
    for i, anchor_idx in enumerate(anchor_indices):
        # Get top-k retrieved documents (excluding anchor)
        retrieved = indices[i].tolist()
        if retrieved[0] == anchor_idx:
            similar = retrieved[1:num_retrieved + 1]
        else:
            similar = [idx for idx in retrieved if idx != anchor_idx][:num_retrieved]

        # Randomly select remaining documents
        num_random = num_docs - len(similar) - 1  # -1 for anchor
        available_indices = set(range(total_docs)) - set(similar) - {anchor_idx}
        random_indices = rng.sample(list(available_indices), num_random)

        # Combine and shuffle: anchor + retrieved + random
        selected = [anchor_idx] + similar + random_indices
        rng.shuffle(selected)
        results.append((anchor_idx, selected))

    return results


def process_sample_from_retrieval(
    item_idx: int,
    all_paragraphs: List[str],
    anchor_idx: int,
    selected_indices: List[int],
) -> dict:
    anchor_para = all_paragraphs[anchor_idx]

    mask = get_random_entity_spacy(anchor_para)
    if not mask:
        return None

    selected_paragraphs = [all_paragraphs[i] for i in selected_indices]
    context = build_context_from_paragraphs(selected_paragraphs)
    masked_context = mask_in_context(context, mask)

    question = QUESTION_TEMPLATE
    prompt = [{"content": question, "role": "user"}]

    return {
        "data_source": "hotpotqa",
        "prompt": prompt,
        "context": masked_context,
        "ability": "memory",
        "reward_model": {
            "ground_truth": [mask],
            "style": "rule",
        },
        "extra_info": {
            "index": item_idx,
            "num_docs": len(selected_indices),
            "anchor_idx": anchor_idx,
            "retrieved_indices": selected_indices,
            "question": question,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Semantic-similarity-based mask dataset preprocessing"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/wikipedia_en_qwen_keys_30w_filtered_v2.jsonl",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default="data/embedding_index",
        help="Directory with pre-built FAISS index (from build_embedding_index.py)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/mask_pretrain_150docs_v2.parquet",
    )
    parser.add_argument("--num_samples", type=int, default=30000)
    parser.add_argument("--num_docs", type=int, default=150)
    parser.add_argument(
        "--num_retrieved",
        type=int,
        default=29,
        help="Number of top retrieved documents to include (remaining will be random)",
    )
    parser.add_argument(
        "--faiss_gpu",
        action="store_true",
        default=False,
        help="Use GPU for FAISS search (requires faiss-gpu)",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="models/Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=32)
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.input_file}...")
    ds = load_dataset("json", data_files=args.input_file, split="train")
    all_paragraphs = ds["paragraph"]
    print(f"Total paragraphs: {len(all_paragraphs)}")

    # Load FAISS index and embeddings
    print(f"Loading FAISS index from {args.index_dir}...")
    metadata_path = os.path.join(args.index_dir, "metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)

    faiss_index = faiss.read_index(os.path.join(args.index_dir, "index.faiss"))
    if args.faiss_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        co.shard = True
        faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co=co)
        print("FAISS index moved to GPU")

    embeddings = np.memmap(
        os.path.join(args.index_dir, "embeddings.memmap"),
        dtype=np.float32,
        mode="r",
        shape=(metadata["total_size"], metadata["dim"]),
    )
    print(f"Index loaded: {metadata['total_size']} vectors, dim={metadata['dim']}")

    assert len(all_paragraphs) == metadata["total_size"], (
        f"Dataset size ({len(all_paragraphs)}) != index size ({metadata['total_size']}). "
        f"Ensure the input file matches the one used to build the index."
    )

    # Batch FAISS search in main process (avoids pickle issues with mmap/FAISS)
    print(f"Running FAISS search for {args.num_samples} samples...")
    print(f"Strategy: {args.num_retrieved} retrieved + {args.num_docs - args.num_retrieved - 1} random documents")
    retrieval_results = batch_faiss_search(
        args.num_samples, embeddings, faiss_index, args.num_docs, args.seed,
        num_retrieved=args.num_retrieved,
    )
    # Free FAISS and mmap — no longer needed
    del faiss_index, embeddings

    # Build dataset from retrieval results (pure Python data, safe for pickle)
    anchor_indices_list = [r[0] for r in retrieval_results]
    selected_indices_list = [r[1] for r in retrieval_results]

    input_ds = HFDataset.from_dict({
        "item_idx": list(range(args.num_samples)),
        "anchor_idx": anchor_indices_list,
        "selected_indices": selected_indices_list,
    })

    def process_fn(batch):
        results = {
            "data_source": [],
            "prompt": [],
            "context": [],
            "ability": [],
            "reward_model": [],
            "extra_info": [],
        }
        for idx, anchor_idx, sel_indices in zip(
            batch["item_idx"], batch["anchor_idx"], batch["selected_indices"]
        ):
            sample = process_sample_from_retrieval(
                item_idx=idx,
                all_paragraphs=all_paragraphs,
                anchor_idx=anchor_idx,
                selected_indices=sel_indices,
            )
            if sample is not None:
                results["data_source"].append(sample["data_source"])
                results["prompt"].append(sample["prompt"])
                results["context"].append(sample["context"])
                results["ability"].append(sample["ability"])
                results["reward_model"].append(sample["reward_model"])
                results["extra_info"].append(sample["extra_info"])
        return results

    print(f"Processing {args.num_samples} samples with {args.num_workers} workers...")
    result_ds = input_ds.map(
        process_fn,
        batched=True,
        batch_size=64,
        num_proc=args.num_workers,
        desc="Processing samples",
        remove_columns=["item_idx", "anchor_idx", "selected_indices"],
    )

    print(f"Generated {len(result_ds)} samples")

    # Save as parquet
    print(f"Saving to {args.output_file}...")
    result_ds.to_parquet(args.output_file)
    print(f"Saved {len(result_ds)} samples to {args.output_file}")

    # Token statistics
    print(f"\nLoading tokenizer from {args.tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )

    def count_tokens(batch):
        counts = []
        for context in batch["context"]:
            tokens = tokenizer.encode(context, add_special_tokens=False)
            counts.append(len(tokens))
        return {"token_count": counts}

    result_ds = result_ds.map(
        count_tokens,
        batched=True,
        batch_size=100,
        num_proc=args.num_workers,
        desc="Counting tokens",
    )

    token_counts = result_ds["token_count"]
    avg_tokens = np.mean(token_counts)
    median_tokens = np.median(token_counts)
    min_tokens = np.min(token_counts)
    max_tokens = np.max(token_counts)
    p90_tokens = np.percentile(token_counts, 90)
    p95_tokens = np.percentile(token_counts, 95)

    print(f"\n{'='*50}")
    print(f"Context Token Statistics:")
    print(f"  Average:   {avg_tokens:.1f}")
    print(f"  Median:    {median_tokens:.1f}")
    print(f"  Min:       {min_tokens}")
    print(f"  Max:       {max_tokens}")
    print(f"  P90:       {p90_tokens:.1f}")
    print(f"  P95:       {p95_tokens:.1f}")
    print(f"  Total samples: {len(token_counts)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
