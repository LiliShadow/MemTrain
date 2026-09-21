"""
Multi-GPU embedding computation and FAISS index building for Wikipedia paragraphs.
Uses torchrun + torch.distributed for efficient 8-GPU parallel encoding.

Usage:
    torchrun --standalone --nnodes=1 --nproc_per_node=8 build_embedding_index.py \
        --input_file /path/to/paragraphs.jsonl \
        --save_dir /path/to/output \
        --model_name BAAI/bge-base-en-v1.5 \
        --batch_size 256 \
        --max_length 512

Outputs:
    {save_dir}/embeddings.memmap   - (N, 768) float32 memmap
    {save_dir}/index.faiss         - FAISS IndexFlatIP index
    {save_dir}/metadata.json       - corpus size, dim, model name
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.distributed as dist
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


@torch.no_grad()
def encode_shard(
    model,
    tokenizer,
    texts,
    batch_size,
    max_length,
    device,
):
    model.eval()
    all_embeddings = []

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc=f"Rank {dist.get_rank()}",
        disable=dist.get_rank() != 0,
    ):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        ).to(device)

        outputs = model(**inputs, return_dict=True)
        # BGE uses CLS pooling
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        embeddings = embeddings.cpu().numpy().astype(np.float32)
        all_embeddings.append(embeddings)

    return np.concatenate(all_embeddings, axis=0)


def gather_embeddings_on_rank0(
    shard_embedding,
    shard_sizes,
    save_dir,
    rank,
    world_size,
    dim,
):
    # Each rank saves its shard to a temp file
    shard_path = os.path.join(save_dir, f"shard_rank{rank}.memmap")
    shard_memmap = np.memmap(
        shard_path,
        dtype=np.float32,
        mode="w+",
        shape=shard_embedding.shape,
    )
    shard_memmap[:] = shard_embedding
    del shard_memmap

    dist.barrier()

    if rank == 0:
        total_size = sum(shard_sizes)
        final_path = os.path.join(save_dir, "embeddings.memmap")
        final_memmap = np.memmap(
            final_path,
            dtype=np.float32,
            mode="w+",
            shape=(total_size, dim),
        )
        offset = 0
        for r in range(world_size):
            r_shard_path = os.path.join(save_dir, f"shard_rank{r}.memmap")
            r_size = shard_sizes[r]
            r_memmap = np.memmap(
                r_shard_path,
                dtype=np.float32,
                mode="r",
                shape=(r_size, dim),
            )
            final_memmap[offset : offset + r_size] = r_memmap[:]
            offset += r_size
            del r_memmap
            os.remove(r_shard_path)

        del final_memmap
        print(f"Saved embeddings: shape=({total_size}, {dim}), path={final_path}")

    dist.barrier()


def build_faiss_index(embeddings_path, save_dir, total_size, dim):
    embeddings = np.memmap(
        embeddings_path,
        dtype=np.float32,
        mode="r",
        shape=(total_size, dim),
    )

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = os.path.join(save_dir, "index.faiss")
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index: {total_size} vectors, path={index_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build embedding index for Wikipedia paragraphs using multi-GPU encoding"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/wikipedia_en_qwen_keys_30w_filtered_v2.jsonl",
        help="Input JSONL file with 'paragraph' field",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data/embedding_index",
        help="Directory to save embeddings.memmap, index.faiss, metadata.json",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="Embedding model name or path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size per GPU for encoding",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max token length for tokenizer truncation",
    )
    args = parser.parse_args()

    # 1. Setup distributed
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
    dist.barrier()

    # 2. Load corpus
    if rank == 0:
        print(f"Loading dataset from {args.input_file}...")
    ds = load_dataset("json", data_files=args.input_file, split="train")
    all_paragraphs = ds["paragraph"]
    total_size = len(all_paragraphs)
    if rank == 0:
        print(f"Total paragraphs: {total_size}")

    # 3. Shard the data across ranks
    shard_start = (total_size * rank) // world_size
    shard_end = (total_size * (rank + 1)) // world_size
    shard_texts = all_paragraphs[shard_start:shard_end]
    shard_size = len(shard_texts)
    del ds, all_paragraphs

    # 4. Load model and tokenizer
    if rank == 0:
        print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModel.from_pretrained(args.model_name).half().to(device).eval()
    dim = model.config.hidden_size

    # 5. Encode shard
    if rank == 0:
        print(f"Encoding with {world_size} GPUs, batch_size={args.batch_size} per GPU...")
    shard_embeddings = encode_shard(
        model, tokenizer, shard_texts,
        args.batch_size, args.max_length, device,
    )
    assert shard_embeddings.shape == (shard_size, dim)

    # 6. Gather shard sizes
    shard_size_tensor = torch.tensor([shard_size], dtype=torch.long, device=device)
    shard_sizes_tensor = [
        torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)
    ]
    dist.all_gather(shard_sizes_tensor, shard_size_tensor)
    shard_sizes = [int(t.item()) for t in shard_sizes_tensor]

    # 7. Gather embeddings into single memmap
    gather_embeddings_on_rank0(
        shard_embeddings, shard_sizes, args.save_dir,
        rank, world_size, dim,
    )

    # 8. Rank 0 builds FAISS index and saves metadata
    if rank == 0:
        embeddings_path = os.path.join(args.save_dir, "embeddings.memmap")
        build_faiss_index(embeddings_path, args.save_dir, total_size, dim)

        metadata = {
            "total_size": total_size,
            "dim": dim,
            "model_name": args.model_name,
            "pooling_method": "cls",
            "normalization": "l2",
            "faiss_index_type": "IndexFlatIP",
            "input_file": args.input_file,
        }
        metadata_path = os.path.join(args.save_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
