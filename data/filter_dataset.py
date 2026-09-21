#!/usr/bin/env python3
"""
过滤数据集，移除不含任何实体的文章。

使用 spaCy NER 进行实体识别，利用 datasets 库的并行处理加速。

使用方法:
    python filter_dataset.py --input input.jsonl --output filtered.jsonl
    python filter_dataset.py --input input.parquet --output filtered.jsonl
    python filter_dataset.py --input input.jsonl --output filtered.jsonl --num_proc 16
    python filter_dataset.py --input input.jsonl --output filtered.jsonl --verbose
    python filter_dataset.py  # 使用默认输入 ../data/wiki.parquet 和输出 filtered.jsonl

输入文件格式: JSON Lines (.jsonl) 或 Parquet (.parquet)
    - JSON Lines: 每行包含一个 JSON 对象，必须有 "paragraph" 字段（或指定 --text_field）
    - Parquet: 必须包含 "paragraph" 列（或指定 --text_field）
输出文件格式: JSON Lines，只保留包含实体且筛选后有符合条件的实体的文章
    - 始终包含 "paragraph" 字段，与 preprocess_mask_dataset_semantic.py 和 build_embedding_index.py 兼容
"""

import argparse
import json
import sys
from pathlib import Path

import datasets
from datasets import load_dataset, Dataset


def load_and_normalize_data(input_path: str, text_field: str = None, verbose: bool = False):
    """
    加载数据集并标准化为包含 'paragraph' 字段的格式。

    支持 .jsonl 和 .parquet 格式，自动检测文本字段并重命名为 'paragraph'。

    Args:
        input_path: 输入文件路径
        text_field: 文本字段名，None 则自动检测常见字段名
        verbose: 是否打印详细信息

    Returns:
        Dataset: 标准化后的数据集，包含 'paragraph' 字段
    """
    input_file = Path(input_path)
    suffix = input_file.suffix.lower()

    if verbose:
        print(f"加载数据集: {input_path}")

    # 根据文件扩展名加载数据
    if suffix == '.parquet':
        dataset = load_dataset("parquet", data_files=input_path, split="train")
    elif suffix in ['.jsonl', '.json', '']:
        dataset = load_dataset("json", data_files=input_path, split="train")
    else:
        # 尝试自动检测格式
        try:
            dataset = load_dataset("parquet", data_files=input_path, split="train")
        except:
            dataset = load_dataset("json", data_files=input_path, split="train")

    if verbose:
        print(f"可用字段: {dataset.column_names}")

    # 自动检测文本字段
    if text_field is None:
        # 优先检查 'paragraph' 字段（与下游兼容）
        if 'paragraph' in dataset.column_names:
            text_field = 'paragraph'
        else:
            # 检查其他常见字段名
            common_fields = ['text', 'content', 'body', 'article', 'passage']
            for field in common_fields:
                if field in dataset.column_names:
                    text_field = field
                    break

    if text_field is None:
        raise ValueError(
            f"无法自动检测文本字段。可用字段: {dataset.column_names}。"
            f"请使用 --text_field 指定，例如: --text_field text"
        )

    if verbose:
        print(f"使用文本字段: '{text_field}'")

    # 如果字段名不是 'paragraph'，需要重命名以兼容下游
    if text_field != 'paragraph':
        if verbose:
            print(f"将 '{text_field}' 重命名为 'paragraph' 以兼容下游脚本")
        dataset = dataset.rename_column(text_field, 'paragraph')

    # 确保 'paragraph' 字段存在
    if 'paragraph' not in dataset.column_names:
        raise ValueError("数据集中没有 'paragraph' 字段")

    return dataset


def filter_dataset(
    input_path: str,
    output_path: str,
    max_occurrence: int = 5,
    max_words: int = 8,
    num_proc: int = None,
    verbose: bool = False,
    text_field: str = None
) -> tuple[int, int, int]:
    """
    过滤数据集，只保留包含符合条件实体的文章。

    Args:
        input_path: 输入文件路径 (.jsonl 或 .parquet)
        output_path: 输出 JSONL 文件路径
        max_occurrence: 实体最大出现次数
        max_words: 实体最大单词数
        num_proc: 并行进程数，None 表示自动检测
        verbose: 是否打印详细信息
        text_field: 文本字段名，None 则自动检测

    Returns:
        (原始数量, 保留数量, 移除数量)
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 加载数据集（自动处理 .parquet 和 .jsonl，标准化为 'paragraph' 字段）
    dataset = load_and_normalize_data(input_path, text_field, verbose)
    total = len(dataset)

    if verbose:
        print(f"原始数据集大小: {total}")
        print(f"加载 spaCy 模型...")

    # 预加载 spaCy 模型（在每个 worker 中）
    from utils import get_spacy_nlp
    _ = get_spacy_nlp()

    if verbose:
        print(f"开始处理数据，并行进程数: {num_proc or '自动'}\n")

    # 定义过滤函数
    def has_valid_entities(example):
        """检查样本是否有符合条件的实体"""
        passage = example.get("paragraph", "")
        if not passage:
            return False

        from utils import extract_entities_spacy, filter_entities

        # 提取实体
        entities = extract_entities_spacy(passage)
        if not entities:
            return False

        # 筛选实体
        filtered = filter_entities(entities, passage, max_occurrence, max_words)
        return len(filtered) > 0

    # 使用 datasets 的 filter 方法进行并行过滤
    filtered_dataset = dataset.filter(
        has_valid_entities,
        num_proc=num_proc,
        desc="过滤数据集"
    )

    kept = len(filtered_dataset)
    removed = total - kept

    # 保存结果
    if verbose:
        print(f"\n保存过滤后的数据集...")

    filtered_dataset.to_json(output_path, lines=True, num_proc=num_proc)

    return total, kept, removed


def main():
    parser = argparse.ArgumentParser(
        description="过滤数据集，移除不含符合条件实体的文章（并行处理）"
    )
    parser.add_argument(
        "--input", "-i",
        default="../data/wiki.parquet",
        help="输入文件路径 (.jsonl 或 .parquet)，默认: ../data/wiki.parquet"
    )
    parser.add_argument(
        "--output", "-o",
        default="filtered.jsonl",
        help="输出 JSONL 文件路径，默认: filtered.jsonl"
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default=None,
        help="文本字段名（用于非 'paragraph' 字段的情况），默认自动检测"
    )
    parser.add_argument(
        "--max_occurrence",
        type=int,
        default=5,
        help="实体在文章中最大出现次数 (默认: 5)"
    )
    parser.add_argument(
        "--max_words",
        type=int,
        default=8,
        help="实体最大单词数 (默认: 8)"
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="并行进程数 (默认: 自动检测)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="打印详细信息"
    )

    args = parser.parse_args()

    print(f"开始过滤...")
    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    print(f"筛选条件: max_occurrence={args.max_occurrence}, max_words={args.max_words}")
    print()

    try:
        total, kept, removed = filter_dataset(
            args.input,
            args.output,
            args.max_occurrence,
            args.max_words,
            args.num_proc,
            args.verbose,
            args.text_field
        )

        print()
        print("=" * 50)
        print(f"处理完成!")
        print(f"原始文章数: {total}")
        print(f"保留文章数: {kept}")
        print(f"移除文章数: {removed}")
        print(f"保留比例: {kept/total*100:.2f}%" if total > 0 else "N/A")
        print("=" * 50)

    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
