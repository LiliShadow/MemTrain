# MemTrain

采用两阶段训练流程：

1. **memtrain**：掩码实体预测预训练（`recurrent.enable=memory`，开启 memory recall 辅助损失），训练模型"记忆"长文档的能力；
2. **memagent / mem1**：以 memtrain 的 checkpoint 为初始模型，分别进行两条 RL（GRPO）训练：
   - **memagent**：多 chunk 读入 + 记忆整理的记忆型 agent（`recurrent.enable=memory`）；
   - **mem1**：多轮 Think-Search 的搜索型 agent（`recurrent.enable=search_agent`，改编自 MEM1）。

## 环境安装

```bash
pip install -r requirements.txt
```

## 数据处理

### 1. memtrain 掩码预训练数据（`mask_pretrain_150docs_top30.parquet`）

```bash
# 1) 过滤 Wikipedia 段落：仅保留含可掩码实体的文章
python data/filter_dataset.py --input wiki.parquet --output filtered.jsonl

# 2) 多卡构建 BGE 向量 + FAISS 索引
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    data/build_embedding_index.py \
    --input_file filtered.jsonl --save_dir embedding_index \
    --model_name BAAI/bge-base-en-v1.5

# 3) 生成掩码预训练数据：150 篇文档拼成长上下文，掩码锚文档中的实体，
#    并混入 top-30 语义相近文档
python data/preprocess_mask_dataset_semantic.py \
    --input_file filtered.jsonl --index_dir embedding_index \
    --output_file taskutils/memory_data/mask_pretrain_150docs_top30.parquet \
    --num_samples 30000 --num_docs 150
```

### 2. memagent QA 数据（`hotpotqa_train_32k.parquet` / `hotpotqa_dev.parquet`）

```bash
cd taskutils/memory_data
bash download_qa_dataset.sh   # 下载 SQuAD/HotpotQA
python processing.py          # 拼接多文档长上下文 + 问题，输出 parquet
```

### 3. mem1 搜索数据（`nq_hotpotqa_train_multi_2/`）

由 NQ + HotpotQA 多跳问题经 `convert_nq_to_hotpotqa.py` 转换为 `train.parquet` / `test_2000.parquet`（多问题混合的搜索任务格式）。

## 实验运行
### 第一步：memtrain（掩码预训练）

```bash
bash experiements/run_memtrain_qwen2.5_7b_dist.sh
```

- 初始模型 `Qwen2.5-7B-Instruct`，训练数据 `mask_pretrain_150docs_top30.parquet`；
- `trainer.save_freq=50`，训练结束后取目标 step 的 checkpoint（如 `global_step_300`）。

训练完成后将 FSDP 分片合并为 HF 格式（后续脚本直接引用该目录）：

```bash
CKPT=checkpoints/memtrain/Qwen2.5-7B-Instruct/global_step_300
BASE=models/Qwen/Qwen2.5-7B-Instruct
CKPT=$CKPT BASE=$BASE bash scripts/merger.sh   # 生成 $CKPT/huggingface
```

### 第二步 A：memagent（基于 memtrain checkpoint）

```bash
bash experiements/memtrain_then_mem/run_memtrain_then_memory_qwen2.5_7B_ins_dist.sh
```

- `MODEL_PATH` 指向 memtrain 的 `global_step_300/huggingface`；
- 训练数据 `hotpotqa_train_32k.parquet`，验证 `hotpotqa_dev.parquet`；
- 对照实验（不经过 memtrain、直接从 base 模型训练）：`experiements/memagent/run_memory_qwen2_5_7b_dist.sh`。

### 第二步 B：mem1（基于 memtrain checkpoint）

```bash
bash experiements/mem1/run_memtrain_then_mem1_qwen2.5-7B-ins_dist.sh
```

- `MODEL_PATH` 同样指向 memtrain 的 `global_step_300/huggingface`；
- 训练数据 `nq_hotpotqa_train_multi_2/train.parquet`，`recurrent.enable=search_agent`；
- 需在脚本中设置 `SEARCH_URL`（外部搜索 API，默认 topk=3、max_turns=6）；
- 对照实验：`experiements/mem1/run_search_agent.sh`（从 base 模型训练）。

## 评测

```bash
# HotpotQA 长文档评测
python taskutils/memory_eval/ruler_hqa.py
```
