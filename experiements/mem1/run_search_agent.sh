#!/bin/bash
# Example launch script for Search Agent training

# Set environment variables


# Data paths
WORKING_DIR=${PWD}
export DATA_DIR=${WORKING_DIR}/taskutils/memory_data/nq_hotpotqa_train_multi_2
export BASE_MODEL="models/Qwen/Qwen3-4B-Instruct-2507"
export PROJECT=MEM1
export EXPERIMENT_NAME=${BASE_MODEL##*/}
export CKPT_DIR=checkpoints/$PROJECT/$EXPERIMENT_NAME

# Search API endpoint
export SEARCH_URL=""

# Set error handling
set -ex
set -o pipefail

# Ray settings
export RAY_TMPDIR=/dev/shm/ray-$USER
mkdir -p $RAY_TMPDIR
export RAY_memory_usage_threshold=0.9
export TENSORBOARD_DIR="${WORKING_DIR}/tensorboard_log/${PROJECT}/${EXPERIMENT_NAME}"

# Create logs directory
mkdir -p logs

# Launch training
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test_2000.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=1000 \
    data.shuffle=True \
    recurrent.enable=search_agent \
    recurrent.search_agent.config.max_turns=6 \
    recurrent.search_agent.config.max_start_length=2048 \
    recurrent.search_agent.config.max_prompt_length=4096 \
    recurrent.search_agent.config.max_response_length=1000 \
    recurrent.search_agent.config.max_obs_length=1000 \
    recurrent.search_agent.config.search_url=$SEARCH_URL \
    recurrent.search_agent.config.topk=3 \
    algorithm.adv_estimator=grpo \
    algorithm.grpo_use_adv=true \
    algorithm.enable_memory_recall=false \
    algorithm.memory_recall_coeff=0.5 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_thinking=true \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384\
    actor_rollout_ref.rollout.n=8 \
    +actor_rollout_ref.rollout.detokenize=true \
    +actor_rollout_ref.rollout.stop="['</search>', '</answer>']" \
    +actor_rollout_ref.rollout.include_stop_str_in_output=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.project_name=$PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=3 \
    trainer.total_training_steps=1000 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1
