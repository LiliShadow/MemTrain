#!/bin/bash
set -x

# at least 1 nodes, 4nodes=3~4days to converge
NNODES=${NNODES:-1}
NGPUS_PER_NODE=8
PROJ_ROOT=$(pwd)/results/
DATASET_ROOT=$(pwd)/taskutils/memory_data
WORKING_DIR=${PWD}
RUNTIME_ENV=${RUNTIME_ENV:-"${PWD}/verl/trainer/runtime_env.yaml"}

MODEL_PATH=models/Qwen/Qwen2.5-7B-Instruct
VAL_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_dev.parquet"
TRAIN_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_train_32k.parquet"
# TRAIN_PATH="${DATASET_ROOT}/mask_pretrain_150docs.parquet"
EXP=memagent/${MODEL_PATH##*/}
CKPT_DIR=checkpoints/$EXP

# Please note that recurrent framewrok will use max_length defined in task config.
# These two values are just for vLLM to decide max_model_length.
MAXLEN=8192 
MAX_NEW_TOKEN=1024

TENSORBOARD_DIR="${WORKING_DIR}/tensorboard_log/${EXP}"

# Inject TENSORBOARD_DIR into runtime env (Ray doesn't allow --runtime-env + --runtime-env-json together)
RUNTIME_ENV_JSON=$(python3 -c "
import yaml, json, sys
with open('${RUNTIME_ENV}') as f:
    env = yaml.safe_load(f)
env.setdefault('env_vars', {})['TENSORBOARD_DIR'] = '${TENSORBOARD_DIR}'
print(json.dumps(env))
")

# python3 -m verl.trainer.main_ppo \
ray job submit --runtime-env-json="${RUNTIME_ENV_JSON}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m verl.trainer.main_ppo \
    recurrent.enable=memory \
    recurrent.memory.config.max_chunks=8 \
    recurrent.memory.config.chunk_size=5000 \
    algorithm.adv_estimator=grpo \
    algorithm.grpo_use_adv=False \
    trainer.save_freq=50 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    trainer.logger=['console','tensorboard'] \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.clip_ratio_high=0.20 \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    data.train_files=$TRAIN_PATH \
    data.val_files=$VAL_PATH \
    data.shuffle=False \
    data.filter_overlong_prompts=True \
    data.train_batch_size=32 \
    data.truncation='center' \
    +data.context_key='context' \
    data.max_prompt_length=$MAXLEN \
    data.max_response_length=$MAX_NEW_TOKEN \
    reward_model.reward_manager='thread' \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384\
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.project_name='verl' \
    trainer.experiment_name=${EXP} \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.test_freq=5 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.total_training_steps=1000