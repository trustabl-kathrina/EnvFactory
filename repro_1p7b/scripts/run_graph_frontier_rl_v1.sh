#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-formal}
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
VERL=/home/u2024311031/verl-agent
PY=/home/u2024311031/.conda/envs/verl-agent/bin/python
MODEL=${ROOT}/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b
DATA=${ROOT}/repro_1p7b/data/graph_frontier_rl_v1
RUN_ROOT=${ROOT}/repro_1p7b/checkpoints/graph_frontier_rl_v1_1p7b
LOG_ROOT=${ROOT}/repro_1p7b/logs/graph_frontier_rl_v1

case "$MODE" in
  smoke_official)
    REWARD_MODE=official
    G=2
    STEPS=1
    MAX_STEPS=1
    SAVE_FREQ=1
    EXP=smoke_official
    OUT=${RUN_ROOT}/smoke_official
    ;;
  smoke_graph)
    REWARD_MODE=graph
    G=2
    STEPS=1
    MAX_STEPS=1
    SAVE_FREQ=1
    EXP=smoke_graph
    OUT=${RUN_ROOT}/smoke_graph
    ;;
  formal)
    REWARD_MODE=graph
    G=4
    STEPS=${TOTAL_STEPS:-60}
    MAX_STEPS=8
    SAVE_FREQ=10
    EXP=formal
    OUT=${RUN_ROOT}/verl_state
    ;;
  *) echo "usage: $0 smoke_official|smoke_graph|formal" >&2; exit 2 ;;
esac

mkdir -p "$OUT" "$LOG_ROOT/$EXP/rollouts"
test -s "$MODEL/model.safetensors"
test -s "$DATA/train.parquet"
test -s "$DATA/validation.parquet"

export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=false
export ENVFACTORY_ROOT="$ROOT"
export ENVFACTORY_REWARD_MODE="$REWARD_MODE"
export ENVFACTORY_RL_ROLLOUT_DIR="$LOG_ROOT/$EXP/rollouts"
export PYTHONPATH="$ROOT:$VERL${PYTHONPATH:+:$PYTHONPATH}"

cd "$VERL"
exec "$PY" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  data.train_files="$DATA/train.parquet" \
  data.val_files="$DATA/validation.parquet" \
  data.train_batch_size=2 \
  data.val_batch_size=1 \
  data.max_prompt_length=4096 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=true \
  data.truncation=error \
  data.return_raw_chat=true \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.model.lora_rank=64 \
  actor_rollout_ref.model.lora_alpha=64 \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$((2 * G)) \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.actor.use_invalid_action_penalty=true \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.enable_chunked_prefill=false \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.free_cache_engine=false \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=true \
  actor_rollout_ref.rollout.max_num_seqs=8 \
  env.env_name=EnvFactory \
  env.seed=20260916 \
  env.max_steps="$MAX_STEPS" \
  +env.max_tool_calls="$MAX_STEPS" \
  env.rollout.n="$G" \
  env.resources_per_worker.num_cpus=0.1 \
  trainer.critic_warmup=0 \
  trainer.logger=['console'] \
  trainer.project_name=envfactory_graph_frontier \
  trainer.experiment_name="graph_frontier_rl_v1_${EXP}" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq=-1 \
  trainer.total_epochs=100 \
  trainer.total_training_steps="$STEPS" \
  trainer.val_before_train=false \
  trainer.val_only=false \
  trainer.default_local_dir="$OUT" \
  ray_init.num_cpus=16
