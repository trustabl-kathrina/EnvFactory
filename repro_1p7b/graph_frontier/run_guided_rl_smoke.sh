#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?usage: run_guided_rl_smoke.sh official|graph_frontier}
shift
if [[ "${MODE}" != "official" && "${MODE}" != "graph_frontier" ]]; then
  echo "invalid reward mode: ${MODE}" >&2
  exit 2
fi

MAIN_ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
VERL_ROOT=/home/u2024311031/verl-agent
VERL_ENV=/home/u2024311031/.conda/envs/verl-agent
MODEL=${MAIN_ROOT}/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b
DATA=${GUIDED_RL_DATA:-${MAIN_ROOT}/repro_1p7b/data/graph_frontier_rl_v1_generated/smoke12_parquet}
LOG_ROOT=${GUIDED_RL_LOG_ROOT:-${MAIN_ROOT}/repro_1p7b/logs/graph_frontier_rl_v1/rl_smoke_${MODE}}
CHECKPOINT_ROOT=${GUIDED_RL_CHECKPOINT_ROOT:-${MAIN_ROOT}/repro_1p7b/checkpoints/graph_frontier_rl_v1_smoke_${MODE}}
ROLLOUT_N=${GUIDED_RL_ROLLOUT_N:-2}

mkdir -p "${LOG_ROOT}" "${CHECKPOINT_ROOT}"
export PATH=${VERL_ENV}/bin:${PATH}
export PYTHONPATH=${VERL_ROOT}:${MAIN_ROOT}
export ENVFACTORY_ROOT=${MAIN_ROOT}
export ENVFACTORY_REWARD_MODE=${MODE}
export ENVFACTORY_RL_ROLLOUT_DIR=${LOG_ROOT}/rollouts
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=false
export JAVA_HOME=${VERL_ENV}
export JVM_PATH=${VERL_ENV}/lib/jvm/lib/server/libjvm.so

(
  while true; do
    date --iso-8601=seconds
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
    sleep 2
  done
) > "${LOG_ROOT}/gpu.csv" 2>&1 &
MONITOR_PID=$!
cleanup() {
  kill "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

cd "${VERL_ROOT}"
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files=${DATA}/train.parquet \
  data.val_files=${DATA}/validation.parquet \
  data.train_batch_size=2 \
  data.val_batch_size=2 \
  data.max_prompt_length=5200 \
  data.max_response_length=768 \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  data.return_raw_chat=True \
  data.shuffle=False \
  actor_rollout_ref.model.path=${MODEL} \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.use_invalid_action_penalty=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
  actor_rollout_ref.rollout.max_model_len=6144 \
  actor_rollout_ref.rollout.max_num_batched_tokens=6144 \
  actor_rollout_ref.rollout.max_num_seqs=8 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.block_size=64 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  env.env_name=EnvFactory \
  env.seed=20260917 \
  env.max_steps=8 \
  env.rollout.n=${ROLLOUT_N} \
  env.resources_per_worker.num_cpus=0.1 \
  trainer.critic_warmup=0 \
  trainer.logger=['console'] \
  trainer.project_name=graph_frontier_guided_rl_v1 \
  trainer.experiment_name=smoke_${MODE} \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.default_local_dir=${CHECKPOINT_ROOT} \
  trainer.resume_mode=disable \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.val_before_train=False \
  trainer.val_only=False \
  ray_init.num_cpus=32 \
  "$@"
