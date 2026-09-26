#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?usage: run_guided_rl_v2_train.sh official|graph RUN_NAME}
RUN_NAME=${2:?usage: run_guided_rl_v2_train.sh official|graph RUN_NAME}
if [[ "${MODE}" != official && "${MODE}" != graph ]]; then
  echo "invalid v2 transport mode: ${MODE}" >&2
  exit 2
fi

MAIN_ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
VERL_ROOT=/home/u2024311031/verl-agent
VERL_ENV=/home/u2024311031/.conda/envs/verl-agent
MODEL=${MAIN_ROOT}/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b
DATA=${GUIDED_RL_DATA:?GUIDED_RL_DATA is required}
LOG_ROOT=${GUIDED_RL_LOG_ROOT:?GUIDED_RL_LOG_ROOT is required}
CHECKPOINT_ROOT=${GUIDED_RL_CHECKPOINT_ROOT:?GUIDED_RL_CHECKPOINT_ROOT is required}
TOTAL_STEPS=${GUIDED_RL_TOTAL_STEPS:?GUIDED_RL_TOTAL_STEPS is required}
ROLLOUT_N=${GUIDED_RL_ROLLOUT_N:-4}
SAVE_FREQ=${GUIDED_RL_SAVE_FREQ:-${TOTAL_STEPS}}
EXPECTED_HASH=5cef60450c57478df241da3438053fcc34ebcacfc2edc2c7ab761ccf4b27ee2f

for required in "${MODEL}/config.json" "${DATA}/train.parquet" "${DATA}/validation.parquet"; do
  [[ -e "${required}" ]] || { echo "missing training input: ${required}" >&2; exit 2; }
done
if [[ "${DATA}" == *parquet_pilot128* ]]; then
  actual_hash=$(sha256sum "${DATA}/train.parquet" | awk '{print $1}')
  [[ "${actual_hash}" == "${EXPECTED_HASH}" ]] || { echo PILOT_DATA_HASH_MISMATCH >&2; exit 3; }
fi
if [[ "${CHECKPOINT_ROOT}" == "${MODEL}" || -e "${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt" ]]; then
  echo "refusing to overwrite model or resume an existing run: ${CHECKPOINT_ROOT}" >&2
  exit 2
fi

mkdir -p "${LOG_ROOT}" "${CHECKPOINT_ROOT}"
printf '%s\n' STARTING > "${LOG_ROOT}/status"
export PATH=${VERL_ENV}/bin:${PATH}
export PYTHONPATH=${VERL_ROOT}:${MAIN_ROOT}
export ENVFACTORY_ROOT=${MAIN_ROOT}
export ENVFACTORY_REWARD_MODE=official
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
cleanup() { kill "${MONITOR_PID}" 2>/dev/null || true; }
trap cleanup EXIT

printf '%s\n' TRAINING > "${LOG_ROOT}/status"
cd "${VERL_ROOT}"
set +e
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  +algorithm.compute_mean_std_cross_steps_in_grpo=False \
  +algorithm.guided_rl_v2_transport=${MODE} \
  +algorithm.guided_rl_v2_tie_tolerance=1e-6 \
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
  trainer.project_name=graph_frontier_guided_rl_v2 \
  trainer.experiment_name=${RUN_NAME} \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.default_local_dir=${CHECKPOINT_ROOT} \
  trainer.resume_mode=disable \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.test_freq=-1 \
  trainer.total_epochs=1000 \
  trainer.total_training_steps=${TOTAL_STEPS} \
  trainer.val_before_train=False \
  trainer.val_only=False \
  ray_init.num_cpus=32 \
  2>&1 | tee "${LOG_ROOT}/train.log"
EXIT_CODE=${PIPESTATUS[0]}
set -e
if [[ ${EXIT_CODE} -eq 0 ]]; then
  printf '%s\n' DONE > "${LOG_ROOT}/status"
else
  printf 'FAILED:%s\n' "${EXIT_CODE}" > "${LOG_ROOT}/status"
fi
exit "${EXIT_CODE}"
