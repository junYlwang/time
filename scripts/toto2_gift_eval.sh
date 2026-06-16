#!/usr/bin/env bash
set -e

SCRIPT_DIR=/mnt/shared-storage-user/wangjunyi/time/scripts/gift-eval-toto2
export GPU=${GPU:-1}
export NUM_SHARDS=${NUM_SHARDS:-8}
export RUN_DIR=${RUN_DIR:-/mnt/shared-storage-user/wangjunyi/time/runs/toto-2.0-2.5b-ft/gift-eval}
export MODEL_PATH=${MODEL_PATH:-/mnt/shared-storage-gpfs2/speechllm-share/junyi/models/toto/Toto-2.0-2.5B-FT}
export DATA_ROOT=${DATA_ROOT:-/mnt/shared-storage-gpfs2/speechllm-share/lishenyi/datasets/gift-eval}
export GIFT_EVAL_ROOT=${GIFT_EVAL_ROOT:-/mnt/shared-storage-user/wangjunyi/gift-eval}
export TOTO_SRC=${TOTO_SRC:-/mnt/shared-storage-user/wangjunyi/toto/toto2}
export CONDA_ENV=${CONDA_ENV:-toto2}
export MODEL_NAME=${MODEL_NAME:-Toto-2.0-2.5B-FT}
export RJOB_NAME_PREFIX=${RJOB_NAME_PREFIX:-gift-eval-toto2-shard}
export CONTEXT_LENGTH=${CONTEXT_LENGTH:-4096}
export BATCH_SIZE=${BATCH_SIZE:-64}
export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-$BATCH_SIZE}
export DECODE_BLOCK_SIZE=${DECODE_BLOCK_SIZE:-0}
export SAVE_PLOT_SERIES=${SAVE_PLOT_SERIES:-3}
export LIMIT=${LIMIT:-0}

for SHARD_INDEX in $(seq 0 $((NUM_SHARDS - 1))); do
  export SHARD_INDEX
  bash "${SCRIPT_DIR}/toto2_gift_eval_shard${SHARD_INDEX}.sh"
done
