#!/usr/bin/env bash
set -e

GPU=${GPU:-1}
CPU=${CPU:-$((GPU * 8))}
MEMORY=${MEMORY:-$((GPU * 80000))}
SHARD_INDEX=${SHARD_INDEX:-0}
NUM_SHARDS=${NUM_SHARDS:?NUM_SHARDS is required}
RUN_DIR=${RUN_DIR:?RUN_DIR is required}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
DATA_ROOT=${DATA_ROOT:?DATA_ROOT is required}
GIFT_EVAL_ROOT=${GIFT_EVAL_ROOT:-/mnt/shared-storage-user/wangjunyi/gift-eval}
TOTO_SRC=${TOTO_SRC:-/mnt/shared-storage-user/wangjunyi/toto/toto2}
CONDA_ENV=${CONDA_ENV:-toto2}
MODEL_NAME=${MODEL_NAME:-Toto-2.0-2.5B-FT}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-4096}
BATCH_SIZE=${BATCH_SIZE:-64}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-$BATCH_SIZE}
DECODE_BLOCK_SIZE=${DECODE_BLOCK_SIZE:-0}
SAVE_PLOT_SERIES=${SAVE_PLOT_SERIES:-3}
LIMIT=${LIMIT:-0}
RJOB_NAME=${RJOB_NAME_PREFIX:-gift-eval-toto2-shard}${SHARD_INDEX}

rjob delete "${RJOB_NAME}"
rjob submit \
  --name="${RJOB_NAME}" \
  --gpu=$GPU \
  --cpu=$CPU \
  --memory=$MEMORY \
  --charged-group="speechllm_gpu" \
  --private-machine=group \
  --namespace="ailab-speechllm" \
  --mount=gpfs://gpfs1/wangjunyi:/mnt/shared-storage-user/wangjunyi \
  --mount=gpfs://gpfs1/brainllm-share:/mnt/shared-storage-user/brainllm-share \
  --mount=gpfs://gpfs2/speechllm-share:/mnt/shared-storage-gpfs2/speechllm-share \
  --mount=gpfs://gpfs2/brainllm2-share:/mnt/shared-storage-gpfs2/brainllm2-share \
  --image=registry.h.pjlab.org.cn/ailab-brainllm-brainllm_gpu/junyi-workspace:wangjunyi-20260529195906 \
  -- bash -exc '
set -ex
. /root/miniconda3/etc/profile.d/conda.sh
cd "'"${GIFT_EVAL_ROOT}"'"
conda activate "'"${CONDA_ENV}"'"
export PYTHONPATH="'"${TOTO_SRC}"':'"${GIFT_EVAL_ROOT}"'/src:${PYTHONPATH}"

OUT_DIR="'"${RUN_DIR}"'/results_shard'"${SHARD_INDEX}"'"
mkdir -p "${OUT_DIR}"

CUDA_VISIBLE_DEVICES=0 python "'"${GIFT_EVAL_ROOT}"'/cli/toto2.py" \
  --model-path "'"${MODEL_PATH}"'" \
  --toto-src "'"${TOTO_SRC}"'" \
  --data-root "'"${DATA_ROOT}"'" \
  --output-dir "${OUT_DIR}" \
  --model-name "'"${MODEL_NAME}"'" \
  --context-length "'"${CONTEXT_LENGTH}"'" \
  --batch-size "'"${BATCH_SIZE}"'" \
  --eval-batch-size "'"${EVAL_BATCH_SIZE}"'" \
  --decode-block-size "'"${DECODE_BLOCK_SIZE}"'" \
  --num-shards "'"${NUM_SHARDS}"'" \
  --shard-index "'"${SHARD_INDEX}"'" \
  --save-plot-series "'"${SAVE_PLOT_SERIES}"'" \
  --limit "'"${LIMIT}"'"
'
