#!/usr/bin/env bash
set -e

GPU=${GPU:-1}
CPU=${CPU:-$((GPU * 8))}
MEMORY=${MEMORY:-$((GPU * 80000))}
SHARD_INDEX=${SHARD_INDEX:-6}
NUM_SHARDS=${NUM_SHARDS:?NUM_SHARDS is required}
RUN_DIR=${RUN_DIR:?RUN_DIR is required}
LLM_CONFIG=${LLM_CONFIG:?LLM_CONFIG is required}
GIFT_EVAL_ROOT=${GIFT_EVAL_ROOT:-/mnt/shared-storage-user/wangjunyi/gift-eval}
RJOB_NAME=${RJOB_NAME_PREFIX:-gift-eval-shard}${SHARD_INDEX}
SAVE_PLOT_SERIES=${SAVE_PLOT_SERIES:-3}
SAVE_PLOT_SAMPLES=${SAVE_PLOT_SAMPLES:-3}

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
cd /mnt/shared-storage-user/wangjunyi/gift-eval
conda activate time

OUT_DIR="'"${RUN_DIR}"'/results_shard'"${SHARD_INDEX}"'"
mkdir -p "${OUT_DIR}"

CUDA_VISIBLE_DEVICES=0 python "'"${GIFT_EVAL_ROOT}"'/cli/codec_llm.py" \
  --llm-config "'"${LLM_CONFIG}"'" \
  --num-shards "'"${NUM_SHARDS}"'" \
  --shard-index "'"${SHARD_INDEX}"'" \
  --output-dir "${OUT_DIR}" \
  --save-plot-series "'"${SAVE_PLOT_SERIES}"'" \
  --save-plot-samples "'"${SAVE_PLOT_SAMPLES}"'"
'
