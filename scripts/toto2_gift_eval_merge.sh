#!/usr/bin/env bash
set -e

GIFT_EVAL_ROOT=${GIFT_EVAL_ROOT:-/mnt/shared-storage-user/wangjunyi/gift-eval}
RUN_DIR=${RUN_DIR:-/mnt/shared-storage-user/wangjunyi/time/runs/toto-2.0-2.5b-ft/gift-eval}
NUM_SHARDS=${NUM_SHARDS:-8}

SHARD_DIRS=()
for SHARD_INDEX in $(seq 0 $((NUM_SHARDS - 1))); do
  SHARD_DIRS+=("${RUN_DIR}/results_shard${SHARD_INDEX}")
done

python "${GIFT_EVAL_ROOT}/cli/merge_codec_llm_shards.py" \
  --shard-dirs "${SHARD_DIRS[@]}" \
  --output-dir "${RUN_DIR}/results_full_merged"
