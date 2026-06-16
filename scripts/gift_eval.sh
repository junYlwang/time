#!/usr/bin/env bash
set -e

SCRIPT_DIR=/mnt/shared-storage-user/wangjunyi/time/scripts/gift-eval
export GPU=1
export NUM_SHARDS=8
export RUN_DIR=/mnt/shared-storage-user/wangjunyi/time/runs/llm-ntp-patch-causal/gift-eval
export LLM_CONFIG=/mnt/shared-storage-user/wangjunyi/time/runs/llm-ntp-patch-causal/2026-06-07_01-07-33_758345/config.yaml
export GIFT_EVAL_ROOT=/mnt/shared-storage-user/wangjunyi/gift-eval
export RJOB_NAME_PREFIX=gift-eval-patch-causal-shard
export SAVE_PLOT_SERIES=3
export SAVE_PLOT_SAMPLES=3

for SHARD_INDEX in $(seq 0 $((NUM_SHARDS - 1))); do
  export SHARD_INDEX
  bash "${SCRIPT_DIR}/gift_eval_shard${SHARD_INDEX}.sh"
done
