#!/usr/bin/env bash
set -e

SCRIPT_DIR=/mnt/shared-storage-user/wangjunyi/time/scripts/gift-eval
export GPU=1
export NUM_SHARDS=8
export RUN_DIR=/mnt/shared-storage-user/wangjunyi/time/runs/llm-ntp-no-pad/gift-eval-plot
export LLM_CONFIG=/mnt/shared-storage-user/wangjunyi/time/runs/llm-ntp-no-pad/2026-06-02_00-52-28_527981/config.yaml
export GIFT_EVAL_ROOT=/mnt/shared-storage-user/wangjunyi/gift-eval
export RJOB_NAME_PREFIX=gift-eval-shard
export SAVE_PLOT_SERIES=3
export SAVE_PLOT_SAMPLES=3

for SHARD_INDEX in $(seq 0 $((NUM_SHARDS - 1))); do
  export SHARD_INDEX
  bash "${SCRIPT_DIR}/gift_eval_shard${SHARD_INDEX}.sh"
done
