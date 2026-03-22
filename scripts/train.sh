#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-/mnt/shared-storage-gpfs2/brainllm2-share/junyi/time/configs/time-codec.yaml}

python /mnt/shared-storage-gpfs2/brainllm2-share/junyi/time/tools/train.py --config "$CONFIG_PATH"
