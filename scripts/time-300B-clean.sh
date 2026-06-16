GPU=1
CPU=$((GPU * 8))
MEMORY=$((GPU * 80000))
rjob delete time-data-clean
rjob submit \
  --name=time-data-clean \
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

cd /mnt/shared-storage-user/wangjunyi/time
conda activate time
python tools/build_time300b_sequence_mae_manifest.py \
  --manifest configs/data_v3.json \
  --split train \
  --output-manifest configs/data_v4_sequence_mae.json \
  --output-sequence-manifest configs/data_v4_sequence_mae.train_sequences.json \
  --indices-dir configs/data_v4_sequence_mae_indices \
  --embedding-cache-dir configs/data_v4_sequence_mae_cache \
  --report-json configs/data_v4_sequence_mae_report.json \
  --encoder-config configs/codec-patch-causal-mae-fsq-data-v1.yaml \
  --encoder-checkpoint /mnt/shared-storage-user/wangjunyi/time/runs/codec-patch-causal-mae/2026-06-07_02-55-36__seed1234__gpu2/checkpoints/codec_steps=00198000_score=-0.1568 \
  --device cuda \
  --embedding-batch-size 4096 \
  --target-token-ratio 0.50 \
  --domain-alpha 0.50 \
  --family-alpha 0.35 \
  --hash-bits 16 \
  --overwrite-cache
'