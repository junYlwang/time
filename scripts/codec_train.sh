GPU=4
CPU=$((GPU * 8))
MEMORY=$((GPU * 80000))
rjob delete codec-base-rfsq2-data-v1
rjob submit \
  --name=codec-base-rfsq2-data-v1 \
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
  --image=registry.h.pjlab.org.cn/ailab-brainllm-brainllm_gpu/junyi-workspace:wangjunyi-20260507140319 \
  --host-network=false \
  -- bash -exc '
set -ex
. /root/miniconda3/etc/profile.d/conda.sh
cd /mnt/shared-storage-user/wangjunyi/time
conda activate time
torchrun --nproc_per_node=4 /mnt/shared-storage-user/wangjunyi/time/tools/codec_train.py \
--config /mnt/shared-storage-user/wangjunyi/time/configs/codec-base-rfsq2-data-v1.yaml
'