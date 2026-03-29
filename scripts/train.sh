GPU=1
CPU=$((GPU * 8))
MEMORY=$((GPU * 160000))
rjob delete time-codec-rfsq-1000-zscore-stft
rjob submit \
  --name=time-codec-rfsq-1000-zscore-stft \
  --gpu=$GPU \
  --cpu=$CPU \
  --memory=$MEMORY \
  --charged-group="brainllm_gpu" \
  --private-machine=group \
  --mount=gpfs://gpfs1/brainllm-share:/mnt/shared-storage-user/brainllm-share \
  --mount=gpfs://gpfs2/speechllm-share:/mnt/shared-storage-gpfs2/speechllm-share \
  --mount=gpfs://gpfs2/brainllm2-share:/mnt/shared-storage-gpfs2/brainllm2-share \
  --image=registry.h.pjlab.org.cn/ailab-brainllm-brainllm_gpu/junyi-workspace:wangjunyi-new \
  --host-network=false \
  -- bash -exc '
set -ex
. /root/miniconda3/etc/profile.d/conda.sh
cd /mnt/shared-storage-gpfs2/brainllm2-share/junyi/time
conda activate time
python -m tools.train \
--config /mnt/shared-storage-gpfs2/brainllm2-share/junyi/time/configs/rfsq-1000-zscore-stft.yaml
'