GPU=1
CPU=$((GPU * 8))
MEMORY=$((GPU * 160000))
rjob delete time-codec-1000
rjob submit \
  --name=time-codec-1000 \
  --gpu=$GPU \
  --cpu=$CPU \
  --memory=$MEMORY \
  --charged-group="brainllm_gpu" \
  --private-machine=group \
  --mount=gpfs://gpfs1/brainllm-share:/mnt/shared-storage-user/brainllm-share \
  --mount=gpfs://gpfs2/speechllm-share:/mnt/shared-storage-gpfs2/speechllm-share \
  --mount=gpfs://gpfs2/brainllm2-share:/mnt/shared-storage-gpfs2/brainllm2-share \
  --image=registry.h.pjlab.org.cn/ailab-brainllm-brainllm_gpu/junyi-workspace:wangjunyi-20260121133940 \
  --host-network=false \
  -- bash -exc '
set -ex
. /root/miniconda3/etc/profile.d/conda.sh
cd /mnt/shared-storage-gpfs2/brainllm2-share/junyi/llm_assess
conda activate codec
torchrun --nproc_per_node=4 train.py \
--config /mnt/shared-storage-gpfs2/brainllm2-share/junyi/llm_assess/config.yaml
'