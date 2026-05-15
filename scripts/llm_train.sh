GPU=2
CPU=$((GPU * 8))
MEMORY=$((GPU * 80000))
rjob delete ts-llm
rjob submit \
  --name=ts-llm \
  --gpu=$GPU \
  --cpu=$CPU \
  --memory=$MEMORY \
  --charged-group="brainllm_gpu" \
  --private-machine=group \
  --mount=gpfs://gpfs1/wangjunyi:/mnt/shared-storage-user/wangjunyi \
  --mount=gpfs://gpfs1/brainllm-share:/mnt/shared-storage-user/brainllm-share \
  --mount=gpfs://gpfs2/speechllm-share:/mnt/shared-storage-gpfs2/speechllm-share \
  --mount=gpfs://gpfs2/brainllm2-share:/mnt/shared-storage-gpfs2/brainllm2-share \
  --image=registry.h.pjlab.org.cn/ailab-brainllm-brainllm_gpu/junyi-workspace:wangjunyi-20260507140319 \
  --host-network=false \
  -- bash -exc '
set -ex
. /root/miniconda3/etc/profile.d/conda.sh
cd /mnt/shared-storage-user/wangjunyi
conda activate codec
torchrun --nproc_per_node=2 /mnt/shared-storage-user/wangjunyi/time/tools/train_ett_llm_lora.py \
--config /mnt/shared-storage-user/wangjunyi/time/configs/ett-llm-lora-qwen3-4b.yaml \
'