GPU=2
CPU=$((GPU * 8))
MEMORY=$((GPU * 80000))
rjob delete codec-causal-patch
rjob submit \
  --name=codec-causal-patch \
  --gpu=$GPU \
  --cpu=$CPU \
  --memory=$MEMORY \
  --charged-group="brainllm_gpu" \
  --private-machine=group \
  --namespace="ailab-brainllm" \
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
torchrun --nproc_per_node=2 /mnt/shared-storage-user/wangjunyi/time/tools/codec_prediction_patch_causal_train.py \
--config /mnt/shared-storage-user/wangjunyi/time/configs/codec-prediction-patch-causal-fsq-data-v1.yaml
'