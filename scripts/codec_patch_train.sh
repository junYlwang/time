GPU=2
CPU=$((GPU * 8))
MEMORY=$((GPU * 80000))
rjob delete codec-patch
rjob submit \
  --name=codec-patch \
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
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501 \
  tools/codec_prediction_patch_causal_train.py \
  --config configs/codec-prediction-patch-causal-fsq-data-v1.yaml &
pid1=$!

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29502 \
  tools/codec_prediction_patch_local_train.py \
  --config configs/codec-prediction-patch-local-fsq-data-v1.yaml &
pid2=$!

set +e
wait $pid1
status1=$?
wait $pid2
status2=$?
set -e

if [ $status1 -ne 0 ] || [ $status2 -ne 0 ]; then
  exit 1
fi
'