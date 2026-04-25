python \
  /mnt/shared-storage-gpfs2/brainllm2-share/junyi/time/tools/infer_ucr.py \
  --config /mnt/shared-storage-gpfs2/brainllm2-share/junyi/time/configs/ucr-base.yaml \
  --ucr-root /mnt/shared-storage-gpfs2/brainllm2-share/junyi/time/data/UCR112 \
  --split TRAIN \
  --max-samples 0 \
  --plot-samples 3 \
  --output-dir /mnt/shared-storage-gpfs2/brainllm2-share/junyi/time/runs/ucr_infer
