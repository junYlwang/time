# time

/root/miniconda3/envs/time/bin/python /mnt/shared-storage-user/wangjunyi/time/tools/check_codec_train_data_nan.py \
  --config /mnt/shared-storage-user/wangjunyi/time/configs/codec-prediction-rfsq2-data-v1.yaml \
  --max_steps 10000 \
  --output_dir /tmp/codec_nan_debug \
  --abs_threshold 0
  --continue_on_bad --max_bad_batches 10