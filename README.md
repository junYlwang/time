# time

/root/miniconda3/envs/time/bin/python /mnt/shared-storage-user/wangjunyi/time/tools/profile_time300b_segment_stats.py \
  --manifest /mnt/shared-storage-user/wangjunyi/time/configs/data_v1.json \
  --split test \
  --segment_length 4096 \
  --samples_per_path 100000 \
  --batch_size 512 \
  --device auto \
  --output_csv /mnt/shared-storage-user/wangjunyi/time/data/data_v1_test_segment_stats.csv
