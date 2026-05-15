for cfg in /mnt/shared-storage-gpfs2/brainllm2-share/junyi/time/runs/ett-3/*-mlp/*/meta/config_resolved.yaml; do
  echo "Testing ${cfg}"
  python /mnt/shared-storage-gpfs2/brainllm2-share/junyi/time/tools/ett_test.py --config "${cfg}"
done