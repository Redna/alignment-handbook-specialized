#!/bin/bash

# Redirect all output of this script to 'nohup.out'
exec > nohup.out 2>&1

echo "Round 1:" > status.log
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/tukan-1_1b/sft/config_lora.yaml > output.log 2>&1 &
pid=$!  # Capture the PID of the last background process
wait $pid  # Wait for the process to complete
python scripts/merge_lora.py recipes/tukan-1_1b/sft/config_lora.yaml

echo "Round 2:" >> status.log
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/tukan-1_1b/sft/config_lora_2.yaml > output_2.log 2>&1 &
pid=$!
wait $pid
python scripts/merge_lora.py recipes/tukan-1_1b/sft/config_lora_2.yaml

echo "Round 3:" >> status.log
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/tukan-1_1b/sft/config_lora_3.yaml > output_3.log 2>&1 &
pid=$!
wait $pid
python scripts/merge_lora.py recipes/tukan-1_1b/sft/config_lora_3.yaml

