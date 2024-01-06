#!/bin/bash

# Your first command
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_dpo.py recipes/tinyllama-1_1b_reasoning_v2_sft/dpo/config_full.yaml > output.log 2>&1

# Your second command
runpodctl remove pod $RUNPOD_POD_ID
