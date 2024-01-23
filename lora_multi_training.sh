#!/bin/bash

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/tukan-1_1b/sft/config_lora.yaml > output.log 2>&1 &