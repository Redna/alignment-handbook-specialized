#!/bin/bash

# Your first command
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_dpo.py recipes/tinyllama-1_1b_reasoning_v2_sft/dpo/config_full.yaml > output.log 2>&1

# Your second command
runpodctl remove pod $RUNPOD_POD_ID


nohup sh -c 'ACCELERATE_LOG_LEVEL=info accelerate launch --config_file finetuning/alignment-handbook/recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 finetuning/alignment-handbook/scripts/run_sft.py finetuning/alignment-handbook/recipes/tukan-1_1b/sft/config_lora_round2.yaml' > output.log 2>&1 &