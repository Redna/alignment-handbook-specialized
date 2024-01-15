#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Supervised fine-tuning script for decoder language models.
"""

import logging
import random
import sys

import datasets
import torch
import transformers
from transformers import set_seed, AutoConfig, AutoTokenizer

from accelerate import Accelerator
from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    SFTConfig,
    apply_chat_template,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)
from trl import SFTTrainer
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

from peft import AutoPeftModelForCausalLM


logger = logging.getLogger(__name__)

def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse()

    model = AutoPeftModelForCausalLM.from_pretrained(training_args.output_dir, device_map="auto")
    model = model.merge_and_unload()
    model.save_pretrained(training_args.output_dir + "_merged")

    tokenizer = AutoTokenizer.from_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir + "_merged")
    
if __name__ == "__main__":
    main()