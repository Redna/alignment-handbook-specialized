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
import os
import random
import sys
import shutil
import math
from alignment.configs import ChainOfLoraConfig

import datasets
from datasets import enable_caching, disable_caching
import torch
import transformers
from transformers import set_seed, AutoConfig
from peft import AutoPeftModelForCausalLM

import gc
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



logger = logging.getLogger(__name__)


def _rmtree_with_exclude(main_directory, excluded_folder=None):
    for root, dirs, files in os.walk(main_directory):
        if excluded_folder and root == os.path.join(main_directory, excluded_folder):
            continue
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            if not excluded_folder or dir != excluded_folder:
                shutil.rmtree(os.path.join(root, dir))

def _plot_word_count_distribution(name, dataset, bin_size=200):
    # Fügen Sie eine neue Spalte "word_count" hinzu, die die Anzahl der Wörter in jedem Text enthält
    dataset = dataset.map(lambda example: {"tokens": len(example["input_ids"])})

    # Zählen Sie die Häufigkeit jeder Wortanzahl
    tokens = list(dataset["tokens"])

    # Erstellen Sie die Bereiche
    max_tokens = min(max(tokens), 4000)
    bins = range(min(tokens), max_tokens + bin_size, bin_size)

     # Teilen Sie die Daten in Bereiche
    histogram, bins = np.histogram(tokens, bins=bins)

    # Erstellen Sie den Plot
    plt.bar(bins[:-1], histogram, width=bin_size)
    plt.xlabel('# of tokens')
    plt.ylabel('Count')
    plt.title('Token count distribution')

    # Speichern Sie den Plot als Bild
    plt.savefig(f'{name}.png')

def filter_long_examples(dataset, max_seq_length, model_args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        use_fast=True
    )

    dataset = dataset.map(lambda examples: tokenizer(examples['text']), batched=True)
    return dataset.filter(lambda example: len(example["input_ids"]) <= max_seq_length)


def build_model_kwargs(model_args, training_args, logger):
    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    optional_kwargs = {}

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        **optional_kwargs)
    logger.info("*** Model loaded! ***")
    
    return model_kwargs

def run_training(model_args, model_kwargs, training_args, train_dataset, eval_dataset, tokenizer, data_args, logger):
    ########################
    # Initialize the Trainer
    ########################
    packing = True

    disable_caching()
    train_dataset = train_dataset.shuffle(seed=training_args.seed)
    eval_dataset = eval_dataset.shuffle(seed=training_args.seed)
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=training_args.max_seq_length,
        tokenizer=tokenizer,
        packing=packing,
        peft_config=get_peft_config(model_args),
    )

    ###############
    # Training loop
    ###############
    if training_args.resume_from_checkpoint:
        logger.info("setting overwrite_output_dir=False to resume training")
        training_args.overwrite_output_dir = False

    logger.info("*** Train ***")

    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint) if training_args.resume_from_checkpoint else trainer.train()

    enable_caching()
    metrics = train_result.metrics
    max_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    return trainer

def merge_and_update_lora_model(model, tokenizer, training_args, round=0):
    model = None
    torch.cuda.empty_cache()

    model = AutoPeftModelForCausalLM.from_pretrained(training_args.output_dir, device_map="auto")
    model = model.merge_and_unload()

    # remove all under training_args.output_dir
    merged_dir = f"{training_args.output_dir}_merged_{round}"
    try: 
        shutil.rmtree(merged_dir)
    except:
        pass

    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    
    return model, merged_dir
    

def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig, ChainOfLoraConfig))
    model_args, data_args, training_args, chain_of_lora = parser.parse()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    accelerator = Accelerator()

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits)
    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, data_args)

    #####################
    # Apply chat template
    #####################

    raw_datasets = raw_datasets.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer, "task": "sft"})
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]


    ############################
    # Filter out too much tokens examples
    ############################
    print(f"Filtering out examples with more than {training_args.max_seq_length} tokens...")
    train_dataset = filter_long_examples(train_dataset, training_args.max_seq_length, model_args)
    eval_dataset = filter_long_examples(eval_dataset, training_args.max_seq_length, model_args)

    print("train_dataset", train_dataset)
    print("eval_dataset", eval_dataset)

    with training_args.main_process_first(desc="Log a few random samples from the processed training set"):
        for index in random.sample(range(len(raw_datasets["train"])), 5):
            logger.info(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")


    # Visualize the distribution as a graph
    _plot_word_count_distribution("train", train_dataset)
    _plot_word_count_distribution("eval", eval_dataset)

    chain_rounds = 1 # default
    if chain_of_lora.chain_of_lora_enabled:
        def calculate_chain_of_lora_parameters(epochs_per_round, num_train_epochs):
            chain_rounds = math.ceil(num_train_epochs / epochs_per_round)
            num_train_epochs_output = num_train_epochs / chain_rounds
            return num_train_epochs_output, chain_rounds
        
        training_args.num_train_epochs, chain_rounds = calculate_chain_of_lora_parameters(chain_of_lora.epochs_per_round, training_args.num_train_epochs)
        logger.info(f"Chain of LoRa enabled. Chain rounds: {chain_rounds}, epochs per round: {chain_of_lora.epochs_per_round}, total epochs: {training_args.num_train_epochs}")

    configured_model_name_or_path = model_args.model_name_or_path

    trainer = None

    output_dir = training_args.output_dir

    resume_from_round = 0

    if chain_of_lora.resume_from_round > 0:
        resume_from_round = chain_of_lora.resume_from_round
        logger.info(f"Resuming from round {resume_from_round}")
        model_args.model_name_or_path = f"{output_dir}_merged_{resume_from_round}"
        training_args.output_dir = f"{output_dir}_merged_res_{resume_from_round}"

    for chain_round in range(resume_from_round, chain_rounds):
        logger.info(f"Chain round {chain_round + 1} of {chain_rounds}")
        
        model_kwargs = build_model_kwargs(model_args, training_args, logger)

        if trainer: 
            del trainer.model
            del trainer
            torch.cuda.empty_cache()
            gc.collect()

        trainer = run_training(model_args, model_kwargs, training_args, train_dataset, eval_dataset, tokenizer, data_args, logger)
        trainer.save_model(training_args.output_dir, _internal_call=True)

        if get_quantization_config(model_args) or chain_of_lora.chain_of_lora_enabled:
            trainer.model, output_dir = merge_and_update_lora_model(trainer.model, tokenizer, training_args, round=chain_round)
            model_args.model_name_or_path = output_dir
            _rmtree_with_exclude(training_args.output_dir, excluded_folder="runs")
        
    model_args.model_name_or_path = configured_model_name_or_path
    trainer.model.name_or_path = configured_model_name_or_path
    trainer.model.config._name_or_path = configured_model_name_or_path
    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    if accelerator.is_main_process:
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "dataset": [dataset_mixer_entry.dataset for dataset_mixer_entry in data_args.dataset_mixer],
            "dataset_tags": [dataset_mixer_entry.dataset for dataset_mixer_entry in data_args.dataset_mixer],
            "tags": ["alignment-handbook-specialized"],
        }
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

        if training_args.push_to_hub is True:
            logger.info("Pushing to hub...")
            trainer.push_to_hub()


    
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
