# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import os
import re
from typing import List, Literal, Optional, Union

from datasets import disable_caching, enable_caching, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError

from .configs import ConvertToHFChatTemplateConfig, DataArguments, RoleBasedConverterConfig, RoleBasedConverterDPOConfig


DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

def transfer_to_role_based(example, system_keys=[], user_keys=["instruction", "input"], assistant_keys=["output"], seperator="\n"):
    """
    Transfers a dataset to a role-based.
    """
    system = {}
    for key in system_keys:
        if "content" in system:
            if example[key] and example[key] != "":
                system["content"] += seperator + example[key]
        else:
            if example[key] and example[key] != "":
                system["role"] = "system"
                system["content"] = example[key]
    
    user = {}
    for key in user_keys:
        if "content" in user:
            if example[key] and example[key] != "":
                user["content"] += seperator + example[key]
        else:
            if example[key] and example[key] != "":
                user["role"] = "user"
                user["content"] = example[key]
    
    assistant = {}
    for key in assistant_keys:
        if "content" in assistant:
            if example[key] and example[key] != "":
                assistant["content"] += seperator + example[key]
        else:
            if example[key] and example[key] != "":
                assistant["role"] = "assistant"
                assistant["content"] = example[key]
    
    example["messages"] = [system, user, assistant]
    return example

def transfer_to_role_based_dpo(example, system_keys=["system"], user_keys=["question"], chosen_keys=["chosen"], rejected_keys=["rejected"], seperator="\n"):
    system = {}
    for key in system_keys:
        if "content" in system:
            if example[key] and example[key] != "":
                system["content"] += seperator + example[key]
        else:
            if example[key] and example[key] != "":
                system["role"] = "system"
                system["content"] = example[key]
    
    user = {}
    for key in user_keys:
        if "content" in user:
            if example[key] and example[key] != "":
                user["content"] += seperator + example[key]
        else:
            if example[key] and example[key] != "":
                user["role"] = "user"
                user["content"] = example[key]
    
    chosen = {}
    for key in chosen_keys:
        if "content" in chosen:
            if example[key] and example[key] != "":
                chosen["content"] += seperator + example[key]
        else:
            if example[key] and example[key] != "":
                chosen["role"] = "assistant"
                chosen["content"] = example[key]
    
    rejected = {}
    for key in rejected_keys:
        if "content" in rejected:
            if example[key] and example[key] != "":
                rejected["content"] += seperator + example[key]
        else:
            if example[key] and example[key] != "":
                rejected["role"] = "assistant"
                rejected["content"] = example[key]
    
    example["chosen"] = [system, user, chosen]
    example["rejected"] = [system, user, rejected]
    return example

def convert_to_hf_chat_template(example, key="conversations", role_key="from", content_key="value", system_value="system", user_value="human", assistant_value="gpt"):
    messages = []
    for entry in example[key]:
        if entry[role_key] == system_value:
            messages.append({"role": "system", "content": entry[content_key]})
        elif entry[role_key] == user_value:
            messages.append({"role": "user", "content": entry[content_key]})
        elif entry[role_key] == assistant_value:
            messages.append({"role": "assistant", "content": entry[content_key]})
    example["messages"] = messages
    return example

def apply_chat_template(
    example, tokenizer, task: Literal["sft", "generation", "rm", "dpo"] = "sft", assistant_prefix="<|assistant|>\n"
):
    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)

    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True if task == "generation" else False
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if chosen_messages[0]["role"] != "system":
                chosen_messages.insert(0, {"role": "system", "content": ""})
            if rejected_messages[0]["role"] != "system":
                rejected_messages.insert(0, {"role": "system", "content": ""})
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "dpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            # Compared to reward modeling, we filter out the prompt, so the text is everything after the last assistant token
            prompt_messages = [[msg for msg in example["chosen"] if msg["role"] == "user"][0]]
            # Insert system message
            if example["chosen"][0]["role"] != "system":
                prompt_messages.insert(0, {"role": "system", "content": ""})
            else:
                prompt_messages.insert(0, example["chosen"][0])
            # TODO: handle case where chosen/rejected also have system messages
            chosen_messages = example["chosen"][1:]
            rejected_messages = example["rejected"][1:]
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            example["text_chosen"] = _strip_prefix(example["text_chosen"], assistant_prefix)
            example["text_rejected"] = _strip_prefix(example["text_rejected"], assistant_prefix)
    
        else:
            raise ValueError(
                f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation', 'rm', 'dpo']}"
        )
    return example


def get_datasets(
    data_config: Union[DataArguments, dict],
    splits: List[str] = ["train", "test"],
    shuffle: bool = True,
) -> DatasetDict:
    """
    Loads one or more datasets with varying training set proportions.

    Args:
        data_config (`DataArguments` or `dict`):
            Dataset configuration and split proportions.
        splits (`List[str]`, *optional*, defaults to `['train', 'test']`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.

    Returns
        [`DatasetDict`]: The dataset dictionary containing the loaded datasets.
    """

    if type(data_config) is DataArguments:
        # Structure of the config to read the datasets and their mix
        # datasets_mixer:
        #     - 'dataset1': 0.5
        #     - 'dataset2': 0.3
        #     - 'dataset3': 0.2
        dataset_mixer = data_config.dataset_mixer
    elif type(data_config) is dict:
        # Structure of the input is:
        #     dataset_mixer = {
        #             "dataset1": 0.5,
        #             "dataset1": 0.3,
        #             "dataset1": 0.2,
        #         }
        dataset_mixer = data_config
    else:
        raise ValueError(f"Data config {data_config} not recognized.")

    raw_datasets = mix_datasets(dataset_mixer, splits=splits, shuffle=shuffle)
    return raw_datasets


def mix_datasets(dataset_mixer: dict, splits: Optional[List[str]] = None, shuffle=True) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
    """
    raw_datasets = DatasetDict()
    raw_val_datasets = []
    fracs = []

    train_subsets = []

    seed = 42

    disable_caching()

    for dataset_args in dataset_mixer:
        seed = dataset_args.random_seed
        fracs.append(dataset_args.proportion)
        if dataset_args.proportion < 0:
            raise ValueError("Dataset fractions cannot be negative.")
        
        for split in splits:
            try:
                # Try first if dataset on a Hub repo
                data_files = dataset_args.data_files if dataset_args.data_files else None
                dataset = load_dataset(dataset_args.dataset, split=split, data_files=data_files)
            except DatasetGenerationError:
                # If not, check local dataset
                dataset = load_from_disk(os.path.join(dataset_args.dataset, split))
            except ValueError as e:
                print(e)
                dataset = None

            ############################
            # random dataset
            ############################
            if dataset:
                dataset = dataset.shuffle(seed=seed)
                dataset = dataset.select(range(int(dataset_args.proportion * len(dataset))))
                ############################
                # Filter out blacklisted sources
                ############################
                if dataset_args.blacklist_sources and dataset_args.blacklist_sources_key:
                    print("*** Filter out blacklisted sources ***")
                    dataset = dataset.filter(
                        lambda example: example[dataset_args.blacklist_sources_key] not in dataset_args.blacklist_sources
                    )
                
                #####################
                # Transfer to role based layout
                #####################
                if dataset_args.converter and type(dataset_args.converter) == RoleBasedConverterConfig:
                    print("*** Transfer to role based layout ***")
                    dataset = dataset.map(transfer_to_role_based, fn_kwargs={"system_keys": dataset_args.converter.system_keys, 
                                                                                "user_keys": dataset_args.converter.user_keys,
                                                                                "assistant_keys": dataset_args.converter.assistant_keys,
                                                                                "seperator": dataset_args.converter.seperator})
                
                if dataset_args.converter and type(dataset_args.converter) == ConvertToHFChatTemplateConfig:
                    print("*** Transfer to role based layout ***")
                    dataset = dataset.map(convert_to_hf_chat_template, fn_kwargs={"key": dataset_args.converter.key,
                                                                                    "role_key": dataset_args.converter.role_key,
                                                                                    "content_key": dataset_args.converter.content_key,
                                                                                    "system_value": dataset_args.converter.system_value,
                                                                                    "user_value": dataset_args.converter.user_value,
                                                                                    "assistant_value": dataset_args.converter.assistant_value})

                
                if dataset_args.converter and type(dataset_args.converter) == RoleBasedConverterDPOConfig:
                    print("*** Transfer to role based layout ***")
                    dataset = dataset.map(transfer_to_role_based_dpo, fn_kwargs={"system_keys": dataset_args.converter.system_keys, 
                                                                                "user_keys": dataset_args.converter.user_keys,
                                                                                "chosen_keys": dataset_args.converter.chosen_keys,
                                                                                "rejected_keys": dataset_args.converter.rejected_keys,
                                                                                "seperator": dataset_args.converter.seperator})

            if "train" in split:
                train_subsets.append(dataset)
            elif "test" in split:
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")

    for i, ds in enumerate(raw_val_datasets):
        if ds is None:
            tmp_split = train_subsets[i].train_test_split(test_size=0.02*fracs[i], shuffle=True, seed=seed)
            train_subsets[i] = tmp_split["train"]
            raw_val_datasets[i] = tmp_split["test"]

    if len(train_subsets) > 0:
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=seed)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(seed=seed)
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with split {split}. Check the dataset has been correctly formatted."
        )

    enable_caching()
    return raw_datasets
