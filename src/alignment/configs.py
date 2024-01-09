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
import dataclasses
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, NewType, Optional, Tuple

import transformers
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, HfArgumentParser


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


DataClassType = NewType("DataClassType", Any)


class H4ArgumentParser(HfArgumentParser):
    def parse_yaml_and_args(self, yaml_arg: str, other_args: Optional[List[str]] = None) -> List[dataclass]:
        """
        Parse a YAML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (`str`):
                The path to the config file used
            other_args (`List[str]`, *optional`):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'].

        Returns:
            [`List[dataclass]`]: a list of dataclasses with the values from the YAML file and the command line
        """
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args = {arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args}
        used_args = {}

        # overwrite the default/loaded value with the value provided to the command line
        # adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                # add only if in keys
                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == List[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]

                    # bool of a non-empty string is True, so we manually check for bools
                    if base_type == bool:
                        if val in ["true", "True"]:
                            inputs[arg] = True
                        else:
                            inputs[arg] = False

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(f"Duplicate argument provided: {arg}, may cause unexpected behavior")

            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs

    def parse(self) -> DataClassType | Tuple[DataClassType]:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a YAML file,
            # let's parse it to get our arguments.
            output = self.parse_yaml_file(os.path.abspath(sys.argv[1]))
        # parse command line args and yaml file
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
            output = self.parse_yaml_and_args(os.path.abspath(sys.argv[1]), sys.argv[2:])
        # parse command line args only
        else:
            output = self.parse_args_into_dataclasses()

        if len(output) == 1:
            output = output[0]
        return output
    
@dataclass
class RopeScalingConfig:
    type: str = field(
        default="dynamic",
        metadata={"help": ("The type of scaling to use. either `dynamic` or `linear`.")},
    )
    factor: Optional[float] = field(
        default=2.0,
        metadata={"help": ("The factor to use for scaling. https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/")},
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """

    base_model_revision: Optional[str] = field(
        default=None,
        metadata={"help": ("The base model checkpoint for weights initialization with PEFT adatpers.")},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    model_code_revision: str = field(default=None, metadata={"help": "The branch of the IFT model"})
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code when loading a model."})
    use_flash_attention_2: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use flash attention 2. You must install this manually by running `pip install flash-attn --no-build-isolation`"
            )
        },
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": ("Whether to use PEFT or not for training.")},
    )
    lora_r: Optional[int] = field(
        default=16,
        metadata={"help": ("LoRA R value.")},
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": ("LoRA alpha.")},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": ("LoRA dropout.")},
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("LoRA target modules.")},
    )
    lora_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("Model layers to unfreeze & train")},
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "use 8 bit precision"})
    load_in_4bit: bool = field(default=False, metadata={"help": "use 4 bit precision"})

    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"}
    )
    use_bnb_nested_quant: bool = field(default=False, metadata={"help": "use nested quantization"})

    rope_scaling: Optional[RopeScalingConfig] = field(
        default=None,
        metadata={"help": ("The properties to define the rope scaling.")},
    )

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")

        if self.rope_scaling:
            self.rope_scaling = RopeScalingConfig(**self.rope_scaling)
@dataclass
class ConvertToHFChatTemplateConfig:
    name: str = "convert_to_hf_chat_template"

    key: Optional[str] = field(
        default="conversations",
        metadata={"help": ("The key to use in the dataset.")},
    )
    role_key: Optional[str] = field(
        default="from",
        metadata={"help": ("The role key to use in the dataset.")},
    )
    content_key: Optional[str] = field(
        default="value",
        metadata={"help": ("The content key to use in the dataset.")},
    )
    system_value: Optional[str] = field(
        default="system",
        metadata={"help": ("The system value to use in the dataset.")},
    )
    user_value: Optional[str] = field(
        default="human",
        metadata={"help": ("The user value to use in the dataset.")},
    )
    assistant_value: Optional[str] = field(
        default="gpt",
        metadata={"help": ("The assistant value to use in the dataset.")},
    )

@dataclass
class RoleBasedConverterConfig:
    name: str = "role_based_converter"
    system_keys: Optional[List[str]] = field(
        default_factory=list,
        metadata={"help": ("The system keys to use in the dataset.")},
    )
    user_keys: Optional[List[str]] = field(
        default_factory=lambda: ["instruction", "input"],
        metadata={"help": ("The user keys to use in the dataset.")},
    )
    assistant_keys: Optional[List[str]] = field(
        default_factory=lambda: ["output"],
        metadata={"help": ("The assistant keys to use in the dataset.")},
    )
    seperator: Optional[str] = field(
        default="\n",
        metadata={"help": ("The seperator to within the content if multiple keys are used.")},
    )

@dataclass
class RoleBasedConverterDPOConfig:
    name: str = "role_based_converter_dpo"
    system_keys: Optional[List[str]] = field(
        default_factory=lambda: ["system"],
        metadata={"help": ("The system keys to use in the dataset.")},
    )
    user_keys: Optional[List[str]] = field(
        default_factory=lambda: ["question"],
        metadata={"help": ("The user keys to use in the dataset.")},
    )
    chosen_keys: Optional[List[str]] = field(
        default_factory=lambda: ["chosen"],
        metadata={"help": ("The assistant keys to use in the dataset.")},
    )
    rejected_keys: Optional[List[str]] = field(
        default_factory=lambda: ["rejected"],
        metadata={"help": ("The assistant keys to use in the dataset.")},
    )
    seperator: Optional[str] = field(
        default="\n",
        metadata={"help": ("The seperator to within the content if multiple keys are used.")},
    )


@dataclass
class DatasetMixerConfig:
    dataset: str = field(metadata={"help": ("The dataset to use.")})
    proportion: float = field(default=1.0, metadata={"help": ("The proportion of the dataset to use.")})
    data_files: Optional[List[str]] = field(
        default_factory=list,
        metadata={"help": ("The data files to use in the dataset.")},
    )
    random_seed: Optional[int] = field(
        default=42,
        metadata={"help": ("The random seed to use in the dataset.")},
    )
    dataset_splits: Optional[List[str]] = field(
        default_factory=lambda: ["train", "test"],
        metadata={"help": ("List of train test splits to use in the dataset")},
    )
    blacklist_sources: Optional[List[str]] = field(
        default=list,
        metadata={"help": ("Filter the sources to use in the dataset.")},
    )
    blacklist_sources_key: Optional[str] = field(
        default=None,
        metadata={"help": ("Filter the sources to use in the dataset.")},
    )
    converter: Optional[RoleBasedConverterConfig | ConvertToHFChatTemplateConfig] = field(
        default=None,
        metadata={"help": ("The properties to define the role based converter.")},
    )

    def __post_init__(self):
        if self.converter:
            if not "name" in self.converter:
                raise ValueError("Converter must have a name.")

            if self.converter["name"] == ConvertToHFChatTemplateConfig.name:
                self.converter = ConvertToHFChatTemplateConfig(**self.converter)
            elif self.converter["name"] == RoleBasedConverterConfig.name:
                self.converter = RoleBasedConverterConfig(**self.converter)
            elif self.converter["name"] == RoleBasedConverterDPOConfig.name:
                self.converter = RoleBasedConverterDPOConfig(**self.converter)
            else:
                raise ValueError(f"Unknown converter name: {self.converter['name']}")


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    dataset_mixer: Optional[List[DatasetMixerConfig]] = field(
        default=None,
        metadata={"help": ("The properties to define the dataset processing.")},
    )
    dataset_splits: Optional[List[str]] = field(
        default_factory=lambda: ["train", "test"],
        metadata={"help": ("List of train test splits to use in the dataset")},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    truncation_side: Optional[str] = field(
        default=None, metadata={"help": "Truncation side to use for the tokenizer."}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": ("Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.")},
    )

    def __post_init__(self):
        dataset_mixer_data = []
        for dataset in self.dataset_mixer:
            dataset_mixer_data.append(DatasetMixerConfig(**dataset))
        self.dataset_mixer = dataset_mixer_data



@dataclass
class SFTConfig(transformers.TrainingArguments):
    """
    Arguments related to the training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/trainer#transformers.TrainingArguments
    """

    max_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": ("Used by TRL for reward model training, which tries to read this parameter in init.")},
    )
    logging_first_step: bool = field(
        default=True,
        metadata={"help": ("Whether to log and evaluate the first global_step or not.")},
    )
    optim: Optional[str] = field(default="adamw_torch")


@dataclass
class DPOConfig(transformers.TrainingArguments):
    """
    Arguments related to the DPO training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/trainer#transformers.TrainingArguments
    """

    beta: Optional[float] = field(
        default=0.1,
        metadata={"help": "The beta factor in DPO loss. Higher beta means less divergence from the initial policy."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": ("The Hub model branch to push the model to.")},
    )
    logging_first_step: bool = field(
        default=True,
        metadata={"help": ("Whether to log and evaluate the first global_step or not.")},
    )
    max_prompt_length: Optional[int] = field(
        default=None,
        metadata={"help": ("For DPO, the maximum length of the prompt to use for conditioning the model.")},
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={"help": ("Used by TRL for reward model training, which tries to read this parameter in init.")},
    )
    optim: Optional[str] = field(default="rmsprop")
    remove_unused_columns: bool = field(default=False)
