import os
import sys
from typing import Tuple, Optional, Dict, Any

import torch
from transformers import HfArgumentParser

from weaverbird.config_factory import BaseModelConfig, FinetuningConfig, GeneratingConfig


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Returns the number of trainable parameters and number of all parameters in the model.
    source: https://github.com/hiyouga/LLaMA-Efficient-Tuning
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by 2
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def parse_configs(configs: Optional[Dict[str, Any]] = None):
    parser = HfArgumentParser((
        BaseModelConfig,
        FinetuningConfig,
        GeneratingConfig
    ))

    return _parse_args(parser, configs)


def _parse_args(parser: HfArgumentParser, args: Optional[Dict[str, Any]] = None) -> Tuple[Any]:
    if args is not None:
        return parser.parse_dict(args)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(os.path.abspath(sys.argv[1]))
    else:
        return parser.parse_args_into_dataclasses()
