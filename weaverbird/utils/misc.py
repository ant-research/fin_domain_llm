import os
import sys
import time
from typing import Tuple, Optional, Dict, Any

import torch
from transformers import HfArgumentParser
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from transformers.modeling_utils import PreTrainedModel

from weaverbird.config_factory import BaseModelConfig, FinetuningConfig, GenerationConfig, WebSearchConfig


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
        GenerationConfig,
        WebSearchConfig
    ))

    parsed_config =  _parse_args(parser, configs)
    return {'model_config': parsed_config[0],
            'finetuning_config': parsed_config[1],
            'generation_config': parsed_config[2],
            'websearch_config': parsed_config[3]}


def _parse_args(parser: HfArgumentParser, args: Optional[Dict[str, Any]] = None) -> Tuple[Any]:
    if args is not None:
        return parser.parse_dict(args)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(os.path.abspath(sys.argv[1]))
    else:
        return parser.parse_args_into_dataclasses()


def torch_gc() -> None:
    """Collects GPU memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    """Configures device map for ChatGLM2.

    Source: https://github.com/hiyouga/ChatGLM-Efficient-Tuning
    """
    num_layers = 28
    layers_per_gpu = 30 / num_gpus
    device_map = {
        "transformer.embedding.word_embeddings": 0,
        "transformer.encoder.final_layernorm": 0,
        "transformer.output_layer": 0,
        "transformer.rotary_pos_emb": 0,
        "transformer.prefix_encoder": 0,
        "lm_head": 0
    }

    added_layers = 2
    target_gpu = 0

    for i in range(num_layers):
        if added_layers >= layers_per_gpu:
            target_gpu += 1
            added_layers = 0
        assert target_gpu < num_gpus
        device_map[f"transformer.encoder.layers.{i}"] = target_gpu
        added_layers += 1

    return device_map


def get_current_time():
    now = time.time()
    time_arr = time.localtime(now)
    return time.strftime("%Y-%m-%d", time_arr)


# Avoid runtime error in model.generate(do_sample=True).
# Borrowed from: https://huggingface.co/THUDM/chatglm-6b/blob/658202d88ac4bb782b99e99ac3adff58b4d0b813/modeling_chatglm.py#L54
class InvalidScoreLogitsProcessor(LogitsProcessor):

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


def get_logits_processor() -> LogitsProcessorList:
    logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    return logits_processor


def dispatch_model(model: PreTrainedModel) -> PreTrainedModel:
    """Dispatches a pre-trained model to GPUs with balanced memory.

    Source: https://github.com/hiyouga/LLaMA-Efficient-Tuning/blob/main/src/llmtuner/extras/misc.py
    """
    if torch.cuda.device_count() > 1:
        from accelerate import dispatch_model

        if 'chatglm' in model.name_or_path:
            device_map = auto_configure_device_map(torch.cuda.device_count())
        else:
            from accelerate.utils import infer_auto_device_map, get_balanced_memory

            if model._no_split_modules is None:
                raise ValueError("The model class needs to implement the `_no_split_modules` attribute.")

            kwargs = {"dtype": model.dtype, "no_split_module_classes": model._no_split_modules}
            max_memory = get_balanced_memory(model, **kwargs)
            device_map = infer_auto_device_map(model, max_memory=max_memory, **kwargs)

        model.tie_weights()
        return dispatch_model(model, device_map)
    else:
        return model.cuda()
