from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Tuple, Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase
)
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils.versions import require_version

from weaverbird.config_factory import BaseModelConfig, FinetuningConfig
from weaverbird.utils import logger, count_parameters

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


def load_model_and_tokenizer(
        model_config: BaseModelConfig,
        finetuning_args: Optional[FinetuningConfig] = None
) -> Tuple[PreTrainedModel, "PreTrainedTokenizer"]:
    """
    Loads pretrained model and tokenizer.
    source: https://github.com/hiyouga/LLaMA-Efficient-Tuning
    """

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_config.cache_dir,
        "revision": model_config.model_revision,
        "use_auth_token": True if model_config.use_auth_token else None,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        Path(model_config.model_name_or_path),
        use_fast=model_config.use_fast_tokenizer,
        padding_side=model_config.padding_side,
        **config_kwargs
    )

    if finetuning_args is not None and finetuning_args.finetuning_type == "full" and model_config.checkpoint_dir is not None:
        model_to_load = model_config.checkpoint_dir[0]
    else:
        model_to_load = model_config.model_name_or_path

    config = AutoConfig.from_pretrained(model_to_load, **config_kwargs)

    if hasattr(config, "fp16") and hasattr(config, "bf16"):  # fix Qwen config
        if model_config.compute_dtype == torch.bfloat16:
            setattr(config, "bf16", True)
        else:
            setattr(config, "fp16", True)

    # Set RoPE scaling
    if model_config.rope_scaling is not None:
        if hasattr(config, "use_dynamic_ntk"):  # for Qwen models
            setattr(config, "use_dynamic_ntk", True)
            setattr(config, "use_logn_attn", True)
            logger.info("Using dynamic NTK scaling.")

        elif hasattr(config, "rope_scaling"):  # for LLaMA models
            require_version("transformers>=4.31.0", "RoPE scaling requires transformers>=4.31.0")

            scaling_factor = 2.0

            setattr(config, "rope_scaling", {"type": model_config.rope_scaling, "factor": scaling_factor})
            logger.info("Using {} scaling strategy and setting scaling factor to {}".format(
                model_config.rope_scaling, scaling_factor
            ))

        else:
            logger.warning("Current model does not support RoPE scaling.")

    # Quantization configurations (using bitsandbytes library).
    if model_config.quantization_bit is not None:
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        if model_config.quantization_bit == 8:
            require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")
            config_kwargs["load_in_8bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        elif model_config.quantization_bit == 4:
            require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
            config_kwargs["load_in_4bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=model_config.compute_dtype,
                bnb_4bit_use_double_quant=model_config.double_quantization,
                bnb_4bit_quant_type=model_config.quantization_type
            )

        config_kwargs["device_map"] = "auto"
        logger.info("Quantizing model to {} bit.".format(model_config.quantization_bit))

    # Load and prepare pre-trained models (without valuehead).
    if 'glm' in model_config.model_name_or_path.lower():
        model = AutoModel.from_pretrained(model_to_load, config=config, **config_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_to_load,
            config=config,
            torch_dtype=model_config.compute_dtype,
            low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
            **config_kwargs
        )

    # Disable custom generate method (for Qwen)
    if "GenerationMixin" not in str(model.generate.__func__):
        model.generate = MethodType(PreTrainedModel.generate, model)

    # Fix LM head (for ChatGLM2)
    if not hasattr(model, "lm_head") and hasattr(model, "transformer"):
        setattr(model, "lm_head", model.transformer.output_layer)

    # Register auto class to save the custom code files.
    if isinstance(config, PretrainedConfig) and "AutoConfig" in getattr(config, "auto_map", {}):
        config.__class__.register_for_auto_class()
    if isinstance(model, PreTrainedModel) and ("AutoModelForCausalLM" in getattr(config, "auto_map", {}) or
                                                 "AutoModel" in getattr(config, "auto_map", {})):
        model.__class__.register_for_auto_class()
    if isinstance(tokenizer, PreTrainedTokenizerBase) and "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        tokenizer.__class__.register_for_auto_class()

    # Prepare model for inference
    model.requires_grad_(False)  # fix all model params
    infer_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16  # detect cuda capability
    model = model.to(infer_dtype) if model_config.quantization_bit is None else model

    _, all_param = count_parameters(model)
    logger.info("num params: {:d}".format(all_param))

    return model, tokenizer
