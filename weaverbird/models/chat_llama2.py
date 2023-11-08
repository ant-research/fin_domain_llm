from typing import Any, List, Optional, Tuple, Dict

import torch
from langchain.llms.base import LLM
from transformers import GenerationConfig
from transformers import PreTrainedModel, PreTrainedTokenizer

from weaverbird.config_factory import BaseModelConfig, FinetuningConfig
from weaverbird.config_factory import GenerationConfig as WBGenerationConfig
from weaverbird.models.llm_loader import load_model_and_tokenizer
from weaverbird.models.template import get_template_and_fix_tokenizer, Template
from weaverbird.utils import dispatch_model, get_logits_processor


class ChatLlama2(LLM):
    """
    LLAMA2 from Meta

    Borrowed from https://github.com/hiyouga/LLaMA-Efficient-Tuning/blob/469f859161dec0e34f4cc849f20e43d442680b5c/src/llmtuner/chat/stream_chat.py
    """
    model: Optional[PreTrainedModel] = None

    tokenizer: Optional[PreTrainedTokenizer] = None

    generation_config: Optional[WBGenerationConfig] = None

    template: Optional[Template] = None

    def __init__(
            self,
            model_config: BaseModelConfig,
            finetuning_config: Optional[FinetuningConfig] = None,
            generation_config: Optional[WBGenerationConfig] = None
    ) -> None:
        super(ChatLlama2, self).__init__()
        self.model, self.tokenizer = load_model_and_tokenizer(model_config, finetuning_config)
        self.model = dispatch_model(self.model)
        self.model = self.model.eval()  # enable evaluation mode
        self.generation_config = generation_config
        self.template = get_template_and_fix_tokenizer("llama2", self.tokenizer)

    @classmethod
    def build_from_config(cls, configs):
        return cls(model_config=configs['model_config'], finetuning_config=configs['finetuning_config'],
                   generation_config=configs['generation_config'])

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        response, _ = self.chat(
            prompt,
            history=[]
        )
        print(f"response:{response}")
        print(f"+++++++++++++++++++++++++++++++++++")
        return response

    def process_args(
            self,
            query: str,
            history: Optional[List[Tuple[str, str]]] = None,
            system: Optional[str] = None,
            **input_kwargs
    ) -> Tuple[Dict[str, Any], int]:
        system = system or ""

        prompt, _ = self.template.encode_oneturn(
            tokenizer=self.tokenizer, query=query, resp="", history=history, system=system
        )
        input_ids = torch.tensor([prompt], device=self.model.device)
        prompt_length = len(input_ids[0])

        do_sample = input_kwargs.pop("do_sample", None)
        temperature = input_kwargs.pop("temperature", None)
        top_p = input_kwargs.pop("top_p", None)
        top_k = input_kwargs.pop("top_k", None)
        repetition_penalty = input_kwargs.pop("repetition_penalty", None)
        max_length = input_kwargs.pop("max_length", None)
        max_new_tokens = input_kwargs.pop("max_new_tokens", None)

        generation_config = self.generation_config.dict()
        generation_config.update(dict(
            do_sample=do_sample if do_sample is not None else generation_config["do_sample"],
            temperature=temperature or generation_config["temperature"],
            top_p=top_p or generation_config["top_p"],
            top_k=top_k or generation_config["top_k"],
            repetition_penalty=repetition_penalty or generation_config["repetition_penalty"],
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            pad_token_id=self.tokenizer.pad_token_id
        ))

        if max_length:
            generation_config.pop("max_new_tokens", None)
            generation_config["max_length"] = max_length

        if max_new_tokens:
            generation_config.pop("max_length", None)
            generation_config["max_new_tokens"] = max_new_tokens

        gen_kwargs = dict(
            inputs=input_ids,
            generation_config=GenerationConfig(**generation_config),
            logits_processor=get_logits_processor()
        )

        return gen_kwargs, prompt_length

    @torch.inference_mode()
    def chat(
            self,
            prompt: str,
            history: Optional[List[Tuple[str, str]]] = None,
            system: Optional[str] = None,
            **input_kwargs
    ) -> Tuple[str, Tuple[int, int]]:
        gen_kwargs, prompt_length = self.process_args(prompt, history, system, **input_kwargs)
        generation_output = self.model.generate(**gen_kwargs)
        outputs = generation_output.tolist()[0][prompt_length:]
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        response_length = len(outputs)
        return response, (prompt_length, response_length)

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "chat_llama2"
