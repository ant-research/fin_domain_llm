from typing import Any, List, Optional

from langchain.llms.base import LLM

from weaverbird.config_factory import BaseModelConfig, FinetuningConfig, GeneratingConfig
from weaverbird.models import load_model_and_tokenizer
from weaverbird.utils import dispatch_model


class LLMChatModel(LLM):
    def __init__(
            self,
            model_config: BaseModelConfig,
            finetuning_config: FinetuningConfig,
            generating_config: GeneratingConfig
    ) -> None:
        super(LLMChatModel, self).__init__()
        self.model, self.tokenizer = load_model_and_tokenizer(model_config, finetuning_config)
        self.model = dispatch_model(self.model)
        self.model = self.model.eval()  # enable evaluation mode
        self.generating_args = generating_config

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=[],
            max_length=self.generating_args.max_length,
            temperature=self.generating_args.temperature
        )
        print(f"response:{response}")
        print(f"+++++++++++++++++++++++++++++++++++")
        return response
