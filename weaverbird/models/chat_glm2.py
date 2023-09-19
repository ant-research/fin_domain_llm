from typing import Any, List, Optional

from langchain.llms.base import LLM
from transformers import PreTrainedModel, PreTrainedTokenizer

from weaverbird.config_factory import BaseModelConfig, FinetuningConfig, GenerationConfig
from weaverbird.models import load_model_and_tokenizer
from weaverbird.utils import dispatch_model


class ChatGLM2(LLM):
    """ GLM2 from THU  """
    model: Optional[PreTrainedModel] = None

    tokenizer: Optional[PreTrainedTokenizer] = None

    generation_config: Optional[GenerationConfig] = None

    def __init__(
            self,
            model_config: BaseModelConfig,
            finetuning_config: Optional[FinetuningConfig] = None,
            generation_config: Optional[GenerationConfig] = None
    ) -> None:
        super(ChatGLM2, self).__init__()
        self.model, self.tokenizer = load_model_and_tokenizer(model_config, finetuning_config)
        self.model = dispatch_model(self.model)
        self.model = self.model.eval()  # enable evaluation mode
        self.generation_config = generation_config

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=[],
            max_length=self.generation_config.max_length,
            temperature=self.generation_config.temperature
        )
        print(f"response:{response}")
        print(f"+++++++++++++++++++++++++++++++++++")
        return response

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "glm2"
