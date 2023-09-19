from typing import (
    Any,
    List,
    Optional,
)

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, ChatResult

from weaverbird.config_factory import FinetuningConfig, GenerationConfig, BaseModelConfig
from weaverbird.models.llm_model import LLMModel


class WeaverBirdChat(BaseChatModel):
    model_name: str = "weaverbird_chat"
    """model name of WeaverBird, default is `weaverbird_chat`"""

    request_timeout: Optional[int] = 60
    """request timeout for chat http requests"""

    max_retries: int = 6
    """Maximum number of retries to make when generating"""

    streaming: Optional[bool] = True
    """streaming mode. not supported yet"""

    llm_model: Optional[LLMModel] = None
    """LLM model to use in weaverbird"""

    retriever_model: Optional[LLMModel] = None
    """retriever model to use in weaverbird"""

    @classmethod
    def build_from_config(cls,
                          llm_model_config: BaseModelConfig,
                          llm_finetuning_config: Optional[FinetuningConfig] = None,
                          llm_generation_config: Optional[GenerationConfig] = None):
        llm_model = LLMModel(model_config=llm_model_config,
                             finetuning_config=llm_finetuning_config,
                             generation_config=llm_generation_config)



        return cls(llm_model=llm_model)

        if retro_config is not None:
            self.retriever = None

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        pass

    @property
    def _llm_type(self) -> str:
        return "weaverbird_chat"
