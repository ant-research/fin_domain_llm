from typing import Any, List, Optional

from langchain.llms.base import LLM
from langchain.schema import BaseMessage, ChatResult
from transformers import PreTrainedModel, PreTrainedTokenizer

from weaverbird.config_factory import BaseModelConfig, FinetuningConfig, GenerationConfig
from weaverbird.models.llm_loader import load_model_and_tokenizer
from weaverbird.utils import dispatch_model
from weaverbird.utils.misc import torch_gc


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

    @classmethod
    def build_from_config(cls, configs):
        return cls(model_config=configs['model_config'], finetuning_config=configs['finetuning_config'],
                   generation_config=configs['generation_config'])

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        history = kwargs.get('hisotry', [])
        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=history,
            max_length=self.generation_config.max_length,
            temperature=self.generation_config.temperature
        )
        print(f"response:{response}")
        return response

    def _generate_answer(self, prompt: str, history: List[List[BaseMessage]] = [], streaming: bool = False):
        if streaming:
            history += [[]]
            for inum, (stream_resp, _) in enumerate(self.model.stream_chat(
                    self.tokenizer,
                    prompt,
                    history=history[
                            -self.generation_config.max_history_message_length:-1] if self.generation_config.max_history_message_length > 1 else [],
                    max_length=self.generation_config.max_length,
                    temperature=self.generation_config.temperature
            )):
                history[-1] = [prompt, stream_resp]
                llm_output = {'history': history}
                yield ChatResult(generations=stream_resp, llm_output=llm_output)
        else:
            response, _ = self.model.chat(
                self.tokenizer,
                prompt,
                history=history[
                        -self.generation_config.max_history_message_length:-1] if self.generation_config.max_history_message_length > 1 else [],
                max_length=self.generation_config.max_length,
                temperature=self.generation_config.temperature
            )
            torch_gc()
            history += [[prompt, response]]
            llm_output = {'history': history}
            yield ChatResult(generations=response, llm_output=llm_output)

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "chat_glm2"
