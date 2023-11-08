from weaverbird.models.chat_glm2 import ChatGLM2
from weaverbird.models.chat_llama2 import ChatLlama2
from weaverbird.models.chat_weaverbird import ChatWeaverBird
from weaverbird.models.llm_loader import load_model_and_tokenizer

__all__ = ['load_model_and_tokenizer',
           'ChatGLM2',
           'ChatLlama2',
           'ChatWeaverBird']


class BaseModel:
    @staticmethod
    def build_from_config(model_config, **kwargs):
        """Build up the runner from runner config.

        Args:
            runner_config (RunnerConfig): config for the runner.

        Returns:
            Runner: the corresponding runner class.
        """
        if 'glm' in model_config.model_name_or_path.lower():
            model_cls = ChatGLM2
        else:
            model_cls = ChatLlama2
        return model_cls(model_config, **kwargs)
