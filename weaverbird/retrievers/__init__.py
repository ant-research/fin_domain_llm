from weaverbird.config_factory import RetroConfig
from weaverbird.retrievers.web_searcher import WebSearcher

__all__ = ['WebSearcher']


class BaseRetro:
    @staticmethod
    def build_from_config(retro_config: RetroConfig):

        for retro_cls in __all__:
            if eval(retro_cls).Config.retro_name == retro_config.retro_name:
                return eval(retro_cls).build_from_config(retro_config)

        raise NotImplementedError('Retro Model retro_config.retro_name not implemented.')
