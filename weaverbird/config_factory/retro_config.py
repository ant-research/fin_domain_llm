from dataclasses import dataclass, field
from typing import Literal, Optional

from weaverbird.config_factory.base_config import BaseConfig


@dataclass
class RetroConfig(BaseConfig):
    retro_name: Optional[Literal["web_searcher_retro", "adaptive_retro"]] = field(
        default="web_searcher_retro",
        metadata={"help": "Which retro model cls to initialize."}
    )

    serp_api_token: Optional[bool] = field(
        default=None,
        metadata={"help": "Token for serpapi. See https://serpapi.com/search-api"}
    )

    engine: Optional[Literal["google", "bing", "baidu"]] = field(
        default="google",
        metadata={"help": "Which search engine to use."}
    )

    num_search_results: Optional[int] = field(
        default=10,
        metadata={"help": "Number of pages per search"}
    )
