from typing import Literal, Optional
from dataclasses import dataclass, field

from weaverbird.config_factory.base_config import BaseConfig


@dataclass
class WebSearchConfig(BaseConfig):
    serp_api_token: Optional[bool] = field(
        default=None,
        metadata={"help": "Token for serpapi. See https://serpapi.com/search-api"}
    )

    engine: Optional[Literal["google", "bing", "baidu"]] = field(
        default="google",
        metadata={"help": "Which search engine to use."}
    )

    num_search_results: Optional[int] = field(
        default=5,
        metadata={"help": "Number of pages per search"}
    )


