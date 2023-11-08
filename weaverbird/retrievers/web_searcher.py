from typing import List

import dateparser
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.pydantic_v1 import Field
from langchain.schema import BaseRetriever, Document
from langchain.utilities import SerpAPIWrapper

from weaverbird.config_factory import RetroConfig
from weaverbird.utils import logger


class WebSearcher(BaseRetriever):
    """Simplest WebSearcher
    borrowed from
    https://github.com/langchain-ai/langchain/blob/ddd07001f354cd09a76a61e1f5c678bf885506d2/libs/langchain/langchain/retrievers/web_research.py
    """

    web_searcher: SerpAPIWrapper = Field(..., description="Web Search API Wrapper")

    class Config:

        """Configuration for this pydantic object."""

        retro_name = 'web_searcher_retro'

    @classmethod
    def build_from_config(cls, search_config: RetroConfig):
        serpapi_api_key = search_config.serp_api_token
        search_config_dict = search_config.dict()
        search_config_dict.pop('serp_api_token')
        return cls(web_searcher=SerpAPIWrapper(serpapi_api_key=serpapi_api_key, params=search_config_dict))

    def _search_result2docs(self, search_results):
        docs = []
        logger.info(f'# search_results {len(search_results)}')
        for result in search_results:
            doc = Document(page_content=result["snippet"].replace('\n', '') if "snippet" in result.keys() else "",
                           metadata={"link": result["link"] if "link" in result.keys() else "",
                                     "title": result["title"] if "title" in result.keys() else "",
                                     "source": result["source"] if "source" in result.keys() else "",
                                     "filename": result["title"] if "title" in result.keys() else "",
                                     "date": dateparser.parse(result['date']).strftime(
                                         "%Y-%m-%d") if 'date' in result.keys() else "",
                                     "score": 100})  # for the moment we fix the score
            docs.append(doc)
        return docs

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Search websites for documents related to the query input.

        Args:
            query: user query

        Returns:
            Relevant documents from various urls.
        """
        search_results = self.web_searcher.results(self.clean_search_query(query))

        return self._search_result2docs(search_results['organic_results'])

    def clean_search_query(self, query: str) -> str:
        # Some search tools (e.g., Google) will
        # fail to return results if query has a
        # leading digit: 1. "LangCh..."
        # Check if the first character is a digit
        if query[0].isdigit():
            # Find the position of the first quote
            first_quote_pos = query.find('"')
            if first_quote_pos != -1:
                # Extract the part of the string after the quote
                query = query[first_quote_pos + 1:]
                # Remove the trailing quote if present
                if query.endswith('"'):
                    query = query[:-1]
        return query.strip()
