import concurrent.futures
from typing import List, Optional

import numpy as np
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import Field
from langchain.schema import BaseRetriever, Document
from langchain.utilities import SerpAPIWrapper
from langchain.vectorstores import VectorStore

from weaverbird.config_factory import WebSearchConfig


def create_index(contexts: List[str], embeddings: Embeddings) -> np.ndarray:
    """
    Create an index of embeddings for a list of contexts.

    Borrowed from https://github.com/langchain-ai/langchain/blob/ddd07001f354cd09a76a61e1f5c678bf885506d2/
    libs/langchain/langchain/retrievers/knn.py

    Args:
        contexts: List of contexts to embed.
        embeddings: Embeddings model to use.

    Returns:
        Index of embeddings.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return np.array(list(executor.map(embeddings.embed_query, contexts)))


class AdaptiveRetriever(BaseRetriever):
    local_kb: Optional[VectorStore] = None
    """local knowledge base to store documents."""

    web_searcher: SerpAPIWrapper = Field(..., description="Web Search API Wrapper")

    @classmethod
    def build_from_config(cls, search_config: WebSearchConfig):
        pass

    def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
