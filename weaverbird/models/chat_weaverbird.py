from typing import (
    Any,
    List,
    Optional, )

from langchain import PromptTemplate
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import LLM
from langchain.schema import (
    BaseMessage,
    ChatResult, Document,
)
from langchain.schema import BaseRetriever

from weaverbird.utils.misc import get_current_time


class ChatWeaverBird(BaseChatModel):
    model_name: str = "chat_weaverbird"
    """model name of WeaverBird, default is `chat_weaverbird`"""

    llm_model: LLM
    """LLM model to use in weaverbird"""

    retriever_model: Optional[BaseRetriever] = None
    """retriever model to use in weaverbird"""

    prompt_template: PromptTemplate
    """template to construct the prompt  """

    streaming: bool = False
    """Whether to stream the results or not."""

    def __init__(self, llm_model, retriever_model, prompt_template):
        super(ChatWeaverBird, self).__init__()
        self.llm_model = llm_model
        self.retriever_model = retriever_model
        self.prompt_template = prompt_template

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            stream: Optional[bool] = None,
            **kwargs: Any,
    ) -> ChatResult:
        chat_history = kwargs.get('chat_history', [])
        docs = []
        if self.retriever_model is not None:
            docs = self.retriever_model._get_relevant_documents()

        should_stream = stream if stream is not None else self.streaming

        if len(docs) > 0:
            prompt = self._generate_prompt(docs, messages)
        else:
            prompt = messages

        for answer_result in self.llm_model._generate_answer(prompt=prompt,
                                                             history=chat_history,
                                                             streaming=should_stream):
            resp = answer_result.generatios
            history = answer_result.llm_output['history']
            history[-1][0] = messages
            response = {
                "prompt": prompt,
                "query": messages,
                "result": resp,
                "source_documents": docs
            }
            yield response, history

    def _generate_prompt(self,
                         related_docs: List[Document],
                         query: List[BaseMessage]):
        cur_time = get_current_time()

        if len(related_docs):
            context = "\n".join(
                [f"{doc.metadata.get('date', '')} {doc.metadata.get('title', '')} {doc.page_content}" for doc in
                 related_docs])
        else:
            context = ''
        # do a concate for query here
        query = ''.join(query)
        kwargs = {'question': query, 'date': cur_time, 'context': context}
        return self.prompt_template.format(kwargs)

    @property
    def _llm_type(self) -> str:
        return "chat_weaverbird"
