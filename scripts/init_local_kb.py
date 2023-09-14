from typing import List

from langchain import FAISS
from langchain.text_splitter import TextSplitter

from weaverbird.document_loaders import LocalKnowledgeBaseLoader
from weaverbird.embeddings import QueryRefEncoder


class NewFinDocTextSplitter(TextSplitter):
    def __init__(self):
        super(NewFinDocTextSplitter, self).__init__()

    def split_text(self, text: str) -> List[str]:
        # split by "Doc" because text has \n
        documents = text.split("Doc ")[1:]

        return documents


def main():
    text_splitter = NewFinDocTextSplitter()
    loader = LocalKnowledgeBaseLoader("report_cn_v0724.txt", text_splitter=text_splitter)
    docs = loader.load()
    print(len(docs))

    model_dir = 'encoder'
    embeddings = QueryRefEncoder(model_dir=model_dir)
    db = FAISS.from_documents(docs, embeddings)

    query = "迈瑞医疗(300760)2022 年三季报发布的业绩是多少"
    docs = db.similarity_search(query)

    print(docs[0].page_content)

    return


if __name__ == '__main__':
    main()
