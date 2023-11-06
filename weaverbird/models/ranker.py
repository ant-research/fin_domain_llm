from weaverbird.utils import Registrable


class ReRanker(Registrable):
    def __init__(self, top_j):
        self.top_j = top_j

    def rank(self, docs):
        pass


@ReRanker.register(name='score_reranker')
class ScoreReranker(ReRanker):
    def __init__(self, top_j=5):
        super(ScoreReranker, self).__init__(top_j=top_j)

    def rank(self, docs):
        docs.sort(key=lambda x: x.metadata["score"], reverse=True)
        return docs[:self.top_j]
