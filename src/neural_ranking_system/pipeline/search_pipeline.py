class SearchPipeline:
    def __init__(self, retriever, ranker=None):
        self.retriever = retriever
        self.ranker = ranker

    def run(self, query, retrieve_k=10, final_k=5):
        docs = self.retriever.retrieve(query, retrieve_k)

        if self.ranker:
            docs = self.ranker.rerank(query, docs)

        return docs[:final_k]