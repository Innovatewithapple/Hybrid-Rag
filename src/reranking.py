class Reranking:
    def __init__(self, model, top_k):
        self.rerankingmodel = model
        self.top_k = top_k

    def rerank_documents(self, question, docs):
        """Return individually ranked Document objects, preserving metadata."""
        if not docs:
            return []
        pairs = [(question, doc.page_content) for doc in docs]
        scores = self.rerankingmodel.predict(pairs)
        reranked = sorted(
            zip(docs, scores), key=lambda item: float(item[1]), reverse=True
        )
        return [doc for doc, _score in reranked[: self.top_k]]

    def __call__(self, inputs):
        question = inputs["question"]
        docs = inputs["context"]
        top_docs = self.rerank_documents(question, docs)
        # Preserve the existing chain's expected output shape.
        return {
            "question": question,
            "context": "\n\n".join(doc.page_content for doc in top_docs),
        }
