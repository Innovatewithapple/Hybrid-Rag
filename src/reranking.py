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



from langchain_core.runnables import RunnableLambda

class ParentChildReranking:
    def __init__(self, model, top_k):
        self.rerankingmodel = model
        self.top_k = top_k

    def __call__(self, inputs):
        question = inputs["question"]
        docs = inputs["context"]
        
        if not docs:
            return {"question": question, "context": []}

        # 1. Deduplicate chunks by their Parent ID
        unique_docs = []
        seen_parent_ids = set()
        for doc in docs:
            p_id = doc.metadata.get('parent_id')
            if p_id not in seen_parent_ids:
                seen_parent_ids.add(p_id)
                unique_docs.append(doc)

        # 2. Score using the Full Parent Context
        pairs = [
            (question, f"{doc.metadata.get('article', '')}\n{doc.metadata.get('parent_text', doc.page_content)}")
            for doc in unique_docs
        ]
        scores = self.rerankingmodel.predict(pairs)
        
        # 3. Sort documents by their relevancy score
        reranked = sorted(
            zip(unique_docs, scores), key=lambda item: float(item[1]), reverse=True
        )
        
        top_docs = [doc for doc, _score in reranked[: self.top_k]]
        
        # 4. FIX: Return top_docs as a LIST of Document objects
        return {
            "question": question,
            "context": top_docs  # Keep this as a list, do not join into a string!
        }