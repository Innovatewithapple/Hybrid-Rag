class Reranking:
    def __init__(self,model,top_k):
        self.rerankingmodel = model
        self.top_k = top_k

    def __call__(self, inputs):
        question = inputs['question']
        docs = inputs['context']

        pairs = [(question, doc.page_content) for doc in docs]
        scores = self.rerankingmodel.predict(pairs)

        reranked = sorted(zip(docs,scores),key=lambda item: float(item[1]),reverse=True)
        top_docs = reranked[:self.top_k]

        return {
            "question":question,
            "context":"\n\n".join(doc.page_content for doc, score in top_docs)
        }