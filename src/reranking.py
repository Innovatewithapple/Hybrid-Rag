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
import time
class ParentChildReranking:
    def __init__(self, model, top_k):
        self.rerankingmodel = model
        self.top_k = top_k


    def __call__(self, inputs):
        total_start = time.perf_counter()

        question = inputs["question"]
        docs = inputs["context"]

        if not docs:
            return {"question": question, "context": []}

        # Deduplication
        dedup_start = time.perf_counter()

        unique_docs = []
        seen_parent_ids = set()

        for doc in docs:
            p_id = doc.metadata.get("parent_id")
            if p_id not in seen_parent_ids:
                seen_parent_ids.add(p_id)
                unique_docs.append(doc)

        dedup_time = time.perf_counter() - dedup_start

        # Prepare pairs
        pairs_start = time.perf_counter()

        pairs = [
            (
                question,
                (f"{doc.metadata['article']}\n"
                if doc.metadata.get("article") else "")
                + doc.metadata.get("parent_text", doc.page_content)
            )
            for doc in unique_docs
        ]

        

        pair_lengths = [len(text) for _, text in pairs]

        pair_prep_time = time.perf_counter() - pairs_start

        # BGE inference
        inference_start = time.perf_counter()

        scores = self.rerankingmodel.predict(pairs)

        inference_time = time.perf_counter() - inference_start

        # Sorting
        sorting_start = time.perf_counter()

        reranked = sorted(
            zip(unique_docs, scores),
            key=lambda item: float(item[1]),
            reverse=True
        )

        top_docs = [doc for doc, _ in reranked[:self.top_k]]

        sorting_time = time.perf_counter() - sorting_start
        total_time = time.perf_counter() - total_start

        print("\n--- RERANKER PROFILING ---")
        print(f"Input documents: {len(docs)}")
        print(f"Unique parents: {len(unique_docs)}")
        print(f"Pairs scored: {len(pairs)}")
        print(f"Average parent characters: {sum(pair_lengths) / len(pair_lengths):.0f}")
        print(f"Longest parent characters: {max(pair_lengths)}")
        print(f"Deduplication: {dedup_time:.3f}s")
        print(f"Pair preparation: {pair_prep_time:.3f}s")
        print(f"BGE inference: {inference_time:.3f}s")
        print(f"Sorting: {sorting_time:.3f}s")
        print(f"Total reranking: {total_time:.3f}s")
        print("--------------------------\n")

        return {
            "question": question,
            "context": top_docs
        }