"""
retriever.py
------------
Builds and queries a FAISS index.
Also applies cross-encoder reranking for better precision.
"""

import faiss
from sentence_transformers import CrossEncoder


class Retriever:
    def __init__(self, dim, rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize FAISS index and reranker.
        """
        self.index = faiss.IndexFlatIP(dim)
        self.reranker = CrossEncoder(rerank_model)

    def build_index(self, corpus_embeddings):
        """
        Add document embeddings to the FAISS index.
        """
        self.index.add(corpus_embeddings)

    def search(self, query_vector, corpus_texts, top_k=5):
        """
        Retrieve top-k documents using FAISS similarity search.
        """
        scores, indices = self.index.search(query_vector, k=top_k)

        results = []
        for rank, idx in enumerate(indices[0]):
            results.append({
                "rank": rank + 1,
                "text": corpus_texts[idx]
            })
        return results

    def rerank(self, query, retrieved_docs):
        """
        Rerank retrieved results using a cross encoder.
        """
        pairs = [[query, d["text"]] for d in retrieved_docs]
        scores = self.reranker.predict(pairs)

        for i, s in enumerate(scores):
            retrieved_docs[i]["rerank_score"] = float(s)

        # Sort by cross-encoder score
        return sorted(retrieved_docs, key=lambda x: x["rerank_score"], reverse=True)
