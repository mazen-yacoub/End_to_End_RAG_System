"""
embedder.py
-----------
Handles embedding of documents and queries using Sentence-Transformers.
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        """
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, docs):
        """
        Convert a list of documents into dense embeddings.
        """
        embeddings = self.model.encode(
            docs,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        faiss.normalize_L2(embeddings)
        return embeddings

    def embed_query(self, query):
        """
        Embed a user query and normalize the vector.
        """
        q_vec = self.model.encode(query, convert_to_numpy=True).reshape(1, -1)
        faiss.normalize_L2(q_vec)
        return q_vec
