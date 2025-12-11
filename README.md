# End-to-End-RAG-System

<img width="1051" height="509" alt="image" src="https://github.com/user-attachments/assets/d87aeefc-33ef-4c6e-9431-f6c6a860ab34" />
---

## Pipeline Architecture

**Document Loading**  
Supports PDF, TXT, CSV, JSONL formats.

**Embedding Layer**  
Converts all documents into dense vectors using `sentence-transformers/all-MiniLM-L6-v2`.

**Vector Index (FAISS)**  
Builds a cosine-similarity index for fast retrieval.

**Semantic Search**  
Retrieves top-k relevant chunks based on vector similarity.

**Cross-Encoder Reranking**  
Reranks retrieved results using a more precise model: `cross-encoder/ms-marco-MiniLM-L-6-v2`.

**LLM Answer Generation**  
Generates final answers using `microsoft/Phi-3.5-mini-instruct`.

---
## Project Structure
```
rag-engine-core/
│
├── data/
│   └── Big_Data.pdf              #  corpus / documents
│
├── notebooks/
│   └── semantic_rag_pipeline.ipynb        #  full notebook
│
├── src/
│   ├── loader.py                 # handle PDFs, CSV, JSONL
│   ├── embedder.py               # embedding functions
│   ├── retriever.py              # FAISS + reranking
│   └── generator.py              # LLM answer generation
│
├── requirements.txt              # dependencies
│
├── README.md                     # full project description
```

---

## Install dependencies:

```bash
pip install -r requirements.txt
