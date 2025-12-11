"""
loader.py
---------
Handles loading documents from PDF, TXT, CSV, and JSONL formats.
Used as the ingestion layer of the RAG pipeline.
"""

from pathlib import Path
import json
import pandas as pd
from pypdf import PdfReader


def load_documents(file_path: str):
    """
    Load text from supported file types and return a list of documents.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.suffix.lower()

    if ext == ".txt":
        return [file_path.read_text(encoding="utf-8")]

    elif ext == ".csv":
        df = pd.read_csv(file_path)
        return df.iloc[:, 0].astype(str).tolist()

    elif ext == ".jsonl":
        docs = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                docs.append(json.loads(line).get("text", ""))
        return docs

    elif ext == ".pdf":
        reader = PdfReader(str(file_path))
        pages = [page.extract_text() for page in reader.pages]
        return [p for p in pages if p]  # remove empty pages

    else:
        raise ValueError(f"Unsupported file format: {ext}")
