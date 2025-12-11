"""
generator.py
-------------
Generates the final answer using an LLM (Phi-3.5-mini-instruct).
"""

from transformers import pipeline
import torch


class AnswerGenerator:
    def __init__(self, model_name="microsoft/Phi-3.5-mini-instruct"):
        """
        Initialize the LLM with automatic device selection.
        """
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=200,
            device=self.device
        )

    def build_prompt(self, top_docs, query):
        """
        Build the RAG prompt using top reranked documents.
        """
        context = "\n\n".join(
            [f"[Doc {i+1}] {d['text']}" for i, d in enumerate(top_docs)]
        )

        prompt = f"""
Use ONLY the following evidence:

{context}

Question: {query}

Provide a clear, concise, factual answer.
"""
        return prompt

    def generate(self, prompt):
        """
        Generate the final LLM answer.
        """
        output = self.generator(prompt)[0]["generated_text"]
        return output
