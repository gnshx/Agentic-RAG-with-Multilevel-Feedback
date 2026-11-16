from typing import Dict, Any, List
from src.llm import LLM
from src.retriever import HybridRetriever
from src.generator import Generator
from src.signals import retrieval_grade

class CRAGBaseline:
    """
    Standardized Reimplementation of Corrective RAG (CRAG) (Yan et al., 2024).
    Uses a Retrieval Evaluator to classify retrieval confidence into:
    - Correct: Proceed with graded context.
    - Ambiguous: Combine retrieved documents with rewritten query search.
    - Incorrect: Discard retrieved docs and trigger corrective fallback search.
    """
    def __init__(self, llm: LLM, retriever: HybridRetriever, generator: Generator, k: int = 5):
        self.llm = llm
        self.retriever = retriever
        self.generator = generator
        self.k = k

    def run(self, question: str) -> Dict[str, Any]:
        calls_before = self.llm.calls

        # Step 1: Initial Retrieval
        docs = self.retriever.search(question, k=self.k)

        # Step 2: Retrieval Evaluator
        grade = retrieval_grade(self.llm, question, docs)

        action = "correct"
        active_docs = docs
        if grade >= 0.70:
            action = "correct"
            active_docs = docs
        elif grade >= 0.30:
            action = "ambiguous"
            # Corrective expansion query
            expanded_query = f"{question} background facts and related entities"
            fallback_docs = self.retriever.search(expanded_query, k=self.k)
            # Combine
            pooled = {d["id"]: d for d in docs}
            for d in fallback_docs:
                pooled[d["id"]] = d
            active_docs = list(pooled.values())[:self.k]
        else:
            action = "incorrect"
            # Corrective fallback: broad search across corpus
            broad_query = f"{question} core domain facts"
            active_docs = self.retriever.search(broad_query, k=self.k * 2)

        # Step 3: Generation
        gen_res = self.generator.answer(question, active_docs)
        calls_used = self.llm.calls - calls_before

        return {
            "question": question,
            "answer": gen_res["answer"],
            "citations": gen_res["citations"],
            "evaluator_grade": grade,
            "action": action,
            "calls": calls_used,
            "docs_used": len(active_docs)
        }
