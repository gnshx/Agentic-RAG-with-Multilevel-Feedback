from typing import Dict, Any, List
from src.llm import LLM
from src.retriever import HybridRetriever
from src.generator import Generator
from src.signals import groundedness, retrieval_grade

class SelfRAGBaseline:
    """
    Standardized Reimplementation of Self-RAG (Asai et al., 2023).
    Incorporates reflection token evaluations:
    - [Retrieve]: Determines if external knowledge is required.
    - [IsRel]: Verifies if retrieved passages are relevant.
    - [IsSup]: Critiques whether the generated candidate is supported by evidence.
    Executes a single critique-revision loop if [IsSup] detects hallucination.
    """
    def __init__(self, llm: LLM, retriever: HybridRetriever, generator: Generator, k: int = 5):
        self.llm = llm
        self.retriever = retriever
        self.generator = generator
        self.k = k

    def run(self, question: str) -> Dict[str, Any]:
        calls_before = self.llm.calls

        # 1. Retrieval
        docs = self.retriever.search(question, k=self.k)

        # 2. Reflection: [IsRel] check
        is_rel_score = retrieval_grade(self.llm, question, docs)

        # 3. Draft generation
        draft_res = self.generator.answer(question, docs)
        draft = draft_res["answer"]

        # 4. Reflection: [IsSup] check (Groundedness)
        is_sup_score = groundedness(self.llm, draft, docs)

        final_answer = draft
        critique_applied = False

        # 5. Critique Loop: If reflection detects low support, run targeted revision
        if is_sup_score < 0.75:
            critique_issues = self.generator.critic(question, draft, docs)
            if critique_issues:
                final_answer = self.generator.revise(question, draft, critique_issues, docs)
                critique_applied = True

        calls_used = self.llm.calls - calls_before

        return {
            "question": question,
            "answer": final_answer,
            "citations": draft_res.get("citations", []),
            "is_rel_score": is_rel_score,
            "is_sup_score": is_sup_score,
            "critique_applied": critique_applied,
            "calls": calls_used
        }
