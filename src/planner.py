import re
from typing import List
from src.llm import LLM
from src.signals import _json

class QueryPlanner:
    """
    Decomposes multi-hop questions into discrete sub-questions and reformulates
    search queries to maximize retrieval recall on complex benchmarks.
    """
    def __init__(self, llm: LLM):
        self.llm = llm

    def decompose(self, question: str) -> List[str]:
        prompt = (
            f"Question: {question}\n"
            "Break this question into at most 3 simple sub-questions that must be answered in order.\n"
            "Return JSON only: {\"subquestions\": [\"sub-question 1\", \"sub-question 2\"]}"
        )
        raw = self.llm.call(prompt, temperature=0.0)
        parsed = _json(raw, {"subquestions": []})
        subs = parsed.get("subquestions", [])
        if not subs or not isinstance(subs, list):
            # Fallback: simple heuristic decomposition
            return [question]
        return [str(s).strip() for s in subs if str(s).strip()]

    def rewrite(self, question: str) -> str:
        prompt = (
            f"Original Question: {question}\n"
            "Rewrite this question to make it self-contained and optimal for keyword and vector search.\n"
            "Return JSON only: {\"rewritten\": \"expanded query string\"}"
        )
        raw = self.llm.call(prompt, temperature=0.0)
        parsed = _json(raw, {"rewritten": question})
        return str(parsed.get("rewritten", question)).strip()

