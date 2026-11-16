from typing import List, Dict, Any, Tuple, Optional
from core.state import DocumentChunk
from core.llm_provider import BaseLLM, get_llm
from config.settings import settings

class GeneratorAgent:
    """
    Synthesizes responses strictly grounded on graded context chunks.
    Incorporates revision critiques when running in Level 2 feedback loops.
    """
    def __init__(self, llm: BaseLLM = None):
        self.llm = llm or get_llm()

    def generate(self, query: str, context_docs: List[DocumentChunk], critique: Optional[str] = None) -> str:
        formatted_context = "\n---\n".join([
            f"[Source: {d.get('source', 'unknown')} | ID: {d.get('id', '')}]\n{d.get('content', '')}"
            for d in context_docs
        ])

        system_prompt = (
            "You are an expert technical AI assistant operating within an Agentic RAG system. "
            "Your objective is to provide precise, rigorous answers strictly grounded in the provided context. "
            "Never fabricate facts. If information is absent, explicitly acknowledge the limitation."
        )

        critique_directive = ""
        if critique:
            critique_directive = f"\n\nCRITICAL AUDIT FEEDBACK (Must address in this revision):\n{critique}\n"

        prompt = f"""
{system_prompt}

Context:
{formatted_context}

Question: {query}
{critique_directive}

Provide a comprehensive, faithful answer:
"""
        response = self.llm.invoke(prompt, system_prompt=system_prompt)
        return response.content


class HallucinationAuditor:
    """
    Level 2 Feedback: Groundedness / Faithfulness Auditor.
    Verifies that the generated response contains only facts supported
    by the provided context documents. Identifies unsupported claims.
    """
    def __init__(self, llm: BaseLLM = None, threshold: float = None):
        self.llm = llm or get_llm()
        self.threshold = threshold or settings.groundedness_threshold

    def audit(self, context_docs: List[DocumentChunk], generation: str) -> Tuple[bool, float, List[str], str]:
        context_text = "\n".join([d.get("content", "") for d in context_docs])
        prompt = f"""
You are a Faithfulness & Hallucination Auditor in an Agentic RAG system.
Audit the following generated answer against the source context documents.

Source Context:
{context_text}

Generated Answer:
{generation}

Determine if every claim in the answer is directly supported by the context.
Respond ONLY in valid JSON format:
{{
  "grounded": true or false,
  "groundedness_score": float between 0.0 and 1.0,
  "unsupported_claims": ["list of any unsupported statements"],
  "critique": "actionable instructions for the generator to fix hallucinations"
}}
"""
        response = self.llm.invoke(prompt)
        res_json = response.to_json()

        score = float(res_json.get("groundedness_score", 0.85))
        is_grounded = bool(res_json.get("grounded", score >= self.threshold))
        unsupported = list(res_json.get("unsupported_claims", []))
        critique = str(res_json.get("critique", "No hallucination detected."))

        passed = is_grounded and (score >= self.threshold) and (len(unsupported) == 0)
        return passed, score, unsupported, critique


class CompletenessAuditor:
    """
    Level 2 Feedback: Answer Completeness & Intent Auditor.
    Determines whether the generated answer thoroughly addresses the original user query.
    """
    def __init__(self, llm: BaseLLM = None, threshold: float = None):
        self.llm = llm or get_llm()
        self.threshold = threshold or settings.completeness_threshold

    def audit(self, query: str, generation: str) -> Tuple[bool, float, List[str], str]:
        prompt = f"""
You are an Answer Completeness Auditor in an Agentic RAG system.
Evaluate whether the generated answer completely and directly answers the user's question.

Question: {query}
Generated Answer: {generation}

Respond ONLY in valid JSON format:
{{
  "complete": true or false,
  "completeness_score": float between 0.0 and 1.0,
  "missing_aspects": ["list of unaddressed aspects of the query"],
  "critique": "actionable feedback on how to make the answer complete"
}}
"""
        response = self.llm.invoke(prompt)
        res_json = response.to_json()

        score = float(res_json.get("completeness_score", 0.80))
        is_complete = bool(res_json.get("complete", score >= self.threshold))
        missing = list(res_json.get("missing_aspects", []))
        critique = str(res_json.get("critique", "Answer is complete."))

        passed = is_complete and (score >= self.threshold) and (len(missing) == 0)
        return passed, score, missing, critique

