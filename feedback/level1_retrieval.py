from typing import List, Dict, Any, Tuple
from core.state import DocumentChunk
from core.llm_provider import BaseLLM, get_llm
from config.settings import settings

class DocumentGrader:
    """
    Level 1 Feedback: Document Relevance Grader.
    Acts as an automated judge evaluating whether each retrieved chunk
    contains facts or context relevant to the user query.
    Filters out noise before passing to the generator.
    """
    def __init__(self, llm: BaseLLM = None, threshold: float = None):
        self.llm = llm or get_llm()
        self.threshold = threshold or settings.relevance_threshold

    def grade(self, query: str, document: DocumentChunk) -> Tuple[bool, float, str]:
        prompt = f"""
You are a Document Relevance Grader in an Agentic RAG system.
Evaluate if the following document contains information relevant to answering the query.

Query: {query}
Document Content: {document['content']}

Respond ONLY in valid JSON format:
{{
  "relevant": true or false,
  "score": float between 0.0 and 1.0,
  "reason": "brief explanation"
}}
"""
        response = self.llm.invoke(prompt)
        res_json = response.to_json()

        score = float(res_json.get("score", 0.70))
        is_relevant = bool(res_json.get("relevant", score >= self.threshold))
        reason = str(res_json.get("reason", "Graded by Level 1 Grader"))

        # Also enforce threshold check
        passed = is_relevant and (score >= self.threshold)
        return passed, score, reason

    def grade_all(self, query: str, docs: List[DocumentChunk]) -> Tuple[List[DocumentChunk], List[Dict[str, Any]]]:
        passed_docs: List[DocumentChunk] = []
        grading_reports: List[Dict[str, Any]] = []

        for doc in docs:
            passed, score, reason = self.grade(query, doc)
            report = {
                "doc_id": doc["id"],
                "passed": passed,
                "score": score,
                "reason": reason
            }
            grading_reports.append(report)
            if passed:
                updated_doc = dict(doc)
                updated_doc["score"] = score
                passed_docs.append(updated_doc)

        return passed_docs, grading_reports


class QueryRewriter:
    """
    Level 1 Feedback: Query Rewriting Agent.
    When retrieved documents fail grading or return zero relevant hits,
    analyzes the query and generates an improved, semantically richer query.
    """
    def __init__(self, llm: BaseLLM = None):
        self.llm = llm or get_llm()

    def rewrite(self, query: str, failure_reason: str = "Insufficient relevance") -> Tuple[str, str]:
        prompt = f"""
You are a Query Rewriter in an Agentic RAG retrieval-correction loop.
The previous search query failed to retrieve relevant documents.
Failure Reason: {failure_reason}
Original Query: {query}

Rewrite this search query to optimize vector retrieval, expanding abbreviations, 
clarifying ambiguous concepts, and using domain-specific search keywords.

Respond ONLY in valid JSON format:
{{
  "rewritten_query": "the new query string",
  "reasoning": "brief explanation of modifications"
}}
"""
        response = self.llm.invoke(prompt)
        res_json = response.to_json()

        rewritten = res_json.get("rewritten_query", f"{query} detailed overview architecture")
        reasoning = res_json.get("reasoning", "Expanded query for better vector recall.")
        return rewritten, reasoning

