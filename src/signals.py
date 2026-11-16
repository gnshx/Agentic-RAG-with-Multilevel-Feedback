import json
import re
from collections import Counter
from typing import List, Dict, Any
from src.llm import LLM

def _json(text: str, default: Any) -> Any:
    """Extract JSON object or array from LLM response with strict fallback."""
    if not text:
        return default
    m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else default
    except Exception:
        return default

def norm(text: str) -> str:
    """Normalize text by lowercasing, removing punctuation and leading articles."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())

def consistency(samples: List[str]) -> float:
    """
    Signal S1: Self-consistency agreement across N sampled direct answers.
    Returns the fraction of normalized answers that agree with the mode. Range [0, 1].
    """
    if not samples:
        return 0.0
    normed = [norm(s) for s in samples if norm(s)]
    if not normed:
        return 0.0
    mode_count = Counter(normed).most_common(1)[0][1]
    return float(mode_count / len(samples))

def retrieval_grade(llm: LLM, q: str, docs: List[Dict[str, Any]]) -> float:
    """
    Signal S2: Retrieval Quality Grader (CRAG-style).
    Grades each retrieved passage chunk as relevant (True) or irrelevant (False).
    Returns fraction of relevant chunks in [0, 1]. Fallback 0.0 forces escalation.
    """
    if not docs:
        return 0.0
    passages_str = "\n".join(f"[{i}] {d.get('text', '')}" for i, d in enumerate(docs))
    prompt = (
        f"Question: {q}\n"
        "For each passage say if it helps answer the question.\n"
        "Return JSON only: {{\"relevant\": [true, false, ...]}}\n"
        f"{passages_str}"
    )
    raw = llm.call(prompt, temperature=0.0)
    parsed = _json(raw, {"relevant": []})
    relevant_list = parsed.get("relevant", [])
    if not isinstance(relevant_list, list) or not relevant_list:
        return 0.0
    return float(sum(bool(x) for x in relevant_list) / len(docs))

def groundedness(llm: LLM, answer: str, docs: List[Dict[str, Any]]) -> float:
    """
    Signal S3: Faithfulness & Groundedness Checker.
    Decomposes the answer into atomic factual claims and checks whether each claim
    is corroborated by the retrieved evidence.
    Returns supported_claims / total_claims in [0, 1]. Fallback 0.0 forces escalation.
    """
    if not answer or not docs:
        return 0.0
    ev = "\n".join(d.get("text", "") for d in docs)
    prompt = (
        "Split the answer into factual claims. For each, say whether the evidence supports it.\n"
        "Return JSON only: {{\"supported\": [true, false, ...]}}\n"
        f"Answer: {answer}\n"
        f"Evidence:\n{ev}"
    )
    raw = llm.call(prompt, temperature=0.0)
    parsed = _json(raw, {"supported": []})
    supported_list = parsed.get("supported", [])
    if not isinstance(supported_list, list) or not supported_list:
        return 0.0
    return float(sum(bool(x) for x in supported_list) / len(supported_list))

def completeness(llm: LLM, q: str, answer: str) -> float:
    """
    Signal S4: Answer Completeness Checker.
    Identifies the sub-requirements of the question and marks which ones were satisfied.
    Returns answered_parts / total_required in [0, 1]. Fallback 0.0 forces escalation.
    """
    if not q or not answer:
        return 0.0
    prompt = (
        "List the parts the question requires, and for each say whether the answer addresses it.\n"
        "Return JSON only: {{\"answered\": [true, false, ...]}}\n"
        f"Question: {q}\n"
        f"Answer: {answer}"
    )
    raw = llm.call(prompt, temperature=0.0)
    parsed = _json(raw, {"answered": []})
    answered_list = parsed.get("answered", [])
    if not isinstance(answered_list, list) or not answered_list:
        return 0.0
    return float(sum(bool(x) for x in answered_list) / len(answered_list))

