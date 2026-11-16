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

def retrieval_score_margin(scores: Any) -> float:
    """
    Cheap Signal: Retrieval score margin.
    Computes difference between top-1 and top-2 retrieval scores (e.g. RRF scores).
    Accepts:
    - List of numeric scores [score1, score2, ...]
    - List of doc dicts [{"score": score1}, {"score": score2}, ...]
    Returns top-1 - top-2 score (>= 0.0). If fewer than 2 scores, returns top-1 (or 0.0 if empty).
    """
    if not scores:
        return 0.0
    extracted: List[float] = []
    for s in scores:
        if isinstance(s, dict):
            val = s.get("score", s.get("rrf_score", s.get("bm25_score", 0.0)))
            extracted.append(float(val))
        elif isinstance(s, (int, float)):
            extracted.append(float(s))
    if not extracted:
        return 0.0
    sorted_scores = sorted(extracted, reverse=True)
    if len(sorted_scores) == 1:
        return float(sorted_scores[0])
    return float(round(sorted_scores[0] - sorted_scores[1], 6))

DEFAULT_STOPWORDS = {
    "the", "a", "an", "is", "was", "are", "were", "of", "in", "to",
    "and", "that", "it", "for", "on", "with", "as", "by", "at", "from", "or"
}

def lexical_groundedness_proxy(
    claims: Any,
    context_texts: Any,
    method: str = "jaccard",
    filter_stopwords: bool = True
) -> float:
    """
    Cheap Signal: Lexical Groundedness Proxy without LLM judge calls.
    Computes token Jaccard similarity or unigram overlap between claims and context texts.
    
    Args:
        claims: A single answer/claim string or a list of claims/sentences.
        context_texts: A single text string, a list of strings, or a list of doc dicts with 'text'.
        method: 'jaccard' (default) for token Jaccard similarity (|A ∩ B| / |A ∪ B|),
                or 'overlap' / 'precision' for claim unigram containment in context (|A ∩ B| / |A|).
        filter_stopwords: Whether to remove common English stopwords so content words govern grounding.
                
    Returns:
        float score in [0.0, 1.0].
    """
    if not claims or not context_texts:
        return 0.0

    if isinstance(claims, str):
        claims_list = [claims]
    elif isinstance(claims, list):
        claims_list = [c if isinstance(c, str) else str(c) for c in claims]
    else:
        claims_list = [str(claims)]

    raw_ctx_tokens = set()
    if isinstance(context_texts, str):
        raw_ctx_tokens = set(norm(context_texts).split())
    elif isinstance(context_texts, list):
        for c in context_texts:
            if isinstance(c, dict):
                text = c.get("text", "")
            else:
                text = str(c)
            raw_ctx_tokens.update(norm(text).split())
    else:
        raw_ctx_tokens = set(norm(str(context_texts)).split())

    if not raw_ctx_tokens:
        return 0.0

    ctx_tokens_pool = raw_ctx_tokens - DEFAULT_STOPWORDS if filter_stopwords else raw_ctx_tokens
    if not ctx_tokens_pool:
        ctx_tokens_pool = raw_ctx_tokens

    scores = []
    for claim in claims_list:
        raw_claim_tokens = set(norm(claim).split())
        if not raw_claim_tokens:
            continue
        claim_tokens = raw_claim_tokens - DEFAULT_STOPWORDS if filter_stopwords else raw_claim_tokens
        if not claim_tokens:
            claim_tokens = raw_claim_tokens

        inter = claim_tokens & ctx_tokens_pool
        if method in ("overlap", "unigram_overlap", "precision"):
            s = len(inter) / len(claim_tokens)
        else:
            union = claim_tokens | ctx_tokens_pool
            s = len(inter) / len(union) if union else 0.0
        scores.append(s)

    if not scores:
        return 0.0
    return float(round(sum(scores) / len(scores), 4))



def retrieval_margin(docs: List[Dict[str, Any]]) -> float:
    """
    Cheap Signal: Top-1 vs Top-2 retrieval confidence margin (normalized).
    Measures the sharpness of the retrieval distribution without calling an LLM judge.
    Returns float in [0.0, 1.0].
    """
    if not docs or len(docs) < 2:
        return 1.0 if docs else 0.0
    s1 = float(docs[0].get("score", 0.0))
    s2 = float(docs[1].get("score", 0.0))
    if s1 <= 0.0:
        return 0.0
    margin = (s1 - s2) / s1
    return float(max(0.0, min(1.0, margin)))


