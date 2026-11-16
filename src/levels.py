import time
from typing import Dict, Any, List
from src.llm import LLM
from src.retriever import HybridRetriever
from src.planner import QueryPlanner
from src.generator import Generator
from src.signals import consistency, retrieval_grade, groundedness, completeness

def run_level_0(state: Dict[str, Any], llm: LLM, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Level 0: Direct LLM answering (Zero Retrieval).
    Samples N=3 answers at temperature 0.7, computes S1 (self-consistency).
    """
    q = state["question"]
    n_samples = cfg.get("levels", {}).get("L0", {}).get("samples", 3)
    samples: List[str] = []

    prompt = (
        f"Question: {q}\n"
        "Answer the question directly without any external retrieval. Be concise and precise."
    )

    for i in range(n_samples):
        ans = llm.call(prompt, temperature=0.7, sample_id=i).strip()
        samples.append(ans)

    s1 = consistency(samples)
    state["drafts"] = samples
    state["answer"] = samples[0] if samples else ""
    state["signals"]["S1"] = s1
    state["citations"] = []
    return state

def run_level_1(state: Dict[str, Any], llm: LLM, retriever: HybridRetriever, generator: Generator, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Level 1: Single-Pass Hybrid RAG.
    Retrieves top-k documents (BM25 + Dense RRF fusion), generates answer with citations,
    and computes S2 (retrieval quality) and S3 (groundedness).
    """
    q = state["question"]
    k = cfg.get("retrieval", {}).get("k", 5)

    docs = retriever.search(q, k=k)
    state["docs"] = docs

    gen_result = generator.answer(q, docs)
    state["answer"] = gen_result["answer"]
    state["citations"] = gen_result["citations"]

    s2 = retrieval_grade(llm, q, docs)
    s3 = groundedness(llm, state["answer"], docs)

    state["signals"]["S2"] = s2
    state["signals"]["S3"] = s3
    return state

def run_level_2(state: Dict[str, Any], llm: LLM, retriever: HybridRetriever, planner: QueryPlanner, generator: Generator, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Level 2: Query Decomposition, Multi-Hop Iterative Retrieval, and Reranking.
    Breaks question into sub-questions, retrieves per sub-question, reranks pooled
    evidence, and verifies S3 (groundedness) and S4 (completeness).
    """
    q = state["question"]
    k = cfg.get("retrieval", {}).get("k", 5)

    # Decompose
    subquestions = planner.decompose(q)
    state["subquestions"] = subquestions

    pooled_docs: Dict[str, Dict[str, Any]] = {}
    # Retain existing L1 docs if present
    for d in state.get("docs", []):
        pooled_docs[d["id"]] = d

    # Retrieve per sub-question ensuring multi-hop representation
    sub_hop_docs: List[Dict[str, Any]] = []
    for sub_q in subquestions:
        sub_docs = retriever.search(sub_q, k=max(2, k // max(1, len(subquestions))))
        for d in sub_docs:
            if d["id"] not in pooled_docs:
                pooled_docs[d["id"]] = d
                sub_hop_docs.append(d)

    # Combine multi-hop sub-question evidence with primary pool
    combined_docs = sub_hop_docs + [d for d in pooled_docs.values() if d not in sub_hop_docs]
    # Keep top k with multi-hop coverage guaranteed
    selected_docs = combined_docs[:k]
    state["docs"] = selected_docs

    # Generate answer with citations
    gen_result = generator.answer(q, selected_docs)
    state["answer"] = gen_result["answer"]
    state["citations"] = gen_result["citations"]

    s3 = groundedness(llm, state["answer"], selected_docs)
    s4 = completeness(llm, q, state["answer"])

    state["signals"]["S3"] = s3
    state["signals"]["S4"] = s4
    return state

def run_level_3(state: Dict[str, Any], llm: LLM, retriever: HybridRetriever, generator: Generator, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Level 3: Full Agentic RAG.
    Broadened retrieval (k_broad), N=3 parallel drafts with iterative critic-revision
    loops, and majority voting across revised candidates.
    """
    q = state["question"]
    k_broad = cfg.get("retrieval", {}).get("k_broad", 10)
    l3_cfg = cfg.get("levels", {}).get("L3", {})
    n_drafts = l3_cfg.get("drafts", 3)
    max_revisions = l3_cfg.get("max_revisions", 2)

    # Broaden retrieval
    broad_docs = retriever.search(q, k=k_broad)
    pooled = {d["id"]: d for d in state.get("docs", [])}
    for d in broad_docs:
        pooled[d["id"]] = d
    state["docs"] = list(pooled.values())[:k_broad]

    # Agentic multi-draft, critic, revise, majority vote
    result = generator.answer_level_3(
        question=q,
        docs=state["docs"],
        n_drafts=n_drafts,
        max_revisions=max_revisions
    )

    state["answer"] = result["answer"]
    state["citations"] = result["citations"]
    state["drafts"] = result.get("all_drafts", [])

    # Evaluate full diagnostic signals
    state["signals"]["S2"] = retrieval_grade(llm, q, state["docs"])
    state["signals"]["S3"] = groundedness(llm, state["answer"], state["docs"])
    state["signals"]["S4"] = completeness(llm, q, state["answer"])
    return state

