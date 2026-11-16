from datetime import datetime, timezone
from typing import Dict, Any, List
from core.state import AgenticRAGState, ExecutionStep, DocumentChunk
from feedback.level1_retrieval import DocumentGrader, QueryRewriter
from feedback.level2_generation import GeneratorAgent, HallucinationAuditor, CompletenessAuditor
from feedback.level3_system_memory import FeedbackRegistry, GoldenCache
from retrieval.vector_store import LocalVectorStore
from config.settings import settings

def make_trace_step(level: str, node: str, action: str, details: Dict[str, Any]) -> ExecutionStep:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "node": node,
        "action": action,
        "details": details
    }

class GraphNodes:
    """Encapsulates all node functions with shared stateful components."""
    def __init__(
        self,
        vector_store: LocalVectorStore,
        registry: FeedbackRegistry,
        golden_cache: GoldenCache,
        grader: DocumentGrader,
        rewriter: QueryRewriter,
        generator: GeneratorAgent,
        hallucination_auditor: HallucinationAuditor,
        completeness_auditor: CompletenessAuditor
    ):
        self.vector_store = vector_store
        self.registry = registry
        self.golden_cache = golden_cache
        self.grader = grader
        self.rewriter = rewriter
        self.generator = generator
        self.hallucination_auditor = hallucination_auditor
        self.completeness_auditor = completeness_auditor

    def check_golden_cache_node(self, state: AgenticRAGState) -> Dict[str, Any]:
        query = state.get("query", "")
        trace = list(state.get("execution_trace", []))

        if settings.enable_golden_cache:
            hit = self.golden_cache.lookup(query)
            if hit:
                gid, cached_answer, sim = hit
                trace.append(make_trace_step(
                    level="Level 3: System Memory",
                    node="check_golden_cache",
                    action="Golden Cache Hit",
                    details={"cached_id": gid, "similarity": sim}
                ))
                return {
                    "cached_answer": True,
                    "golden_cache_hit_id": gid,
                    "generation": cached_answer,
                    "is_grounded": True,
                    "is_complete": True,
                    "groundedness_score": 1.0,
                    "completeness_score": 1.0,
                    "execution_trace": trace
                }

        trace.append(make_trace_step(
            level="Level 3: System Memory",
            node="check_golden_cache",
            action="Golden Cache Miss",
            details={"status": "Query routed to retrieval pipeline"}
        ))
        return {
            "cached_answer": False,
            "execution_trace": trace
        }

    def retrieve_node(self, state: AgenticRAGState) -> Dict[str, Any]:
        query = state.get("current_query") or state.get("query", "")
        docs = self.vector_store.search(query, top_k=settings.top_k_documents)
        trace = list(state.get("execution_trace", []))

        trace.append(make_trace_step(
            level="Level 1: Retrieval",
            node="retrieve",
            action=f"Retrieved {len(docs)} candidate chunks",
            details={
                "search_query": query,
                "candidate_ids": [d["id"] for d in docs],
                "top_score": docs[0]["score"] if docs else 0.0
            }
        ))
        return {
            "retrieved_docs": docs,
            "execution_trace": trace
        }

    def grade_documents_node(self, state: AgenticRAGState) -> Dict[str, Any]:
        query = state.get("current_query") or state.get("query", "")
        docs = state.get("retrieved_docs", [])
        passed_docs, reports = self.grader.grade_all(query, docs)
        trace = list(state.get("execution_trace", []))

        trace.append(make_trace_step(
            level="Level 1: Retrieval",
            node="grade_documents",
            action=f"Filtered {len(passed_docs)}/{len(docs)} documents as relevant",
            details={"grading_reports": reports}
        ))
        return {
            "graded_docs": passed_docs,
            "execution_trace": trace
        }

    def rewrite_query_node(self, state: AgenticRAGState) -> Dict[str, Any]:
        prev_query = state.get("current_query") or state.get("query", "")
        retries = state.get("retrieval_retries", 0) + 1
        new_query, reasoning = self.rewriter.rewrite(
            prev_query,
            failure_reason="No documents met the relevance confidence threshold."
        )
        trace = list(state.get("execution_trace", []))

        trace.append(make_trace_step(
            level="Level 1: Retrieval",
            node="rewrite_query",
            action=f"Query reformulated (Attempt {retries}/{settings.max_retrieval_retries})",
            details={"previous_query": prev_query, "new_query": new_query, "reasoning": reasoning}
        ))
        return {
            "current_query": new_query,
            "retrieval_retries": retries,
            "execution_trace": trace
        }

    def generate_node(self, state: AgenticRAGState) -> Dict[str, Any]:
        query = state.get("query", "")
        docs = state.get("graded_docs") or state.get("retrieved_docs", [])
        critique = state.get("critique")
        answer = self.generator.generate(query, docs, critique=critique)
        trace = list(state.get("execution_trace", []))

        trace.append(make_trace_step(
            level="Level 2: Generation",
            node="generate",
            action="Synthesized candidate response",
            details={"char_length": len(answer), "context_chunk_count": len(docs)}
        ))
        return {
            "generation": answer,
            "execution_trace": trace
        }

    def audit_generation_node(self, state: AgenticRAGState) -> Dict[str, Any]:
        query = state.get("query", "")
        docs = state.get("graded_docs") or state.get("retrieved_docs", [])
        generation = state.get("generation", "")

        is_grounded, grounded_score, ungrounded, ground_critique = self.hallucination_auditor.audit(docs, generation)
        is_complete, complete_score, missing, complete_critique = self.completeness_auditor.audit(query, generation)

        combined_critique = []
        if not is_grounded:
            combined_critique.append(f"Groundedness Failure: {ground_critique}")
        if not is_complete:
            combined_critique.append(f"Completeness Failure: {complete_critique}")

        critique_str = "\n".join(combined_critique) if combined_critique else "Audit passed."
        trace = list(state.get("execution_trace", []))

        trace.append(make_trace_step(
            level="Level 2: Generation",
            node="audit_generation",
            action="Audited faithfulness and completeness",
            details={
                "groundedness_score": grounded_score,
                "is_grounded": is_grounded,
                "completeness_score": complete_score,
                "is_complete": is_complete,
                "critique": critique_str
            }
        ))
        return {
            "is_grounded": is_grounded,
            "groundedness_score": grounded_score,
            "is_complete": is_complete,
            "completeness_score": complete_score,
            "critique": critique_str,
            "execution_trace": trace
        }

    def revise_generation_node(self, state: AgenticRAGState) -> Dict[str, Any]:
        retries = state.get("generation_retries", 0) + 1
        query = state.get("query", "")
        docs = state.get("graded_docs") or state.get("retrieved_docs", [])
        critique = state.get("critique", "")

        revised_answer = self.generator.generate(query, docs, critique=critique)
        trace = list(state.get("execution_trace", []))

        trace.append(make_trace_step(
            level="Level 2: Generation",
            node="revise_generation",
            action=f"Revised response based on audit critique (Revision {retries}/{settings.max_generation_retries})",
            details={"critique_applied": critique}
        ))
        return {
            "generation": revised_answer,
            "generation_retries": retries,
            "execution_trace": trace
        }

    def log_system_feedback_node(self, state: AgenticRAGState) -> Dict[str, Any]:
        trace = list(state.get("execution_trace", []))
        log_id = self.registry.log_inference(state)

        trace.append(make_trace_step(
            level="Level 3: System Memory",
            node="log_system_feedback",
            action="Logged inference execution trail to SQLite registry",
            details={"log_id": log_id}
        ))
        return {
            "system_log_id": log_id,
            "execution_trace": trace
        }

