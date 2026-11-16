from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from core.state import AgenticRAGState
from graph.nodes import GraphNodes
from config.settings import settings
from retrieval.vector_store import LocalVectorStore
from feedback.level1_retrieval import DocumentGrader, QueryRewriter
from feedback.level2_generation import GeneratorAgent, HallucinationAuditor, CompletenessAuditor
from feedback.level3_system_memory import FeedbackRegistry, GoldenCache, SOPArchitect
from core.llm_provider import get_llm

def decide_after_golden_cache(state: AgenticRAGState) -> str:
    if state.get("cached_answer"):
        return END
    return "retrieve"

def decide_after_grading(state: AgenticRAGState) -> str:
    graded_docs = state.get("graded_docs", [])
    retries = state.get("retrieval_retries", 0)

    if graded_docs and len(graded_docs) > 0:
        return "generate"

    if retries < settings.max_retrieval_retries:
        return "rewrite_query"

    # Retries reached limit; fallback to generation with unverified or empty docs
    return "generate"

def decide_after_audit(state: AgenticRAGState) -> str:
    is_grounded = state.get("is_grounded", False)
    is_complete = state.get("is_complete", False)
    retries = state.get("generation_retries", 0)

    if is_grounded and is_complete:
        return "log_system_feedback"

    if retries < settings.max_generation_retries:
        return "revise_generation"

    # Max retries reached; log and conclude
    return "log_system_feedback"

class AgenticRAGWorkflow:
    """
    Main State Machine orchestrating Level 1, Level 2, and Level 3
    feedback loops into a resilient Agentic RAG graph.
    """
    def __init__(
        self,
        vector_store: Optional[LocalVectorStore] = None,
        registry: Optional[FeedbackRegistry] = None,
        golden_cache: Optional[GoldenCache] = None
    ):
        self.vector_store = vector_store or LocalVectorStore(db_path=settings.database_path)
        self.registry = registry or FeedbackRegistry()
        self.golden_cache = golden_cache or GoldenCache(self.registry)

        llm = get_llm()
        self.grader = DocumentGrader(llm=llm)
        self.rewriter = QueryRewriter(llm=llm)
        self.generator = GeneratorAgent(llm=llm)
        self.hallucination_auditor = HallucinationAuditor(llm=llm)
        self.completeness_auditor = CompletenessAuditor(llm=llm)
        self.sop_architect = SOPArchitect(self.registry, llm=llm)

        self.nodes = GraphNodes(
            vector_store=self.vector_store,
            registry=self.registry,
            golden_cache=self.golden_cache,
            grader=self.grader,
            rewriter=self.rewriter,
            generator=self.generator,
            hallucination_auditor=self.hallucination_auditor,
            completeness_auditor=self.completeness_auditor
        )

        self.app = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgenticRAGState)

        # Register all nodes
        graph.add_node("check_golden_cache", self.nodes.check_golden_cache_node)
        graph.add_node("retrieve", self.nodes.retrieve_node)
        graph.add_node("grade_documents", self.nodes.grade_documents_node)
        graph.add_node("rewrite_query", self.nodes.rewrite_query_node)
        graph.add_node("generate", self.nodes.generate_node)
        graph.add_node("audit_generation", self.nodes.audit_generation_node)
        graph.add_node("revise_generation", self.nodes.revise_generation_node)
        graph.add_node("log_system_feedback", self.nodes.log_system_feedback_node)

        # Entry point: Level 3 Golden Memory Cache check
        graph.set_entry_point("check_golden_cache")

        # Routing from Golden Cache
        graph.add_conditional_edges("check_golden_cache", decide_after_golden_cache, {
            END: END,
            "retrieve": "retrieve"
        })

        # Level 1 Feedback Loop: Retrieval -> Grade -> [Rewrite or Generate]
        graph.add_edge("retrieve", "grade_documents")
        graph.add_conditional_edges("grade_documents", decide_after_grading, {
            "generate": "generate",
            "rewrite_query": "rewrite_query"
        })
        graph.add_edge("rewrite_query", "retrieve")

        # Level 2 Feedback Loop: Generate -> Audit -> [Revise or Log]
        graph.add_edge("generate", "audit_generation")
        graph.add_conditional_edges("audit_generation", decide_after_audit, {
            "revise_generation": "revise_generation",
            "log_system_feedback": "log_system_feedback"
        })
        graph.add_edge("revise_generation", "audit_generation")

        # Final exit from Level 3 System Logging
        graph.add_edge("log_system_feedback", END)

        return graph.compile()

    def run(self, query: str, session_id: str = "default_session") -> AgenticRAGState:
        initial_state: AgenticRAGState = {
            "query": query,
            "current_query": query,
            "session_id": session_id,
            "retrieved_docs": [],
            "graded_docs": [],
            "retrieval_retries": 0,
            "retrieval_fallback": False,
            "generation": "",
            "generation_retries": 0,
            "critique": "",
            "groundedness_score": 0.0,
            "completeness_score": 0.0,
            "is_grounded": False,
            "is_complete": False,
            "cached_answer": False,
            "golden_cache_hit_id": None,
            "system_log_id": None,
            "execution_trace": []
        }
        final_state = self.app.invoke(initial_state)
        return final_state

