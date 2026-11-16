import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from config.settings import settings
from retrieval.document_loader import DocumentLoader
from retrieval.vector_store import LocalVectorStore
from feedback.level3_system_memory import FeedbackRegistry, GoldenCache, SOPArchitect
from graph.workflow import AgenticRAGWorkflow
from api.schemas import (
    QueryRequest, QueryResponse, FeedbackRequest, FeedbackResponse, DiagnosticsResponse
)

# Shared global state
vector_store = LocalVectorStore()
registry = FeedbackRegistry()
golden_cache = GoldenCache(registry)
workflow: AgenticRAGWorkflow = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load sample docs and index them
    global workflow
    loader = DocumentLoader()
    sample_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs")
    chunks = loader.load_directory(sample_dir)
    if chunks:
        vector_store.add_documents(chunks)
        print(f"[Startup] Indexed {len(chunks)} document chunks into local vector store.")
    else:
        print("[Startup] No sample documents found; initialized empty vector store.")

    workflow = AgenticRAGWorkflow(
        vector_store=vector_store,
        registry=registry,
        golden_cache=golden_cache
    )
    yield
    print("[Shutdown] Gracefully closed Agentic RAG services.")

app = FastAPI(
    title="Agentic RAG with Multilevel Feedback System",
    description="Local enterprise-grade multi-agent RAG featuring query grading, hallucination auditing, and self-improving memory loops.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "llm_provider": settings.llm_provider,
        "indexed_chunks": len(vector_store.documents)
    }

@app.post("/api/query", response_model=QueryResponse)
def execute_query(req: QueryRequest):
    if not workflow:
        raise HTTPException(status_code=500, detail="Workflow engine not initialized.")

    final_state = workflow.run(query=req.query, session_id=req.session_id or "default")

    return QueryResponse(
        query=final_state.get("query", ""),
        current_query=final_state.get("current_query", ""),
        generation=final_state.get("generation", ""),
        cached_answer=final_state.get("cached_answer", False),
        is_grounded=final_state.get("is_grounded", False),
        is_complete=final_state.get("is_complete", False),
        groundedness_score=final_state.get("groundedness_score", 0.0),
        completeness_score=final_state.get("completeness_score", 0.0),
        retrieval_retries=final_state.get("retrieval_retries", 0),
        generation_retries=final_state.get("generation_retries", 0),
        system_log_id=final_state.get("system_log_id"),
        graded_docs=final_state.get("graded_docs", []),
        execution_trace=final_state.get("execution_trace", [])
    )

@app.post("/api/feedback", response_model=FeedbackResponse)
def submit_feedback(req: FeedbackRequest):
    # Retrieve the logged inference from Level 3 registry
    inference = registry.get_inference(req.inference_id)
    if not inference:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inference record '{req.inference_id}' not found."
        )

    # Log human review in SQLite
    fid = registry.log_human_feedback(
        inference_id=req.inference_id,
        rating=req.rating,
        comment=req.comment or "",
        corrected_answer=req.corrected_answer
    )

    # Dynamic Weight Adjustment across contributing document chunks
    delta = settings.dynamic_weight_increment * req.rating
    doc_ids = inference.get("doc_ids", [])
    reweighted_count = 0
    for doc_id in doc_ids:
        if vector_store.update_weight(doc_id, delta):
            reweighted_count += 1

    # Optional promotion to Golden Exemplar Cache
    cached_in_golden = False
    if req.add_to_golden_cache or (req.rating > 0 and req.corrected_answer):
        answer_to_cache = req.corrected_answer if req.corrected_answer else inference.get("generation", "")
        golden_cache.store_exemplar(
            query=inference.get("query", ""),
            answer=answer_to_cache,
            verified_by="human_feedback_promotion"
        )
        cached_in_golden = True

    return FeedbackResponse(
        status="success",
        feedback_id=fid,
        reweighted_chunks=reweighted_count,
        cached_in_golden=cached_in_golden
    )

@app.get("/api/history")
def get_history(limit: int = 10):
    return registry.get_recent_inferences(limit=min(limit, 50))

@app.get("/api/diagnostics", response_model=DiagnosticsResponse)
def get_system_diagnostics():
    sop_architect = SOPArchitect(registry)
    result = sop_architect.analyze_system_performance()
    if result.get("status") == "insufficient_data":
        return DiagnosticsResponse(
            telemetry={"total_inferences": 0},
            diagnosis="System is warm and awaiting initial queries.",
            recommended_sop_update="Run baseline evaluation queries.",
            confidence=1.0
        )
    return DiagnosticsResponse(
        telemetry=result["telemetry"],
        diagnosis=result["diagnosis"],
        recommended_sop_update=result["recommended_sop_update"],
        confidence=result["confidence"]
    )

@app.get("/api/documents")
def list_indexed_documents():
    return [
        {
            "id": doc["id"],
            "source": doc["source"],
            "dynamic_weight": doc.get("dynamic_weight", 0.0),
            "preview": doc["content"][:120] + "..."
        }
        for doc in vector_store.documents.values()
    ]

