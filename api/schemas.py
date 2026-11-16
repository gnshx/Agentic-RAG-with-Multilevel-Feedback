from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000, description="The user question to process")
    session_id: Optional[str] = Field(default="session_local", description="Session identifier for multi-turn tracking")
    mode: Optional[str] = Field(default="agentic_workflow", description="Execution mode: 'agentic_workflow', 'thresholds', or 'learned'")

class ExecutionStepSchema(BaseModel):
    timestamp: str
    level: str
    node: str
    action: str
    details: Dict[str, Any]

class DocumentChunkSchema(BaseModel):
    id: str
    content: str
    source: str
    score: float
    dynamic_weight: float
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    query: str
    current_query: str
    generation: str
    cached_answer: bool
    is_grounded: bool
    is_complete: bool
    groundedness_score: float
    completeness_score: float
    retrieval_retries: int
    generation_retries: int
    system_log_id: Optional[str] = None
    graded_docs: List[DocumentChunkSchema] = []
    execution_trace: List[ExecutionStepSchema] = []

class FeedbackRequest(BaseModel):
    inference_id: str = Field(..., description="The inference log ID returned from a prior query")
    rating: int = Field(..., ge=-1, le=1, description="+1 for positive helpful, -1 for negative/hallucinated, 0 for neutral")
    comment: Optional[str] = Field(default="", description="Optional human explanation or critique")
    corrected_answer: Optional[str] = Field(default=None, description="Optional ground-truth human correction")
    add_to_golden_cache: bool = Field(default=False, description="Whether to promote this answer into the Level 3 golden cache")

class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str
    reweighted_chunks: int
    cached_in_golden: bool

class DiagnosticsResponse(BaseModel):
    telemetry: Dict[str, Any]
    diagnosis: str
    recommended_sop_update: str
    confidence: float

