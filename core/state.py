from typing import TypedDict, List, Dict, Any, Optional

class DocumentChunk(TypedDict):
    id: str
    content: str
    source: str
    score: float
    dynamic_weight: float
    metadata: Dict[str, Any]

class ExecutionStep(TypedDict):
    timestamp: str
    level: str  # "Level 1: Retrieval", "Level 2: Generation", "Level 3: System Memory"
    node: str
    action: str
    details: Dict[str, Any]

class AgenticRAGState(TypedDict):
    # User Query & Evolution
    query: str
    current_query: str
    session_id: str

    # Retrieval State (Level 1)
    retrieved_docs: List[DocumentChunk]
    graded_docs: List[DocumentChunk]
    retrieval_retries: int
    retrieval_fallback: bool

    # Generation & Evaluation State (Level 2)
    generation: str
    generation_retries: int
    critique: str
    groundedness_score: float
    completeness_score: float
    is_grounded: bool
    is_complete: bool

    # System & Memory State (Level 3)
    cached_answer: bool
    golden_cache_hit_id: Optional[str]
    system_log_id: Optional[str]

    # Full observability trace
    execution_trace: List[ExecutionStep]

