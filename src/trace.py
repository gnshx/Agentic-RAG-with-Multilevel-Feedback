import json
import os
from typing import Dict, Any, List

def safe_json_default(obj: Any) -> Any:
    """Safe serializer for numpy scalars, decimals, and custom objects."""
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "__float__"):
        return float(obj)
    if hasattr(obj, "__int__"):
        return int(obj)
    return str(obj)

class TraceLogger:
    """
    Appends per-query execution traces and feedback signals to persistent JSONL files
    for evaluation, offline threshold tuning, and ablation studies.
    """
    def __init__(self, trace_file: str = "results/trace.jsonl", signals_file: str = "results/signals.jsonl"):
        self.trace_file = trace_file
        self.signals_file = signals_file
        os.makedirs(os.path.dirname(self.trace_file) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.signals_file) or ".", exist_ok=True)

    def log_query_trace(self, record: Dict[str, Any]):
        with open(self.trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=safe_json_default) + "\n")

    def log_all_level_signals(self, question_id: str, question: str, levels_data: Dict[int, Dict[str, Any]]):
        entry = {
            "id": question_id,
            "question": question,
            "by_level": levels_data
        }
        with open(self.signals_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=safe_json_default) + "\n")

