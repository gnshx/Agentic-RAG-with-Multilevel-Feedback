import time
import numpy as np
from typing import Dict, Any, List, Optional

# Standard reference token pricing ($/1M tokens)
# e.g., lightweight/fast instruction models (comparable to GPT-4o-mini / Gemini-1.5-Flash / Qwen2.5-3B)
INPUT_PRICE_PER_1M = 0.150   # $0.150 per 1M input tokens
OUTPUT_PRICE_PER_1M = 0.600  # $0.600 per 1M output tokens

class CostTracker:
    """
    Tracks token counts, execution latencies, and estimated financial costs ($ USD)
    per query and across levels, providing p50/p90/p95 latency metrics.
    """
    def __init__(self, input_rate_per_1m: float = INPUT_PRICE_PER_1M, output_rate_per_1m: float = OUTPUT_PRICE_PER_1M):
        self.input_rate = input_rate_per_1m
        self.output_rate = output_rate_per_1m

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Fast, robust token estimator: 1 token ~= 4 characters / 0.75 words,
        ensuring zero latency overhead in benchmark loops.
        """
        if not text:
            return 0
        char_count = len(text)
        word_count = len(text.split())
        return max(1, int(round((char_count / 4.0 + word_count * 1.3) / 2.0)))

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        cost = (prompt_tokens / 1_000_000.0) * self.input_rate + (completion_tokens / 1_000_000.0) * self.output_rate
        return round(cost, 7)

    def summarize_latencies(self, latencies: List[float]) -> Dict[str, float]:
        if not latencies:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
        arr = np.array(latencies)
        return {
            "p50": round(float(np.percentile(arr, 50)), 4),
            "p90": round(float(np.percentile(arr, 90)), 4),
            "p95": round(float(np.percentile(arr, 95)), 4),
            "p99": round(float(np.percentile(arr, 99)), 4),
            "mean": round(float(np.mean(arr)), 4)
        }

