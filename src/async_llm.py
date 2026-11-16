import asyncio
import time
import json
import httpx
from typing import List, Dict, Any, Optional, Callable
from src.cost_tracker import CostTracker

class AsyncOllamaClient:
    """
    High-performance asynchronous non-blocking client for local and remote Ollama backends.
    Supports connection pooling, bounded concurrency, wall-clock latency timing,
    and automatic exponential backoff.
    """
    def __init__(
        self,
        base_url: str = "http://localhost:11435",
        model: str = "qwen2.5-coder:7b",
        timeout: float = 60.0,
        max_concurrency: int = 4
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.cost_tracker = CostTracker()
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    async def generate(self, prompt: str, temperature: float = 0.0, max_retries: int = 3) -> Dict[str, Any]:
        """
        Execute an async completion request and capture authentic wall-clock metrics.
        Returns dict with: response, latency_ms, prompt_tokens, completion_tokens.
        """
        async with self.semaphore:
            if self.client is None:
                self.client = httpx.AsyncClient(timeout=self.timeout)

            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature}
            }

            p_tokens = self.cost_tracker.estimate_tokens(prompt)

            for attempt in range(max_retries):
                t0 = time.perf_counter()
                try:
                    res = await self.client.post(url, json=payload)
                    dt_ms = (time.perf_counter() - t0) * 1000.0

                    if res.status_code == 200:
                        data = res.json()
                        text = data.get("response", "").strip()
                        c_tokens = self.cost_tracker.estimate_tokens(text)
                        cost_usd = self.cost_tracker.calculate_cost(p_tokens, c_tokens)

                        return {
                            "response": text,
                            "latency_ms": dt_ms,
                            "prompt_tokens": p_tokens,
                            "completion_tokens": c_tokens,
                            "cost_usd": cost_usd,
                            "success": True,
                            "attempt": attempt + 1
                        }
                    else:
                        await asyncio.sleep(min(2 ** attempt, 4))
                except Exception as e:
                    if attempt == max_retries - 1:
                        dt_ms = (time.perf_counter() - t0) * 1000.0
                        return {
                            "response": "",
                            "latency_ms": dt_ms,
                            "prompt_tokens": p_tokens,
                            "completion_tokens": 0,
                            "cost_usd": 0.0,
                            "success": False,
                            "error": str(e),
                            "attempt": attempt + 1
                        }
                    await asyncio.sleep(min(2 ** attempt, 4))

            return {"response": "", "latency_ms": 0.0, "success": False, "error": "Max retries exceeded"}

    async def batch_generate(self, prompts: List[str], temperature: float = 0.0) -> List[Dict[str, Any]]:
        """Run a batch of prompts concurrently within semaphore limits."""
        tasks = [self.generate(p, temperature=temperature) for p in prompts]
        return await asyncio.gather(*tasks)
