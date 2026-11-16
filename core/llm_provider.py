import json
import re
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional
from config.settings import settings

class LLMResponse:
    def __init__(self, content: str, raw: Optional[Dict[str, Any]] = None):
        self.content = content.strip()
        self.raw = raw or {}

    def to_json(self) -> Dict[str, Any]:
        """Safely extract JSON object from markdown blocks or raw text."""
        cleaned = self.content.strip()
        # Remove markdown fences if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback regex search for json block
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
            return {"raw_text": self.content}

class BaseLLM:
    def invoke(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        raise NotImplementedError

class LocalMockLLM(BaseLLM):
    """
    Deterministic, rule-based simulator that allows 100% offline verification,
    unit testing, and demo execution without requiring GPU or external API keys.
    """
    def invoke(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        prompt_lower = prompt.lower()
        sys_lower = (system_prompt or "").lower()

        # 1. Document Grader Prompt
        if "document relevance grader" in prompt_lower or "relevance grader" in sys_lower:
            is_relevant = True
            relevance_score = 0.85
            reason = "Document contains direct semantic keywords and conceptual matches for the query."

            # Check for explicitly irrelevant dummy test conditions
            if "banana" in prompt_lower and "neural" in prompt_lower:
                is_relevant = False
                relevance_score = 0.20
                reason = "Document discusses unrelated subject matter."

            return LLMResponse(json.dumps({
                "relevant": is_relevant,
                "score": relevance_score,
                "reason": reason
            }))

        # 2. Query Rewriter Prompt
        if "query rewriter in an agentic rag" in prompt_lower or "query rewriter" in sys_lower:
            match = re.search(r"original query:\s*([^\n\r]+)", prompt, re.IGNORECASE)
            base_q = match.group(1).strip() if match else prompt.strip().split("\n")[-1]
            expanded_query = f"{base_q} core architecture principles and system components"
            return LLMResponse(json.dumps({
                "rewritten_query": expanded_query,
                "reasoning": "Expanded query with domain terminology and semantic variations for higher vector recall."
            }))

        # 3. Groundedness / Faithfulness Auditor Prompt
        if "faithfulness & hallucination auditor" in prompt_lower:
            # Check if the generated answer specifically contains flagged test phrases
            is_hallucinated = "unsupported hallucination" in prompt_lower or "fabricated hallucination" in prompt_lower
            if is_hallucinated:
                return LLMResponse(json.dumps({
                    "grounded": False,
                    "groundedness_score": 0.45,
                    "unsupported_claims": ["Contains statements not corroborated by the retrieved documents."],
                    "critique": "Remove claims regarding untested components and restrict output solely to the provided context."
                }))
            return LLMResponse(json.dumps({
                "grounded": True,
                "groundedness_score": 0.95,
                "unsupported_claims": [],
                "critique": "All statements are strictly supported by retrieved context."
            }))

        # 4. Answer Completeness Auditor Prompt
        if "answer completeness auditor" in prompt_lower:
            is_incomplete = "incomplete" in prompt_lower or "too brief" in prompt_lower
            if is_incomplete:
                return LLMResponse(json.dumps({
                    "complete": False,
                    "completeness_score": 0.50,
                    "missing_aspects": ["The answer omits operational execution details."],
                    "critique": "Expand the response to explicitly detail how the feedback loops are wired."
                }))
            return LLMResponse(json.dumps({
                "complete": True,
                "completeness_score": 0.92,
                "missing_aspects": [],
                "critique": "Response thoroughly answers all facets of the query."
            }))

        # 5. Generator Prompt (Synthesizes answer from context)
        if "provide a comprehensive, faithful answer:" in prompt_lower or ("context:" in prompt_lower and "question:" in prompt_lower):
            return LLMResponse(
                "Based on the verified context, the Agentic RAG architecture incorporates a multi-tiered "
                "feedback mechanism. At Level 1, queries are graded and dynamically rewritten if initial "
                "vector retrieval falls below relevance thresholds. At Level 2, generated candidates undergo "
                "rigorous factuality and completeness audits, triggering targeted generator revisions if "
                "unsupported claims appear. At Level 3, human feedback logs and dynamic document re-weighting "
                "continuously optimize long-term retrieval and cache verified golden exemplars."
            )

        # 6. System Diagnostics / SOP Architect
        if "system sop architect" in prompt_lower:
            return LLMResponse(json.dumps({
                "diagnosis": "System operates within target latency; minor retrieval retries observed for acronym-heavy queries.",
                "recommended_sop_update": "Pre-expand query acronyms before similarity search to decrease Level 1 retries.",
                "confidence": 0.88
            }))

        # Generic fallback
        return LLMResponse("Processed query successfully with local agent intelligence.")

class OllamaLLM(BaseLLM):
    """Local Ollama client using HTTP interface."""
    def __init__(self, base_url: str, model: str, allow_fallback: bool = False):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.allow_fallback = allow_fallback

    def invoke(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                res = json.loads(response.read().decode("utf-8"))
                return LLMResponse(content=res.get("response", ""), raw=res)
        except Exception as e:
            if self.allow_fallback:
                print(f"[Warning] Ollama connection failed ({e}). Falling back to LocalMockLLM.")
                return LocalMockLLM().invoke(prompt, system_prompt)
            raise RuntimeError(
                f"Ollama invocation failed at {self.base_url} with model '{self.model}': {e}. "
                "Ensure Ollama is running or switch 'llm_provider' to 'local_mock' in config/settings."
            ) from e

class OpenAICompatibleLLM(BaseLLM):
    """OpenAI compatible HTTP client."""
    def __init__(self, api_base: str, api_key: str, model: str = "gpt-4o-mini", allow_fallback: bool = False):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.allow_fallback = allow_fallback

    def invoke(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        url = f"{self.api_base}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                res = json.loads(response.read().decode("utf-8"))
                content = res["choices"][0]["message"]["content"]
                return LLMResponse(content=content, raw=res)
        except Exception as e:
            if self.allow_fallback:
                print(f"[Warning] OpenAI API connection failed ({e}). Falling back to LocalMockLLM.")
                return LocalMockLLM().invoke(prompt, system_prompt)
            raise RuntimeError(
                f"OpenAI-compatible LLM invocation failed at {self.api_base} with model '{self.model}': {e}. "
                "Verify API endpoint and credentials or switch 'llm_provider' to 'local_mock' in config/settings."
            ) from e

def get_llm(allow_fallback: bool = False) -> BaseLLM:
    """Factory method to get the configured LLM provider."""
    provider = settings.llm_provider.lower().strip()
    if provider == "ollama":
        return OllamaLLM(base_url=settings.ollama_base_url, model=settings.ollama_model, allow_fallback=allow_fallback)
    elif provider == "openai_compatible" and settings.openai_api_key:
        return OpenAICompatibleLLM(api_base=settings.openai_api_base, api_key=settings.openai_api_key, allow_fallback=allow_fallback)
    return LocalMockLLM()
