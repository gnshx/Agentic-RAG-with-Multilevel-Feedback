# LLD Skeleton: Core Modules

Starter code to copy into `src/`. It is a skeleton: fill in the LLM provider call and test it before relying on it.

## src/llm.py (cache, retry, call counter)

```python
import hashlib, json, os, time

class LLM:
    def __init__(self, backend, cache_path="results/llm_cache.json", max_retries=3):
        self.backend = backend            # function(prompt, temperature) -> str
        self.cache_path = cache_path
        self.max_retries = max_retries
        self.calls = 0
        self.cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    def call(self, prompt, temperature=0.0, sample_id=0):
        key = hashlib.sha256(f"{prompt}|{temperature}|{sample_id}".encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        for attempt in range(self.max_retries):
            try:
                out = self.backend(prompt, temperature)
                self.calls += 1
                self.cache[key] = out
                return out
            except Exception:
                time.sleep(2 ** attempt)
        raise RuntimeError("LLM call failed after retries")

    def save(self):
        json.dump(self.cache, open(self.cache_path, "w"))
```

Note: pass a different `sample_id` for each of the 3 self-consistency samples, otherwise the cache returns the same answer three times.

## src/signals.py (the four feedback signals)

```python
import json, re
from collections import Counter

def _json(text, default):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else default
    except Exception:
        return default

def norm(a):
    return re.sub(r"[^a-z0-9 ]", "", a.lower()).strip()

def consistency(samples):                       # S1
    normed = [norm(s) for s in samples]
    return Counter(normed).most_common(1)[0][1] / len(normed)

def retrieval_grade(llm, q, docs):              # S2
    if not docs: return 0.0
    prompt = ("Question: {q}\nFor each passage say if it helps answer the question.\n"
              "Return JSON only: {{\"relevant\": [true/false, ...]}}\n{p}").format(
              q=q, p="\n".join(f"[{i}] {d['text']}" for i, d in enumerate(docs)))
    r = _json(llm.call(prompt), {"relevant": []}).get("relevant", [])
    return sum(bool(x) for x in r) / len(docs) if r else 0.0

def groundedness(llm, answer, docs):            # S3
    ev = "\n".join(d["text"] for d in docs)
    prompt = ("Split the answer into factual claims. For each, say whether the evidence "
              "supports it.\nReturn JSON only: {{\"supported\": [true/false, ...]}}\n"
              "Answer: {a}\nEvidence:\n{e}").format(a=answer, e=ev)
    r = _json(llm.call(prompt), {"supported": []}).get("supported", [])
    return sum(bool(x) for x in r) / len(r) if r else 0.0

def completeness(llm, q, answer):               # S4
    prompt = ("List the parts the question requires, and for each say whether the answer "
              "addresses it.\nReturn JSON only: {{\"answered\": [true/false, ...]}}\n"
              "Question: {q}\nAnswer: {a}").format(q=q, a=answer)
    r = _json(llm.call(prompt), {"answered": []}).get("answered", [])
    return sum(bool(x) for x in r) / len(r) if r else 0.0
```

A parse failure returns 0.0, which forces escalation (the safe direction).

## src/controller.py (escalation logic)

```python
import time

def run_controller(question, levels, cfg, llm):
    state = {"question": question, "level": 0, "docs": [], "answer": "",
             "signals": {}, "trace": [], "status": "ok"}
    start_calls, t0 = llm.calls, time.time()

    while True:
        state = levels[state["level"]](state)          # runs level, fills answer + signals
        used = llm.calls - start_calls
        state["trace"].append({"level": state["level"], "signals": dict(state["signals"]),
                               "calls": used})
        if used >= cfg["budget"]["max_calls_per_query"]:
            state["status"] = "budget_exhausted"; break
        if state["level"] == 3 or passes(state["level"], state["signals"], cfg):
            break
        state["level"] += 1

    state["calls"], state["latency"] = llm.calls - start_calls, time.time() - t0
    return state

def passes(level, s, cfg):
    L = cfg["levels"]
    if level == 0: return s["S1"] >= L["L0"]["tau"]
    if level == 1: return s["S2"] >= L["L1"]["tau_retrieval"] and s["S3"] >= L["L1"]["tau_ground"]
    if level == 2: return s["S3"] >= L["L2"]["tau_ground"] and s["S4"] >= L["L2"]["tau_complete"]
    return True
```

## Offline threshold sweep (src/eval/sweep_thresholds.py idea)

Because `run_feedbackrag.py` logs the signals and answers for **every level of every question**, you can replay any threshold choice without new LLM calls:

```python
def replay(records, cfg):
    # records[q][level] = {"answer":..., "signals":..., "calls":...}
    total_calls, preds = 0, {}
    for q, by_level in records.items():
        for lvl in range(4):
            total_calls += by_level[lvl]["calls"]
            if lvl == 3 or passes(lvl, by_level[lvl]["signals"], cfg):
                preds[q] = by_level[lvl]["answer"]; break
    return preds, total_calls / len(records)
```

Sweep the thresholds over a grid on dev, pick the best F1-versus-calls trade-off, then apply the chosen values once to test.

## Prompt templates (starting points)

- **Decompose:** "Break this question into at most 3 simple sub-questions that must be answered in order. Return JSON: {\"subquestions\": [...]}".
- **Generate:** "Answer using only the passages below. Cite passage numbers like [1]. If the passages do not contain the answer, say you cannot find it."
- **Critic (L3):** "Check the draft against the passages. List unsupported or missing points. Return JSON: {\"issues\": [...]}".
- **Revise (L3):** "Rewrite the answer to fix these issues using only the passages."
