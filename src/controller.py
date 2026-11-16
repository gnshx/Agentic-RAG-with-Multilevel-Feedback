import time
from typing import Dict, Any, List, Callable, Optional
from src.llm import LLM
from src.learned_policy import LearnedEscalationPolicy
from src.cost_tracker import CostTracker

def passes(level: int, signals: Dict[str, float], cfg: Dict[str, Any]) -> bool:
    """
    Evaluates whether the signals at the current level meet the promotion/exit thresholds.
    Returns True to RETURN (pass), or False to ESCALATE to the next level.
    """
    levels_cfg = cfg.get("levels", {})

    if level == 0:
        tau_0 = levels_cfg.get("L0", {}).get("tau", 0.67)
        return float(signals.get("S1", 0.0)) >= tau_0

    if level == 1:
        tau_retrieval = levels_cfg.get("L1", {}).get("tau_retrieval", 0.50)
        tau_ground = levels_cfg.get("L1", {}).get("tau_ground", 0.80)
        return (float(signals.get("S2", 0.0)) >= tau_retrieval and
                float(signals.get("S3", 0.0)) >= tau_ground)

    if level == 2:
        tau_ground = levels_cfg.get("L2", {}).get("tau_ground", 0.80)
        tau_complete = levels_cfg.get("L2", {}).get("tau_complete", 0.80)
        return (float(signals.get("S3", 0.0)) >= tau_ground and
                float(signals.get("S4", 0.0)) >= tau_complete)

    # Level 3 is the terminal level, always returns
    return True

def run_controller(
    question: str,
    levels: Dict[int, Callable[[Dict[str, Any]], Dict[str, Any]]],
    cfg: Dict[str, Any],
    llm: LLM,
    policy: Optional[LearnedEscalationPolicy] = None,
    mode: str = "thresholds"
) -> Dict[str, Any]:
    """
    Multilevel Feedback Controller (MLFQ pattern):
    Starts processing at Level 0 (cheapest direct generation) and conditionally
    escalates through Level 1, Level 2, and Level 3.
    Supports either:
    1. 'thresholds': Rule-based signal gating with configured tau thresholds
    2. 'learned': Machine-learned escalation policy predicting escalation utility
    Enforces budget guard and logs tokens, dollars, and latencies.
    """
    cost_tracker = CostTracker()
    state: Dict[str, Any] = {
        "question": question,
        "level": 0,
        "docs": [],
        "subquestions": [],
        "drafts": [],
        "answer": "",
        "citations": [],
        "signals": {},
        "calls": 0,
        "latency": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "trace": [],
        "status": "ok",
        "mode": mode
    }

    start_calls = getattr(llm, "logical_calls", llm.calls)
    start_cost = getattr(llm, "total_cost_usd", 0.0)
    start_ptokens = getattr(llm, "total_prompt_tokens", 0)
    start_ctokens = getattr(llm, "total_completion_tokens", 0)
    t0 = time.time()
    max_calls = cfg.get("budget", {}).get("max_calls_per_query", 20)

    while True:
        current_lvl = state["level"]
        level_func = levels.get(current_lvl)
        if not level_func:
            break

        level_t0 = time.time()
        calls_before = getattr(llm, "logical_calls", llm.calls)
        cost_before = getattr(llm, "total_cost_usd", 0.0)

        # Execute current level
        state = level_func(state)

        curr_calls = getattr(llm, "logical_calls", llm.calls)
        calls_in_level = curr_calls - calls_before
        total_used_calls = curr_calls - start_calls
        level_cost = round(getattr(llm, "total_cost_usd", 0.0) - cost_before, 7)
        level_latency = round(time.time() - level_t0, 4)

        # Determine promotion vs escalation
        should_escalate = False
        decision_meta = {}

        if current_lvl >= 3:
            should_escalate = False
            decision_meta = {"action": "terminal_return", "reason": "Reached Level 3"}
        elif mode == "learned" and policy is not None and policy.is_fitted:
            esc, prob = policy.should_escalate(state, current_lvl)
            should_escalate = esc
            decision_meta = {
                "action": "escalate" if esc else "return",
                "prob_escalate": prob,
                "policy": policy.model_type
            }
        else:
            # Rule-based thresholds
            pass_level = passes(current_lvl, state["signals"], cfg)
            should_escalate = not pass_level
            decision_meta = {
                "action": "escalate" if should_escalate else "return",
                "passed_thresholds": pass_level
            }

        # Record step in trace
        state["trace"].append({
            "level": current_lvl,
            "signals": dict(state.get("signals", {})),
            "calls_used_level": calls_in_level,
            "total_calls_so_far": total_used_calls,
            "cost_usd_level": level_cost,
            "level_latency": level_latency,
            "decision": decision_meta,
            "answer_preview": state.get("answer", "")[:100]
        })

        # Check budget guard
        if total_used_calls >= max_calls:
            state["status"] = "budget_exhausted"
            break

        # Stop if no escalation requested
        if not should_escalate:
            break

        # Escalate to next level
        state["level"] += 1

    state["calls"] = getattr(llm, "logical_calls", llm.calls) - start_calls
    state["latency"] = round(time.time() - t0, 4)
    state["prompt_tokens"] = getattr(llm, "total_prompt_tokens", 0) - start_ptokens
    state["completion_tokens"] = getattr(llm, "total_completion_tokens", 0) - start_ctokens
    state["total_tokens"] = state["prompt_tokens"] + state["completion_tokens"]
    state["cost_usd"] = round(getattr(llm, "total_cost_usd", 0.0) - start_cost, 6)

    return state


