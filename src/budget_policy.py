import os
import pickle
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from src.cost_tracker import CostTracker
from src.signals import retrieval_score_margin, lexical_groundedness_proxy
from src.learned_policy import extract_level_features, FEATURE_NAMES

# Dial values required for budget sweeps
DEFAULT_LAMBDAS = [0.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 2000.0, 5000.0]

BUDGET_FEATURE_NAMES = FEATURE_NAMES + [
    "retrieval_score_margin",
    "lexical_groundedness_proxy"
]


def extract_budget_features(state_or_record: Dict[str, Any], level: int) -> np.ndarray:
    """
    Extracts multi-signal feature vector for cost-aware budget policy:
    - 11 base features (S1-S4 signals, question length, answer length, doc counts, top score, level)
    - 2 cheap signals (retrieval score margin and lexical groundedness proxy)
    """
    base_feat = extract_level_features(state_or_record, level)

    docs = state_or_record.get("docs", [])
    ans = state_or_record.get("answer", "")

    # Cheap signals computed at zero LLM cost
    margin = retrieval_score_margin(docs)
    lex_gnd = lexical_groundedness_proxy(ans, docs, method="overlap") if (ans and docs) else 0.0

    return np.append(base_feat, [margin, lex_gnd]).astype(np.float32)


def get_record_cost(data: Dict[str, Any], tracker: Optional[CostTracker] = None) -> float:
    """Computes or extracts the dollar cost of a level execution record."""
    if "cost_usd" in data and float(data["cost_usd"]) > 0.0:
        return float(data["cost_usd"])
    if tracker is None:
        tracker = CostTracker()
    ans = data.get("answer", "")
    docs_text = " ".join(d.get("text", "") for d in data.get("docs", []))
    q = data.get("question", "")
    p_tok = tracker.estimate_tokens(q + " " + docs_text)
    c_tok = tracker.estimate_tokens(ans)
    cost = tracker.calculate_cost(p_tok, c_tok)
    if cost <= 0.0 and data.get("calls", 0) > 0:
        # Fallback based on average calls (~$0.000010 per call)
        cost = float(data.get("calls", 0)) * 0.000010
    return float(cost)


class BudgetDialPolicy:
    """
    Cost-Aware Escalation Policy with Continuous Budget Dial (lambda):
    Models agentic escalation under explicit budget constraints:
      Escalate from level L to L+1 iff:
        E[Delta F1] >= lambda * Delta Cost
    where:
      - E[Delta F1] is predicted by a supervised regression model trained on dev signals.
      - Delta Cost is the expected incremental monetary cost of executing level L+1.
      - lambda in [0, inf) is the operational trade-off knob (0 = max quality, high = budget conserving).
    """

    def __init__(
        self,
        lambda_dial: float = 1000.0,
        model_type: str = "gradient_boosting",
        budget_lambda: Optional[float] = None
    ):
        if budget_lambda is not None:
            lambda_dial = budget_lambda
        self.lambda_dial = float(lambda_dial)
        self.model_type = model_type
        if model_type == "ridge":
            self.model = Ridge(alpha=1.0, random_state=42)
        else:
            self.model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)

        self.is_fitted = False
        self.feature_names = BUDGET_FEATURE_NAMES
        self.cost_tracker = CostTracker()
        # Default empirical incremental costs ($ USD) by transition L -> L+1
        self.mean_delta_costs: Dict[int, float] = {
            0: 2.25e-5,  # L0 -> L1
            1: 1.80e-5,  # L1 -> L2
            2: 3.69e-5   # L2 -> L3
        }

    def train_on_dev_signals(
        self,
        dev_records: Dict[str, Dict[int, Dict[str, Any]]],
        split_name: str = "dev"
    ) -> Dict[str, Any]:
        """
        Trains the expected F1-gain regressor strictly on dev split transitions.
        Enforces data leakage guard.
        """
        assert split_name.lower() == "dev", (
            f"DATA LEAKAGE VIOLATION: Budget policy MUST ONLY be trained on 'dev' split! "
            f"Received: '{split_name}'."
        )

        X_list: List[np.ndarray] = []
        y_list: List[float] = []
        delta_costs: Dict[int, List[float]] = {0: [], 1: [], 2: []}

        for qid, by_level in dev_records.items():
            for lvl in range(3):
                curr_data = by_level[lvl]
                next_data = by_level[lvl + 1]

                feat = extract_budget_features(curr_data, lvl)
                curr_f1 = float(curr_data.get("f1", 0.0))
                next_f1 = float(next_data.get("f1", 0.0))
                delta_f1 = next_f1 - curr_f1

                X_list.append(feat)
                y_list.append(delta_f1)

                inc_cost = get_record_cost(next_data, self.cost_tracker)
                delta_costs[lvl].append(inc_cost)

        X = np.array(X_list)
        y = np.array(y_list)

        self.model.fit(X, y)
        self.is_fitted = True

        for lvl in range(3):
            if delta_costs[lvl]:
                self.mean_delta_costs[lvl] = float(np.mean(delta_costs[lvl]))

        preds = self.model.predict(X)
        mae = float(np.mean(np.abs(preds - y)))

        return {
            "model_type": self.model_type,
            "training_samples": len(X),
            "mae_delta_f1": round(mae, 4),
            "mean_delta_costs": {k: round(v, 7) for k, v in self.mean_delta_costs.items()}
        }

    def predict_f1_gain(self, state_or_record: Dict[str, Any], level: int) -> float:
        """Predicts expected F1 gain E[Delta F1] if escalating from current level."""
        if level >= 3:
            return 0.0
        if self.is_fitted:
            feat = extract_budget_features(state_or_record, level).reshape(1, -1)
            pred = float(self.model.predict(feat)[0])
            return round(pred, 4)

        # Heuristic fallback before fitting
        signals = state_or_record.get("signals", {})
        if level == 0:
            s1 = float(signals.get("S1", 0.33))
            return round(max(0.05, 0.70 - s1), 4)
        elif level == 1:
            s2 = float(signals.get("S2", 0.0))
            s3 = float(signals.get("S3", 0.0))
            return round(max(0.05, 0.60 - min(s2, s3)), 4)
        elif level == 2:
            s3 = float(signals.get("S3", 0.0))
            s4 = float(signals.get("S4", 0.0))
            return round(max(0.05, 0.50 - min(s3, s4)), 4)
        return 0.0

    def estimate_delta_cost(self, state_or_record: Dict[str, Any], level: int) -> float:
        """Estimates incremental cost of escalating from level to level + 1."""
        if level >= 3:
            return 0.0
        return self.mean_delta_costs.get(level, 2.5e-5)

    def should_escalate(
        self,
        state_or_record: Dict[str, Any],
        level: int,
        lambda_dial: Optional[float] = None
    ) -> Tuple[bool, float, float]:
        """
        Decides whether to escalate from current level.
        Returns:
            (should_escalate: bool, expected_gain: float, cost_threshold: float)
        """
        if level >= 3:
            return False, 0.0, 0.0

        dial = self.lambda_dial if lambda_dial is None else float(lambda_dial)
        pred_gain = self.predict_f1_gain(state_or_record, level)
        delta_cost = self.estimate_delta_cost(state_or_record, level)

        cost_threshold = round(dial * delta_cost, 5)
        decision = bool(pred_gain >= cost_threshold)
        return decision, pred_gain, cost_threshold

    def evaluate_on_records(
        self,
        records: Dict[str, Dict[int, Dict[str, Any]]],
        lambda_dial: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluates the policy across a dictionary of query level records for a specific lambda.
        """
        dial = self.lambda_dial if lambda_dial is None else float(lambda_dial)
        f1_list: List[float] = []
        em_list: List[float] = []
        calls_list: List[float] = []
        cost_list: List[float] = []
        resolved_levels: List[int] = []

        for qid, by_lvl in records.items():
            curr_lvl = 0
            cum_calls = 0.0
            cum_cost = 0.0

            while curr_lvl < 3:
                lvl_data = by_lvl[curr_lvl]
                cum_calls += float(lvl_data.get("calls", 0.0))
                cum_cost += get_record_cost(lvl_data, self.cost_tracker)

                # Check escalation decision
                escalate, _, _ = self.should_escalate(lvl_data, curr_lvl, lambda_dial=dial)
                if escalate:
                    curr_lvl += 1
                else:
                    break

            if curr_lvl == 3:
                lvl_data = by_lvl[3]
                cum_calls += float(lvl_data.get("calls", 0.0))
                cum_cost += get_record_cost(lvl_data, self.cost_tracker)

            final_data = by_lvl[curr_lvl]
            f1_list.append(float(final_data.get("f1", 0.0)))
            em_list.append(float(final_data.get("em", 0.0)))
            calls_list.append(cum_calls)
            cost_list.append(cum_cost)
            resolved_levels.append(curr_lvl)

        n = max(len(records), 1)
        level_dist = {
            f"Level_{lvl}": f"{(resolved_levels.count(lvl) / n) * 100:.1f}% ({resolved_levels.count(lvl)}/{n})"
            for lvl in range(4)
        }

        return {
            "lambda": dial,
            "num_queries": len(records),
            "f1": round(float(np.mean(f1_list)) * 100, 2),
            "em": round(float(np.mean(em_list)) * 100, 2),
            "avg_calls": round(float(np.mean(calls_list)), 2),
            "avg_cost_usd": round(float(np.mean(cost_list)), 6),
            "total_cost_usd": round(float(np.sum(cost_list)), 5),
            "level_distribution": level_dist,
            "resolved_counts": {lvl: resolved_levels.count(lvl) for lvl in range(4)}
        }

    def sweep_lambda(
        self,
        records: Dict[str, Dict[int, Dict[str, Any]]],
        lambdas: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Sweeps lambda values (default [0, 10, 50, 100, 500, 1000, 2000, 5000])
        and returns the Pareto evaluation frontier.
        """
        if lambdas is None:
            lambdas = DEFAULT_LAMBDAS

        results = []
        for lam in lambdas:
            res = self.evaluate_on_records(records, lambda_dial=lam)
            results.append(res)
        return results

    def save(self, filepath: str = "results/budget_policy.pkl"):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str = "results/budget_policy.pkl") -> Optional["BudgetDialPolicy"]:
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                return pickle.load(f)
        return None


# Alias for backward-compatibility
CostAwareBudgetPolicy = BudgetDialPolicy


# Alias for compatibility with src/__init__.py
BudgetDialPolicy = CostAwareBudgetPolicy



def compute_oracle(
    records: Dict[str, Dict[int, Dict[str, Any]]],
    lambda_dial: Optional[float] = None
) -> Dict[str, Any]:
    """
    Computes the theoretical Oracle Policy:
    - If lambda_dial is None or 0: Unconstrained Accuracy Oracle (selects level with max F1 per query).
    - If lambda_dial > 0: Cost-Aware Oracle (maximizes F1_l - lambda * CumulativeCost_l per query).
    Ties broken in favor of the lowest level (least compute).
    """
    tracker = CostTracker()
    chosen_levels: List[int] = []
    f1_list: List[float] = []
    em_list: List[float] = []
    calls_list: List[float] = []
    cost_list: List[float] = []
    per_query: Dict[str, int] = {}

    for qid, by_lvl in records.items():
        best_lvl = 0
        best_score = -float("inf")
        best_cum_cost = 0.0
        best_cum_calls = 0.0

        cum_calls = 0.0
        cum_cost = 0.0

        for lvl in range(4):
            lvl_data = by_lvl[lvl]
            cum_calls += float(lvl_data.get("calls", 0.0))
            cum_cost += get_record_cost(lvl_data, tracker)

            cur_f1 = float(lvl_data.get("f1", 0.0))
            if lambda_dial is None or lambda_dial == 0.0:
                utility = cur_f1
            else:
                utility = cur_f1 - float(lambda_dial) * cum_cost

            # Strict inequality ensures tie-breaking in favor of earliest level
            if utility > best_score:
                best_score = utility
                best_lvl = lvl
                best_cum_cost = cum_cost
                best_cum_calls = cum_calls

        chosen_levels.append(best_lvl)
        final_data = by_lvl[best_lvl]
        f1_list.append(float(final_data.get("f1", 0.0)))
        em_list.append(float(final_data.get("em", 0.0)))
        calls_list.append(best_cum_calls)
        cost_list.append(best_cum_cost)
        per_query[qid] = best_lvl

    n = max(len(records), 1)
    level_dist = {
        f"Level_{lvl}": f"{(chosen_levels.count(lvl) / n) * 100:.1f}% ({chosen_levels.count(lvl)}/{n})"
        for lvl in range(4)
    }

    return {
        "policy": "Oracle",
        "lambda": lambda_dial,
        "num_queries": len(records),
        "f1": round(float(np.mean(f1_list)) * 100, 2),
        "em": round(float(np.mean(em_list)) * 100, 2),
        "avg_calls": round(float(np.mean(calls_list)), 2),
        "avg_cost_usd": round(float(np.mean(cost_list)), 6),
        "level_distribution": level_dist,
        "resolved_counts": {lvl: chosen_levels.count(lvl) for lvl in range(4)},
        "per_query_choices": per_query
    }


def compute_headroom(
    policy_result: Dict[str, Any],
    oracle_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Computes empirical headroom: difference between Oracle upper bound and Policy achieved metrics.
    """
    oracle_f1 = float(oracle_result.get("f1", 0.0))
    policy_f1 = float(policy_result.get("f1", 0.0))
    oracle_em = float(oracle_result.get("em", 0.0))
    policy_em = float(policy_result.get("em", 0.0))

    gap_f1 = round(oracle_f1 - policy_f1, 2)
    gap_em = round(oracle_em - policy_em, 2)
    gap_f1_pct = round((gap_f1 / oracle_f1) * 100, 2) if oracle_f1 > 0 else 0.0

    return {
        "oracle_f1": oracle_f1,
        "policy_f1": policy_f1,
        "headroom_f1": gap_f1,
        "headroom_f1_pct": gap_f1_pct,
        "oracle_em": oracle_em,
        "policy_em": policy_em,
        "headroom_em": gap_em,
        "calls_ratio": round(float(policy_result.get("avg_calls", 1.0)) / max(float(oracle_result.get("avg_calls", 1.0)), 0.01), 2),
        "cost_ratio": round(float(policy_result.get("avg_cost_usd", 1.0)) / max(float(oracle_result.get("avg_cost_usd", 1.0)), 1e-8), 2)
    }
