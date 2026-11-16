import os
import pickle
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

FEATURE_NAMES = [
    "S1_consistency",
    "S2_retrieval_grade",
    "S3_groundedness",
    "S4_completeness",
    "q_len_char",
    "q_len_words",
    "ans_len_char",
    "ans_len_words",
    "num_retrieved_chunks",
    "top_retrieval_score",
    "current_level"
]

def extract_level_features(state_or_record: Dict[str, Any], level: int) -> np.ndarray:
    """
    Extracts dense multi-modal feature vector for the given query state at a specific level:
    - 4 feedback signals (S1, S2, S3, S4)
    - Query length (chars and words)
    - Answer length (chars and words)
    - Number of retrieved evidence chunks
    - Top retrieval fusion score
    - Level identifier
    """
    signals = state_or_record.get("signals", {})
    q = state_or_record.get("question", "")
    ans = state_or_record.get("answer", "")
    docs = state_or_record.get("docs", [])

    s1 = float(signals.get("S1", 0.0))
    s2 = float(signals.get("S2", 0.0))
    s3 = float(signals.get("S3", 0.0))
    s4 = float(signals.get("S4", 0.0))

    q_char = float(len(q))
    q_words = float(len(q.split()))
    ans_char = float(len(ans))
    ans_words = float(len(ans.split()))

    num_chunks = float(len(docs))
    top_score = float(docs[0].get("score", 0.0)) if docs else 0.0
    lvl_val = float(level)

    return np.array([
        s1, s2, s3, s4,
        q_char, q_words,
        ans_char, ans_words,
        num_chunks, top_score,
        lvl_val
    ], dtype=np.float32)

class LearnedEscalationPolicy:
    """
    Machine Learning Escalation Policy:
    Replaces static hand-crafted threshold heuristics with a trained classifier
    (Gradient Boosting or Logistic Regression) that predicts the utility of escalating
    to the next processing level based on observed feedback signals and query context.
    """
    def __init__(self, model_type: str = "gradient_boosting"):
        self.model_type = model_type
        if model_type == "logistic_regression":
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            self.model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        self.is_fitted = False
        self.feature_names = FEATURE_NAMES

    def train_on_dev_signals(
        self,
        dev_records: Dict[str, Dict[int, Dict[str, Any]]],
        split_name: str = "dev"
    ) -> Dict[str, Any]:
        """
        Train escalation model strictly on the dev split signals.
        Enforces runtime safety assertion to prevent data leakage.
        """
        assert split_name.lower() == "dev", (
            f"DATA LEAKAGE VIOLATION: Learned policy MUST ONLY be trained on 'dev' split! "
            f"Received: '{split_name}'."
        )

        X_list: List[np.ndarray] = []
        y_list: List[int] = []

        # Construct training pairs from dev records
        for qid, by_level in dev_records.items():
            for lvl in range(3):  # Levels 0, 1, 2 consider escalating to lvl+1
                curr_data = by_level[lvl]
                next_data = by_level[lvl + 1]

                feat = extract_level_features(curr_data, lvl)
                curr_f1 = curr_data.get("f1", 0.0)
                next_f1 = next_data.get("f1", 0.0)

                # Ground truth target:
                # 1 = escalating improved F1, or current answer had sub-threshold quality (F1 < 0.70)
                # 0 = current answer was already optimal and next level gave no benefit
                should_escalate = 1 if (next_f1 > curr_f1 or curr_f1 < 0.70) else 0

                X_list.append(feat)
                y_list.append(should_escalate)

        X = np.array(X_list)
        y = np.array(y_list)

        # Cross-validation performance check
        cv_scores = cross_val_score(self.model, X, y, cv=3)
        self.model.fit(X, y)
        self.is_fitted = True

        report = {
            "model_type": self.model_type,
            "training_samples": len(X),
            "cv_accuracy_mean": round(float(np.mean(cv_scores)), 4),
            "cv_accuracy_std": round(float(np.std(cv_scores)), 4),
            "positive_escalation_ratio": round(float(np.mean(y)), 4)
        }
        return report

    def should_escalate(self, state: Dict[str, Any], level: int, probability_threshold: float = 0.50) -> Tuple[bool, float]:
        """
        Predicts whether the query should escalate from the current level.
        Returns (should_escalate: bool, probability: float).
        """
        if not self.is_fitted or level >= 3:
            return False, 0.0

        feat = extract_level_features(state, level).reshape(1, -1)
        prob_escalate = float(self.model.predict_proba(feat)[0][1])
        decision = prob_escalate >= probability_threshold
        return decision, round(prob_escalate, 4)

    def save(self, filepath: str = "results/learned_policy.pkl"):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str = "results/learned_policy.pkl") -> Optional["LearnedEscalationPolicy"]:
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                return pickle.load(f)
        return None
