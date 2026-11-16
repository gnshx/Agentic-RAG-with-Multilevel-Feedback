import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union


def compute_ece(
    confidences: List[float],
    accuracies: List[float],
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    Bins confidence scores into n_bins equal intervals over [0, 1].
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    Properly includes 0.0 confidence in the first bin and 1.0 in the last bin.
    """
    if not confidences or not accuracies or len(confidences) != len(accuracies):
        return {"ece": 0.0, "mce": 0.0, "n_bins": n_bins, "bins": []}

    conf = np.array(confidences, dtype=np.float64)
    acc = np.array(accuracies, dtype=np.float64)
    n = len(conf)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_details = []

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        # First bin includes left boundary 0.0; all bins include right boundary
        if i == 0:
            in_bin = (conf >= low) & (conf <= high) if n_bins == 1 else (conf >= low) & (conf <= high) if i == n_bins - 1 else (conf >= low) & (conf < high)
        elif i == n_bins - 1:
            in_bin = (conf >= low) & (conf <= high)
        else:
            in_bin = (conf >= low) & (conf < high)

        count = int(np.sum(in_bin))
        if count > 0:
            bin_acc = float(np.mean(acc[in_bin]))
            bin_conf = float(np.mean(conf[in_bin]))
            diff = abs(bin_acc - bin_conf)
            prop = count / float(n)
            ece += prop * diff
            mce = max(mce, diff)

            bin_details.append({
                "bin": f"[{low:.1f}, {high:.1f}]" if i == 0 else f"({low:.1f}, {high:.1f}]",
                "count": count,
                "acc": round(bin_acc, 4),
                "conf": round(bin_conf, 4),
                "gap": round(diff, 4)
            })

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "n_bins": n_bins,
        "bins": bin_details
    }


def compute_signals_ece(
    records: Union[List[Dict[str, Any]], Dict[str, Dict[int, Dict[str, Any]]]],
    f1_threshold: float = 0.70
) -> Dict[str, Dict[str, Any]]:
    """
    Computes Expected Calibration Error (ECE) for feedback signals S1, S2, S3, S4.
    - S1: Level 0 self-consistency agreement vs L0 correctness
    - S2: Level 1 retrieval relevance grade vs L1 correctness
    - S3: Level 1 groundedness score vs L1 correctness
    - S4: Level 2 completeness score vs L2 correctness
    """
    # Normalize input to list of per-level dicts
    flat_records: List[Dict[str, Any]] = []
    if isinstance(records, dict):
        for qid, by_lvl in records.items():
            for lvl_idx, data in by_lvl.items():
                flat_records.append({
                    "level": int(lvl_idx),
                    "signals": data.get("signals", {}),
                    "f1": float(data.get("f1", 0.0)),
                    "em": float(data.get("em", 0.0))
                })
    else:
        flat_records = records

    s1_conf, s1_acc = [], []
    s2_conf, s2_acc = [], []
    s3_conf, s3_acc = [], []
    s4_conf, s4_acc = [], []

    for item in flat_records:
        lvl = item.get("level", 0)
        signals = item.get("signals", {})
        is_correct = 1.0 if item.get("f1", 0.0) >= f1_threshold else 0.0

        if lvl == 0 and "S1" in signals:
            s1_conf.append(float(signals["S1"]))
            s1_acc.append(is_correct)
        elif lvl == 1:
            if "S2" in signals:
                s2_conf.append(float(signals["S2"]))
                s2_acc.append(is_correct)
            if "S3" in signals:
                s3_conf.append(float(signals["S3"]))
                s3_acc.append(is_correct)
        elif lvl == 2 and "S4" in signals:
            s4_conf.append(float(signals["S4"]))
            s4_acc.append(is_correct)

    return {
        "S1": compute_ece(s1_conf, s1_acc),
        "S2": compute_ece(s2_conf, s2_acc),
        "S3": compute_ece(s3_conf, s3_acc),
        "S4": compute_ece(s4_conf, s4_acc)
    }


class ConformalEscalationController:
    """
    Split Conformal Risk Control for MLFQ Early Exits.
    Calibrates early-exit confidence thresholds tau_alpha on dev split data
    such that the test empirical error rate among early exits is bounded by alpha:
        P(Error | Early Exit) <= alpha
    where alpha in {0.05, 0.10, 0.15}.
    """

    def __init__(
        self,
        target_alpha: float = 0.10,
        f1_threshold: float = 0.70
    ):
        self.target_alpha = float(target_alpha)
        self.f1_threshold = float(f1_threshold)
        # Calibrated early-exit confidence thresholds tau_alpha per level
        self.calibrated_thresholds: Dict[int, float] = {}
        # Conformal non-conformity quantiles q_hat
        self.calibrated_q_hat: Dict[int, float] = {}
        self.dev_stats: Dict[int, Dict[str, Any]] = {}
        self.is_calibrated = False

    def compute_confidence(self, signals: Dict[str, float], level: int) -> float:
        """
        Computes composite confidence score C in [0, 1] for level L.
        Higher score implies higher reliability / lower error risk.
        """
        s1 = float(signals.get("S1", 0.33))
        s2 = float(signals.get("S2", 0.0))
        s3 = float(signals.get("S3", 0.0))
        s4 = float(signals.get("S4", 0.0))

        if level == 0:
            return s1
        elif level == 1:
            return min(s2, s3)
        elif level == 2:
            return min(s3, s4)
        else:
            return 1.0

    def compute_nonconformity(self, signals: Dict[str, float], level: int) -> float:
        """
        Computes nonconformity score R = 1 - Confidence in [0, 1].
        Higher R implies higher uncertainty.
        """
        return round(1.0 - self.compute_confidence(signals, level), 6)

    def calibrate_on_dev(
        self,
        dev_records: Union[List[Dict[str, Any]], Dict[str, Dict[int, Dict[str, Any]]]],
        split_name: str = "dev"
    ):
        """
        Calibrate risk-controlled early-exit threshold tau_alpha per level on dev split data.
        Selects the minimum confidence threshold tau (maximizing coverage) that guarantees:
            Empirical Error Rate(Exit | Confidence >= tau) <= alpha.
        """
        assert split_name.lower() == "dev", (
            f"DATA LEAKAGE VIOLATION: Conformal thresholds MUST ONLY be calibrated on 'dev' split! "
            f"Received: '{split_name}'."
        )

        flat_records: List[Dict[str, Any]] = []
        if isinstance(dev_records, dict):
            for qid, by_lvl in dev_records.items():
                for lvl_idx, data in by_lvl.items():
                    flat_records.append({
                        "level": int(lvl_idx),
                        "signals": data.get("signals", {}),
                        "f1": float(data.get("f1", 0.0)),
                        "em": float(data.get("em", 0.0))
                    })
        else:
            flat_records = dev_records

        by_level_data: Dict[int, List[Tuple[float, int]]] = {0: [], 1: [], 2: []}
        for rec in flat_records:
            lvl = rec.get("level", 0)
            if lvl in by_level_data:
                conf = self.compute_confidence(rec.get("signals", {}), lvl)
                is_error = 1 if rec.get("f1", 0.0) < self.f1_threshold else 0
                by_level_data[lvl].append((conf, is_error))

        for lvl in range(3):
            pairs = by_level_data[lvl]
            if not pairs:
                self.calibrated_thresholds[lvl] = 1.0
                self.calibrated_q_hat[lvl] = 0.0
                continue

            scores = [p[0] for p in pairs]
            errors = [p[1] for p in pairs]
            unique_taus = sorted(list(set(scores)), reverse=True)

            valid_thresholds = []
            for tau in unique_taus:
                exited_errors = [e for s, e in zip(scores, errors) if s >= tau]
                if not exited_errors:
                    continue
                err_rate = sum(exited_errors) / len(exited_errors)
                if err_rate <= self.target_alpha:
                    cov = len(exited_errors) / len(scores)
                    valid_thresholds.append((tau, err_rate, cov))

            if valid_thresholds:
                # Pick lowest tau satisfying risk constraint to maximize coverage
                best_tau, best_err, best_cov = valid_thresholds[-1]
                self.calibrated_thresholds[lvl] = round(best_tau, 4)
                self.calibrated_q_hat[lvl] = round(1.0 - best_tau, 4)
                self.dev_stats[lvl] = {
                    "tau": round(best_tau, 4),
                    "dev_coverage": round(best_cov * 100, 2),
                    "dev_error_rate": round(best_err * 100, 2)
                }
            else:
                # Conservative fallback if dev error cannot be bounded at target alpha
                self.calibrated_thresholds[lvl] = 1.0
                self.calibrated_q_hat[lvl] = 0.0
                self.dev_stats[lvl] = {
                    "tau": 1.0,
                    "dev_coverage": 0.0,
                    "dev_error_rate": 0.0,
                    "note": f"No threshold satisfied alpha <= {self.target_alpha} on dev"
                }

        self.is_calibrated = True

    def should_exit_conformal(
        self,
        signals: Dict[str, float],
        level: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determines whether early exit is allowed at level L under risk control.
        Exit is allowed iff Confidence >= tau_alpha (or nonconformity R <= q_hat).
        """
        conf = self.compute_confidence(signals, level)
        r = round(1.0 - conf, 6)
        tau = self.calibrated_thresholds.get(level, 1.0)
        q_hat = self.calibrated_q_hat.get(level, 0.0)

        exit_allowed = (conf >= tau) or (r <= q_hat + 1e-6)

        return exit_allowed, {
            "level": level,
            "confidence": round(conf, 4),
            "nonconformity": round(r, 4),
            "threshold_tau": round(tau, 4),
            "threshold_q_hat": round(q_hat, 4),
            "target_alpha": self.target_alpha,
            "early_exit": exit_allowed
        }

    def evaluate_coverage_and_risk(
        self,
        test_records: Union[List[Dict[str, Any]], Dict[str, Dict[int, Dict[str, Any]]]],
        level: int = 0
    ) -> Dict[str, Any]:
        """
        Evaluates early-exit coverage (% exited) and empirical test error rate
        for a specific level on test split records.
        """
        flat_records: List[Dict[str, Any]] = []
        if isinstance(test_records, dict):
            for qid, by_lvl in test_records.items():
                if level in by_lvl:
                    data = by_lvl[level]
                    flat_records.append({
                        "signals": data.get("signals", {}),
                        "f1": float(data.get("f1", 0.0)),
                        "em": float(data.get("em", 0.0))
                    })
        else:
            flat_records = [r for r in test_records if r.get("level", 0) == level]

        n_total = len(flat_records)
        if n_total == 0:
            return {
                "level": level,
                "target_alpha": self.target_alpha,
                "coverage_pct": 0.0,
                "empirical_error_rate_pct": 0.0,
                "test_alpha_satisfied": True,
                "exited_queries": 0,
                "total_queries": 0
            }

        exited_count = 0
        error_count = 0

        for rec in flat_records:
            signals = rec.get("signals", {})
            can_exit, _ = self.should_exit_conformal(signals, level)
            if can_exit:
                exited_count += 1
                if rec.get("f1", 0.0) < self.f1_threshold:
                    error_count += 1

        coverage = (exited_count / n_total) * 100.0
        err_rate = (error_count / exited_count) * 100.0 if exited_count > 0 else 0.0
        satisfied = bool((err_rate / 100.0) <= self.target_alpha)

        return {
            "level": level,
            "target_alpha": self.target_alpha,
            "threshold_tau": self.calibrated_thresholds.get(level, 1.0),
            "coverage_pct": round(coverage, 2),
            "empirical_error_rate_pct": round(err_rate, 2),
            "test_alpha_satisfied": satisfied,
            "exited_queries": exited_count,
            "error_queries_in_exit": error_count,
            "total_queries": n_total
        }

    @classmethod
    def calibrate_and_sweep_alphas(
        cls,
        dev_records: Union[List[Dict[str, Any]], Dict[str, Dict[int, Dict[str, Any]]]],
        test_records: Union[List[Dict[str, Any]], Dict[str, Dict[int, Dict[str, Any]]]],
        alphas: Optional[List[float]] = None,
        level: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Sweeps target risk alpha across [0.05, 0.10, 0.15], calibrating on dev
        and reporting test coverage and empirical error rate.
        """
        if alphas is None:
            alphas = [0.05, 0.10, 0.15]

        results = []
        for a in alphas:
            ctrl = cls(target_alpha=a)
            ctrl.calibrate_on_dev(dev_records, split_name="dev")
            eval_res = ctrl.evaluate_coverage_and_risk(test_records, level=level)
            results.append(eval_res)
        return results
