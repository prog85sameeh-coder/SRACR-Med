from dataclasses import dataclass
import numpy as np


@dataclass
class PolicyDecision:
    global_risk: float
    target_level: int
    controller_level: int
    policy: str
    target_coverage: float
    key_rotation: bool
    components: dict


class AdaptiveCoverageController:
    """Stateful controller separating WHERE from HOW MUCH.

    WHERE: reliability-adjusted saliency ranks tiles.
    HOW MUCH: global risk chooses the coverage/policy level.
    """

    NAMES = {1: "BASELINE", 2: "RISK_PRIORITIZED", 3: "HIGH_RISK", 4: "CRITICAL"}

    def __init__(self, thresholds, hysteresis, coverage_by_level):
        self.thresholds = [float(x) for x in thresholds]
        if len(self.thresholds) != 3 or not (self.thresholds[0] < self.thresholds[1] < self.thresholds[2]):
            raise ValueError("controller thresholds must contain 3 ascending values")
        self.h = float(hysteresis)
        self.coverage = {int(k): float(v) for k, v in coverage_by_level.items()}
        for level in (1, 2, 3, 4):
            if level not in self.coverage:
                raise ValueError("coverage_by_level must define levels 1..4")
        self.level = 1

    def desired_level(self, risk, threat):
        risk = float(risk)
        threat = float(threat)
        if threat >= 0.90 or risk >= self.thresholds[2]:
            return 4
        if risk >= self.thresholds[1]:
            return 3
        if risk >= self.thresholds[0]:
            return 2
        return 1

    def update(self, global_risk, threat):
        desired = self.desired_level(global_risk, threat)
        old = self.level
        if desired > old:
            self.level = desired
        elif desired < old:
            # Stepwise de-escalation with hysteresis prevents oscillation.
            boundary_idx = max(0, old - 2)
            boundary = self.thresholds[boundary_idx] if old > 1 else 0.0
            if float(global_risk) < boundary - self.h and float(threat) < 0.90:
                self.level = max(desired, old - 1)
        rotate = self.level == 4 or float(threat) >= 0.90
        return old, desired, self.level, self.coverage[self.level], rotate


def saliency_burden(reliable_saliency):
    s = np.asarray(reliable_saliency, dtype=np.float32)
    if s.size == 0:
        return 0.0
    mean = float(s.mean())
    p95 = float(np.percentile(s, 95))
    return float(np.clip(0.5 * mean + 0.5 * p95, 0.0, 1.0))


def global_risk_score(reliable_saliency, clinical, threat, uncertainty, weights):
    a = saliency_burden(reliable_saliency)
    c = float(np.clip(clinical, 0, 1))
    t = float(np.clip(threat, 0, 1))
    u = float(np.clip(uncertainty, 0, 1))
    w = np.asarray([
        float(weights.get("saliency", 0.30)),
        float(weights.get("clinical", 0.25)),
        float(weights.get("threat", 0.30)),
        float(weights.get("uncertainty", 0.15)),
    ], dtype=np.float64)
    if np.any(w < 0) or not np.isclose(w.sum(), 1.0):
        raise ValueError("global-risk weights must be non-negative and sum to 1.0")
    g = float(np.clip(np.dot(w, np.asarray([a, c, t, u], dtype=np.float64)), 0, 1))
    return g, {"saliency_burden": a, "clinical": c, "threat": t, "uncertainty": u}
