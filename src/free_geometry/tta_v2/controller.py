"""Checkpoint selection / fallback controller for protocol v2.

Pure functions over probe trace records (the evaluate() output of
ProbeEvaluator) — no torch, no model state, so a finished run can be replayed
offline and the exact same checkpoint selection reproduced.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ControllerConfig:
    """Probe-trace selection knobs.

    tau_qual:  per-component relative degradation ceiling — a candidate step
               is disqualified if ANY (probe, mask, component) rel_change vs
               the step-0 baseline exceeds this.
    eps_abs:   floor on |v_0| in the rel_change denominator, so components
               with a near-zero baseline (e.g. a null couple loss) report
               absolute change instead of a blow-up ratio.
    min_delta: minimum mean relative improvement for a non-zero step to be
               picked over the baseline.
    patience / min_step: live early-stopping guard for should_stop.
    """

    tau_qual: float = 0.05
    eps_abs: float = 1e-3
    min_delta: float = 0.005
    patience: int = 3
    min_step: int = 30


def rel_change(v_t: float, v_0: float, eps_abs: float) -> float:
    """Relative change (v_t - v_0) / max(|v_0|, eps_abs)."""
    return (v_t - v_0) / max(abs(v_0), eps_abs)


def _index_step(entry: Dict) -> Dict[Tuple[str, int], Dict[str, float]]:
    return {(r["pair_id"], r["mask_id"]): r["components"] for r in entry["records"]}


def _score_step(entry_t: Dict, base_idx: Dict[Tuple[str, int], Dict[str, float]],
                cfg: ControllerConfig) -> Tuple[float, Optional[str]]:
    """Improvement score of one step vs the step-0 baseline: the mean of
    -rel_change over every (record, component) pair. Returns (improvement,
    disqualification reason or None). Improvement is still computed when
    disqualified (callers may log it); a step only wins selection when the
    reason is None."""
    idx_t = _index_step(entry_t)
    if set(idx_t) != set(base_idx):
        missing = sorted(set(base_idx) - set(idx_t))
        extra = sorted(set(idx_t) - set(base_idx))
        return 0.0, f"record set mismatch vs baseline: missing={missing} extra={extra}"
    terms: List[float] = []
    reason: Optional[str] = None
    for key in sorted(idx_t):
        comp_t, comp_b = idx_t[key], base_idx[key]
        for name in sorted(comp_t):
            if name not in comp_b:
                reason = reason or f"{key[0]}/{key[1]}: no baseline component {name}"
                continue
            rc = rel_change(float(comp_t[name]), float(comp_b[name]), cfg.eps_abs)
            if rc > cfg.tau_qual and reason is None:
                reason = (f"{key[0]}/{key[1]}/{name} rel_change={rc:+.4f}"
                          f" > tau_qual={cfg.tau_qual}")
            terms.append(-rc)
    if not terms:
        reason = reason or "no comparable components"
    improvement = float(sum(terms) / len(terms)) if terms else 0.0
    return improvement, reason


def select(trace: List[Dict], cfg: ControllerConfig) -> Dict:
    """Pick the checkpoint step from a probe trace.

    trace: list of evaluate() results in any order; step 0 MUST be present as
    the baseline. Every candidate step t (including 0) is scored per
    (pair_id, mask_id, component) against its step-0 counterpart; t is
    qualified when all rel_change <= cfg.tau_qual. The selected step is the
    qualified one with the highest improvement score (ties -> earliest step).
    Falls back to step 0 when the best improvement is < cfg.min_delta or when
    no non-zero step qualified.

    Returns {"selected_step", "fell_back_to_baseline", "improvement" (score of
    the selected step; 0.0 on fallback), "disqualified": {step: reason}}.
    The per-record "total" field is deliberately ignored (raw scale-mixed sum,
    inspection only).
    """
    ordered = sorted(trace, key=lambda e: int(e["step"]))
    if not ordered or int(ordered[0]["step"]) != 0:
        raise ValueError("trace must contain the step-0 (baseline) entry")
    base_idx = _index_step(ordered[0])
    disqualified: Dict[int, str] = {}
    best_imp, best_step = 0.0, 0
    for entry in ordered:
        step = int(entry["step"])
        imp, reason = _score_step(entry, base_idx, cfg)
        if reason is not None:
            disqualified[step] = reason
            continue
        if imp > best_imp:
            best_imp, best_step = imp, step
    fell_back = best_step == 0 or best_imp < cfg.min_delta
    return {"selected_step": 0 if fell_back else best_step,
            "fell_back_to_baseline": fell_back,
            "improvement": 0.0 if fell_back else best_imp,
            "disqualified": disqualified}


def should_stop(trace: List[Dict], cfg: ControllerConfig) -> bool:
    """Live early-stopping guard (disabled unless the caller wires it in):
    True when the improvement score of each of the last cfg.patience
    checkpoints is < cfg.min_delta and the latest step has reached
    cfg.min_step. The score here ignores tau qualification (plateau detection
    only); it is always False with fewer than patience+1 trace entries.
    """
    ordered = sorted(trace, key=lambda e: int(e["step"]))
    if len(ordered) < cfg.patience + 1:
        return False
    if int(ordered[-1]["step"]) < cfg.min_step:
        return False
    base_idx = _index_step(ordered[0])
    for entry in ordered[-cfg.patience:]:
        imp, _ = _score_step(entry, base_idx, cfg)
        if imp >= cfg.min_delta:
            return False
    return True
