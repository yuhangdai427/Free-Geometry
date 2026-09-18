"""Checkpoint selection / fallback controller for protocol v2.

Pure functions over probe trace records (the evaluate() output of
ProbeEvaluator) — no torch, no model state, so a finished run can be replayed
offline and the exact same checkpoint selection reproduced.

Validity semantics (paired with ProbeEvaluator's record schema):
  - a component whose step-0 value is None or non-finite is PERMANENTLY
    unavailable and never participates in scoring (at any step);
  - a component that is valid at step 0 but None or non-finite at a candidate
    step DISQUALIFIES that candidate — an invalidated metric is never
    rewarded as an improvement;
  - a candidate whose record set or per-record component names deviate from
    the step-0 schema is DISQUALIFIED (schema validation).
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class ControllerConfig:
    """Probe-trace selection knobs.

    tau_qual:  per-component relative degradation ceiling — a candidate step
               is disqualified if ANY participating (probe, mask, component)
               rel_change vs the step-0 baseline exceeds this.
    eps_abs:   floor on |v_0| in the rel_change denominator, so components
               with a near-zero baseline (e.g. a null couple loss) report
               absolute change instead of a blow-up ratio.
    min_delta: minimum mean relative improvement for a non-zero step to be
               picked over the baseline (and the improvement margin tracked
               by should_stop's best+patience logic).
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


def _finite_float(v) -> Optional[float]:
    """v coerced to a finite float, else None (None / bool / nan / inf / non-numeric)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _participating(base_idx: Dict[Tuple[str, int], Dict[str, float]]
                   ) -> Set[Tuple[Tuple[str, int], str]]:
    """(record key, component) pairs whose step-0 value is present and finite —
    the permanently-available metric set."""
    p: Set[Tuple[Tuple[str, int], str]] = set()
    for key, comps in base_idx.items():
        for name, v in comps.items():
            if _finite_float(v) is not None:
                p.add((key, name))
    return p


def _score_step(entry_t: Dict, base_idx: Dict[Tuple[str, int], Dict[str, float]],
                participating: Set[Tuple[Tuple[str, int], str]],
                cfg: ControllerConfig) -> Tuple[Optional[float], Optional[str]]:
    """Improvement score of one step vs the step-0 baseline: the mean of
    -rel_change over every participating (record, component) pair.

    Returns (improvement, reason). improvement is None when the step is
    disqualified or unscoreable (reason set); a step only wins selection when
    reason is None."""
    idx_t = _index_step(entry_t)
    if set(idx_t) != set(base_idx):
        missing = sorted(set(base_idx) - set(idx_t))
        extra = sorted(set(idx_t) - set(base_idx))
        return None, f"record set mismatch vs baseline: missing={missing} extra={extra}"
    for key in sorted(idx_t):
        if set(idx_t[key]) != set(base_idx[key]):
            return None, (f"{key[0]}/{key[1]}: component set mismatch vs baseline: "
                          f"missing={sorted(set(base_idx[key]) - set(idx_t[key]))} "
                          f"extra={sorted(set(idx_t[key]) - set(base_idx[key]))}")
    if not participating:
        return None, "no comparable components"
    terms: List[float] = []
    for key, name in sorted(participating):
        v = idx_t[key][name]
        fv = _finite_float(v)
        if fv is None:
            return None, (f"{key[0]}/{key[1]}/{name}: component invalid or "
                          f"non-finite at candidate step")
        fb = _finite_float(base_idx[key][name])  # finite by construction of P
        rc = rel_change(fv, fb, cfg.eps_abs)
        if rc > cfg.tau_qual:
            return None, (f"{key[0]}/{key[1]}/{name} rel_change={rc:+.4f}"
                          f" > tau_qual={cfg.tau_qual}")
        terms.append(-rc)
    return float(sum(terms) / len(terms)), None


def select(trace: List[Dict], cfg: ControllerConfig) -> Dict:
    """Pick the checkpoint step from a probe trace.

    trace: list of evaluate() results in any order; step 0 MUST be present as
    the baseline. Every candidate step t (including 0) is scored per
    participating (pair_id, mask_id, component) against its step-0
    counterpart; t is qualified when its schema matches the baseline, all
    participating components are finite at t, and all rel_change <=
    cfg.tau_qual. The selected step is the qualified one with the highest
    improvement score (ties -> earliest step). Falls back to step 0 when the
    best improvement is < cfg.min_delta or when no non-zero step qualified.

    Returns {"selected_step", "fell_back_to_baseline", "improvement" (score of
    the selected step; 0.0 on fallback), "n_comparable" (number of
    participating (record, component) pairs), "disqualified": {step: reason}}.
    The per-record "total" field is deliberately ignored (raw scale-mixed sum,
    inspection only).
    """
    ordered = sorted(trace, key=lambda e: int(e["step"]))
    if not ordered or int(ordered[0]["step"]) != 0:
        raise ValueError("trace must contain the step-0 (baseline) entry")
    base_idx = _index_step(ordered[0])
    participating = _participating(base_idx)
    disqualified: Dict[int, str] = {}
    best_imp, best_step = 0.0, 0
    for entry in ordered:
        step = int(entry["step"])
        imp, reason = _score_step(entry, base_idx, participating, cfg)
        if reason is not None:
            disqualified[step] = reason
            continue
        if imp > best_imp:
            best_imp, best_step = imp, step
    fell_back = best_step == 0 or best_imp < cfg.min_delta
    return {"selected_step": 0 if fell_back else best_step,
            "fell_back_to_baseline": fell_back,
            "improvement": 0.0 if fell_back else best_imp,
            "n_comparable": len(participating),
            "disqualified": disqualified}


def should_stop(trace: List[Dict], cfg: ControllerConfig) -> bool:
    """Live early-stopping guard (disabled unless the caller wires it in):
    best+patience plateau detection over the improvement score. A running
    best_score/best_step is maintained across the trace; each scoreable entry
    that fails to improve the best score by >= cfg.min_delta increments the
    stale counter (a new best resets it; disqualified/unscoreable entries are
    skipped and neither improve nor count). True once the stale counter
    reaches cfg.patience at an entry with step >= cfg.min_step. Always False
    with fewer than patience+1 trace entries.
    """
    ordered = sorted(trace, key=lambda e: int(e["step"]))
    if len(ordered) < cfg.patience + 1:
        return False
    base_idx = _index_step(ordered[0])
    participating = _participating(base_idx)
    best: Optional[float] = None
    stale = 0
    for entry in ordered:
        step = int(entry["step"])
        imp, reason = _score_step(entry, base_idx, participating, cfg)
        if reason is not None:
            continue
        improved = best is None or imp > best + cfg.min_delta
        if improved:
            best = imp
            stale = 0
        else:
            stale += 1
        if step >= cfg.min_step and stale >= cfg.patience:
            return True
    return False
