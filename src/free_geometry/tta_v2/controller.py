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

    Selection semantics (select): candidates are ranked by their improvement
    score — the mean of -rel_change over every participating (record,
    component) pair, where the rel_change denominator carries the abs_floor_rel
    per-component noise floor (see below). A candidate is VETOED only by:
      - schema/record-set mismatch vs the step-0 baseline,
      - a component that was valid at step 0 turning None/non-finite
        (validity chain), or
      - a component whose rel_change exceeds the veto threshold
        (catastrophic safety net, see below).
    Fallback to the baseline happens ONLY when the best non-vetoed
    candidate's improvement is < min_delta.

    catastrophic_rel: the safety-net veto threshold on rel_change (with the
               abs_floor envelope denominator). Default 0.5 = a 50% REAL
               degradation on any single (record, component) vetoes the
               candidate; mild degradations merely lower its rank. Set to
               None to restore the legacy tau_qual rule (veto when ANY
               component exceeds tau_qual = 0.05) for regression comparison.
    tau_qual:  legacy veto threshold, used ONLY when catastrophic_rel is None.
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
    # Per-component ABSOLUTE noise floor, in units of the component's own
    # step-0 typical scale (mean of |V0| over all (pair, mask) records). The
    # rel_change denominator is widened to
    #     max(|v0|, eps_abs, abs_floor_rel * typical_k / tau_qual)
    # so the effective per-component tolerance is the WIDER of
    #     tau_qual * |v0|   (relative envelope)
    #     abs_floor_rel * typical_k   (absolute noise floor)
    # This stops near-zero-baseline records of a small-scale component
    # (couple ~1e-3 on one probe record while the component's typical scale is
    # ~5e-2) from turning noise-level wiggles into +100..+6000% veto votes.
    # 0.0 disables the floor (denominator max(|v0|, eps_abs)).
    abs_floor_rel: float = 0.2
    # Safety-net veto threshold on rel_change; None restores the legacy
    # tau_qual qualification rule. See the class docstring.
    catastrophic_rel: Optional[float] = 0.5


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


def _component_scales(base_idx: Dict[Tuple[str, int], Dict[str, float]]
                      ) -> Dict[str, float]:
    """Typical absolute scale per component: the mean of |V0| over all step-0
    (pair, mask) records with finite values. Drives the abs_floor_rel
    per-component noise floor (components with no finite step-0 record get no
    floor and fall back to the eps_abs rule)."""
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for comps in base_idx.values():
        for name, v in comps.items():
            fv = _finite_float(v)
            if fv is None:
                continue
            sums[name] = sums.get(name, 0.0) + abs(fv)
            counts[name] = counts.get(name, 0) + 1
    return {n: sums[n] / counts[n] for n in sums}


def _score_step(entry_t: Dict, base_idx: Dict[Tuple[str, int], Dict[str, float]],
                participating: Set[Tuple[Tuple[str, int], str]],
                cfg: ControllerConfig,
                scales: Dict[str, float]) -> Tuple[Optional[float], Optional[str]]:
    """Improvement score of one step vs the step-0 baseline: the mean of
    -rel_change over every participating (record, component) pair.

    rel_change denominator per (record, component): the wider of the relative
    envelope and the component's absolute noise floor —
        denom = max(|v0|, eps_abs, abs_floor_rel * typical_k / tau_qual)
    (see ControllerConfig.abs_floor_rel).

    Veto rule (returns reason, improvement None):
      - record-set / component-schema mismatch vs the baseline,
      - a participating component turning None / non-finite at this step,
      - rel_change above the veto threshold: catastrophic_rel when set (the
        safety net), tau_qual when catastrophic_rel is None (legacy rule).

    Returns (improvement, reason); a step only wins selection when reason is
    None."""
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
    use_floor = cfg.abs_floor_rel > 0.0 and cfg.tau_qual > 0.0
    legacy = cfg.catastrophic_rel is None
    threshold = cfg.tau_qual if legacy else float(cfg.catastrophic_rel)
    threshold_name = "tau_qual" if legacy else "catastrophic_rel"
    terms: List[float] = []
    for key, name in sorted(participating):
        v = idx_t[key][name]
        fv = _finite_float(v)
        if fv is None:
            return None, (f"{key[0]}/{key[1]}/{name}: component invalid or "
                          f"non-finite at candidate step")
        fb = _finite_float(base_idx[key][name])  # finite by construction of P
        denom = max(abs(fb), cfg.eps_abs)
        if use_floor:
            denom = max(denom, cfg.abs_floor_rel * scales.get(name, 0.0) / cfg.tau_qual)
        rc = (fv - fb) / denom
        if rc > threshold:
            return None, (f"{key[0]}/{key[1]}/{name} rel_change={rc:+.4f}"
                          f" > {threshold_name}={threshold}")
        terms.append(-rc)
    return float(sum(terms) / len(terms)), None


def select(trace: List[Dict], cfg: ControllerConfig) -> Dict:
    """Pick the checkpoint step from a probe trace.

    trace: list of evaluate() results in any order; step 0 MUST be present as
    the baseline. Every candidate step t (including 0) is scored per
    participating (pair_id, mask_id, component) against its step-0
    counterpart. Main rule: the non-vetoed candidate with the highest
    improvement score wins (ties -> earliest step). Safety net: a candidate is
    vetoed when any participating component's rel_change (abs_floor envelope
    denominator) exceeds cfg.catastrophic_rel — or, when catastrophic_rel is
    None, the legacy rule vetoes at cfg.tau_qual — and always on schema
    mismatch or a component turning invalid at the candidate step. Fallback to
    step 0 happens ONLY when the best non-vetoed improvement is <
    cfg.min_delta (or no non-zero step survived).

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
    scales = _component_scales(base_idx)
    disqualified: Dict[int, str] = {}
    best_imp, best_step = 0.0, 0
    for entry in ordered:
        step = int(entry["step"])
        imp, reason = _score_step(entry, base_idx, participating, cfg, scales)
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
    scales = _component_scales(base_idx)
    best: Optional[float] = None
    stale = 0
    for entry in ordered:
        step = int(entry["step"])
        imp, reason = _score_step(entry, base_idx, participating, cfg, scales)
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
