#!/usr/bin/env python3
"""Paired acceptance rates and bootstrap CIs for strict Co-visibility results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta, fisher_exact

HIGHER_IS_BETTER = {"auc03", "fscore", "auc30"}
METRICS = ("auc03", "fscore", "auc30", "overall")
COMPARISONS = {
    "base_8v_vs_base_4v": ("base_8v", "base_4v"),
    "base_8v_extract_4v_vs_base_4v": ("base_8v_extract_4v", "base_4v"),
    "lora_4v_vs_base_4v": ("lora_4v", "base_4v"),
    "lora_8v_vs_base_8v": ("lora_8v", "base_8v"),
}


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = z * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [float(center - half_width), float(center + half_width)]


def clopper_pearson(successes: int, total: int, alpha: float = 0.05) -> list[float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return [lower, upper]


def load_cases(root: Path, datasets: list[str]) -> dict[str, dict]:
    cases = {}
    for dataset in datasets:
        payload = json.loads((root / dataset / "window_metrics.json").read_text(encoding="utf-8"))[dataset]
        for scene in sorted(payload["base_4v"]):
            case_id = f"{dataset}/{scene}"
            cases[case_id] = {arm: values[scene][0] for arm, values in payload.items()}
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/covisibility_010_strict_top5_vggt")
    parser.add_argument("--datasets", nargs="+", default=["eth3d", "scannetpp"])
    parser.add_argument("--bootstrap-samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--output", default="results/covisibility_010_strict_top5_vggt/paired_statistics.json")
    args = parser.parse_args()

    cases = load_cases(Path(args.root), args.datasets)
    rng = np.random.default_rng(args.seed)
    output = {"unit": "one fixed 8V sequence from one scene", "case_ids": sorted(cases), "comparisons": {}}
    for name, (candidate, reference) in COMPARISONS.items():
        report = {}
        for metric in METRICS:
            sign = 1.0 if metric in HIGHER_IS_BETTER else -1.0
            raw_delta = np.asarray([cases[case][candidate][metric] - cases[case][reference][metric] for case in output["case_ids"]])
            favorable = sign * raw_delta
            ties = np.isclose(favorable, 0.0, rtol=0.0, atol=1e-12)
            accepted = favorable >= -1e-12  # Tie or strictly better.
            bootstrap = favorable[rng.integers(0, len(favorable), (args.bootstrap_samples, len(favorable)))].mean(axis=1)
            report[metric] = {
                "accepted_count": int(accepted.sum()),
                "strictly_better_count": int((favorable > 1e-12).sum()),
                "tie_count": int(ties.sum()),
                "total_cases": len(favorable),
                "acceptance_rate": float(accepted.mean()),
                "acceptance_rate_wilson_95ci": wilson(int(accepted.sum()), len(favorable)),
                "mean_raw_delta_candidate_minus_reference": float(raw_delta.mean()),
                "mean_favorable_delta": float(favorable.mean()),
                "mean_favorable_delta_bootstrap_95ci": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
            }
        output["comparisons"][name] = report
    teacher_bad = []
    student_bad = []
    for case in output["case_ids"]:
        arms = cases[case]
        teacher_bad.append(
            arms["base_8v_extract_4v"]["fscore"] < arms["base_4v"]["fscore"]
            and arms["base_8v_extract_4v"]["overall"] > arms["base_4v"]["overall"]
        )
        student_bad.append(
            arms["lora_4v"]["fscore"] < arms["base_4v"]["fscore"]
            and arms["lora_4v"]["overall"] > arms["base_4v"]["overall"]
        )
    both = sum(t and s for t, s in zip(teacher_bad, student_bad))
    teacher_count = sum(teacher_bad)
    teacher_good_student_bad = sum((not t) and s for t, s in zip(teacher_bad, student_bad))
    neither = sum((not t) and (not s) for t, s in zip(teacher_bad, student_bad))
    output["conditional_bad_teacher_analysis"] = {
        "definition": "bad means strictly lower F1 and strictly higher CD/overall than Base 4V on the same sequence",
        "teacher_bad_probability": teacher_count / len(teacher_bad),
        "teacher_bad_probability_exact_95ci": clopper_pearson(teacher_count, len(teacher_bad)),
        "student_bad_given_teacher_bad_probability": both / teacher_count,
        "student_bad_given_teacher_bad_exact_95ci": clopper_pearson(both, teacher_count),
        "student_bad_given_teacher_not_bad_probability": teacher_good_student_bad / (len(teacher_bad) - teacher_count),
        "contingency_teacher_rows_student_columns_bad_good": [[both, teacher_count - both], [teacher_good_student_bad, neither]],
        "fisher_exact_two_sided_p": float(fisher_exact([[both, teacher_count - both], [teacher_good_student_bad, neither]]).pvalue),
    }
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
