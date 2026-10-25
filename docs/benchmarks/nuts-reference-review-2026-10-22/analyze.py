"""Recompute descriptive retained-work ratios from the existing M16 cohort."""

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def analyze(root):
    bundle = root / "docs/benchmarks/m16/posteriordb-complete-v3"
    grouped = defaultdict(lambda: {"calls": 0, "seconds": 0.0, "chains": set()})
    with (bundle / "phase-costs.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            if (
                row["phase"] != "retained"
                or row["status"] != "completed"
                or row["method"] not in {"nuts_rs", "nuts_cov_diag", "nuts_fisher_diag", "nuts_fisher_rank4"}
            ):
                continue
            key = row["case"], row["method"], int(row["repetition"])
            value = grouped[key]
            assert row["chain"] not in value["chains"], key
            value["chains"].add(row["chain"])
            value["calls"] += int(row["gradient_api_calls"])
            value["seconds"] += float(row["seconds"])

    with (bundle / "summary.csv").open(newline="") as source:
        summaries = {(row["case"], row["method"]): row for row in csv.DictReader(source)}

    output = {"cohort": str(bundle.relative_to(root)), "baseline": "nuts_rs", "methods": {}}
    ess_field = "median_minimum_bulk_ess_per_retained_gradient_call"
    for method in ("nuts_cov_diag", "nuts_fisher_diag", "nuts_fisher_rank4"):
        cases = []
        for case, current_method in sorted(summaries):
            if current_method != method:
                continue
            rep_ratios = []
            cost_ratios = []
            for repetition in range(3):
                value = grouped[case, method, repetition]
                baseline = grouped[case, "nuts_rs", repetition]
                assert len(value["chains"]) == len(baseline["chains"]) == 4
                rep_ratios.append(value["calls"] / baseline["calls"])
                cost_ratios.append(
                    (value["seconds"] / value["calls"])
                    / (baseline["seconds"] / baseline["calls"])
                )
            cases.append({
                "case": case,
                "retained_call_ratios": rep_ratios,
                "median_retained_call_ratio": median(rep_ratios),
                "retained_seconds_per_call_ratios": cost_ratios,
                "median_retained_seconds_per_call_ratio": median(cost_ratios),
                "retained_ess_per_call_ratio": (
                    float(summaries[case, method][ess_field])
                    / float(summaries[case, "nuts_rs"][ess_field])
                ),
            })
        assert len(cases) == 20
        output["methods"][method] = {
            "cases": cases,
            "median_retained_call_ratio": median(row["median_retained_call_ratio"] for row in cases),
            "median_retained_seconds_per_call_ratio": median(
                row["median_retained_seconds_per_call_ratio"] for row in cases
            ),
            "median_retained_ess_per_call_ratio": median(row["retained_ess_per_call_ratio"] for row in cases),
        }
    output["limits"] = (
        "Existing 20-target cohort, all targets included without a quality filter. "
        "Three repeated runs of declared seed sets. Policies differ. Ratios of "
        "summary ESS medians are not medians of paired ESS ratios. Seconds per "
        "call includes all retained sampler work and the shared derivative backend. "
        "This is descriptive evidence, not a default-selection rule or a new timing run."
    )
    return output


if __name__ == "__main__":
    import sys

    root = Path(__file__).resolve().parents[3]
    result = analyze(root)
    if len(sys.argv) == 2:
        Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
    else:
        print(json.dumps(result, indent=2))
