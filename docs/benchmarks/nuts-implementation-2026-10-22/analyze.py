"""Summarize all native ablation outcomes and bind diagnostics to raw draws."""
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path


def observations(case, q0, q1):
    if case == "correlated":
        x, y = q0, (q1 - 0.6 * q0) / 0.8
    elif case == "banana":
        x, y = q0, q1 - 0.4 * (q0 * q0 - 1.0)
    elif case == "funnel":
        x, y = q0, q1 * math.exp(-0.5 * q0)
    elif case == "rotated":
        x, y = (0.8 * q0 + 0.6 * q1) / 0.01, -0.6 * q0 + 0.8 * q1
    else:
        assert case == "logistic"
        x, y = q0, q1
    return x, y, x * x, y * y, x * y


def analyze(directory):
    protocol = json.loads((directory / "protocol.json").read_text())
    path = directory / "adaptation"
    work = list(csv.DictReader((path / "work.csv").open()))
    diagnostics = list(csv.DictReader((path / "diagnostics.csv").open()))
    raw = path / "draws.csv"
    source = raw.open() if raw.exists() else gzip.open(path / "draws.csv.gz", "rt")
    counts = defaultdict(int)
    values = defaultdict(lambda: [[] for _ in range(5)])
    with source:
        for row in csv.DictReader(source):
            key = row["case"], row["method"]
            chain = *key, int(row["chain"])
            assert int(row["index"]) == counts[chain], chain
            counts[chain] += 1
            q0, q1 = float(row["q0"]), float(row["q1"])
            assert math.isfinite(q0) and math.isfinite(q1)
            for column, value in zip(values[key], observations(key[0], q0, q1)):
                column.append(value)
    groups = []
    for case in protocol["adaptation"]["targets"]:
        for method in protocol["adaptation"]["methods"]:
            rows = [r for r in work if r["case"] == case and r["method"] == method]
            assert len(rows) == 4
            complete = [r for r in rows if r["status"] == "completed"]
            summary = {"case": case, "method": method, "completed_chains": len(complete),
                       "failed_chains": [r for r in rows if r["status"] != "completed"]}
            for r in complete:
                assert counts[case, method, int(r["chain"])] == 2048
            ds = sorted([r for r in diagnostics if r["case"] == case and r["method"] == method], key=lambda r: int(r["observable"]))
            assert len(ds) == 5
            summary["diagnostics"] = ds
            if len(complete) == 4 and all(d["status"] == "defined" for d in ds):
                means = [math.fsum(column) / len(column) for column in values[case, method]]
                for value, d in zip(means, ds):
                    assert abs(value - float(d["mean"])) <= 2e-12 * max(1.0, abs(value))
                calls = sum(int(r["retained_calls"]) for r in complete)
                retained_ns = sum(int(r["retained_ns"]) for r in complete)
                ess = min(float(d["bulk_ess"]) for d in ds)
                summary.update({
                    "retained_calls": calls, "warmup_calls": sum(int(r["warm_calls"]) for r in complete),
                    "retained_divergences": sum(int(r["divergences"]) for r in complete),
                    "warmup_divergences": sum(int(r["warm_divergences"]) for r in complete),
                    "depth_saturations": sum(int(r["depth_saturations"]) for r in complete),
                    "minimum_bulk_ess": ess, "maximum_rhat": max(float(d["rhat"]) for d in ds),
                    "minimum_tail_ess": min(float(d["tail_ess"]) for d in ds),
                    "minimum_bulk_ess_per_retained_call": ess / calls,
                    "minimum_bulk_ess_per_retained_second": ess / (retained_ns * 1e-9),
                    "retained_nanoseconds": retained_ns,
                    "realized_ranks": [int(r["rank"]) for r in complete],
                    "final_steps": [float(r["step"]) for r in complete],
                    "independently_recomputed_means": means,
                })
                if case != "logistic":
                    reference = [0.0, 0.0, 1.0, 1.0, 0.0]
                    summary["analytic_reference_means"] = reference
                    summary["absolute_errors_over_mcse"] = [abs(m - r) / float(d["mcse_mean"]) for m, r, d in zip(means, reference, ds)]
            groups.append(summary)
    comparisons = []
    for case in protocol["adaptation"]["targets"]:
        for kind in ("diag", "rank2"):
            a, b = (next(r for r in groups if r["case"] == case and r["method"] == f"fisher_{kind}{period}") for period in (80, 40))
            if "retained_calls" in a and "retained_calls" in b:
                comparisons.append({"case": case, "kind": kind,
                                    "period40_over_period80_calls": b["retained_calls"] / a["retained_calls"],
                                    "period40_over_period80_ess_per_call": b["minimum_bulk_ess_per_retained_call"] / a["minimum_bulk_ess_per_retained_call"],
                                    "period80_divergences": a["retained_divergences"],
                                    "period40_divergences": b["retained_divergences"]})
    return {"chain_outcomes": len(work), "raw_draw_rows": sum(counts.values()),
            "groups": groups, "period_comparisons": comparisons,
            "limits": "One declared seed set, five 2D targets; all chains and divergent/repeated states retained; no quality filter, selection rule or convergence guarantee; analytic-reference errors are descriptive, not a pass/fail test."}


if __name__ == "__main__":
    import sys

    directory = Path(__file__).resolve().parent
    result = analyze(directory)
    output = Path(sys.argv[1]) if len(sys.argv) == 2 else directory / "adaptation-summary.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
