"""Fresh-process scalar/batch construction, fused evaluation and local-fit timings.

Run after building the release regression_benchmark example. Retain every outcome;
ratios are eligible only if both searches meet the same declared score tolerance
and free estimates agree. Memory is Linux VmHWM, not an allocation count.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def eligible_pair(b, s):
    """Scientific gates are independent of timing; failed searches remain ineligible."""
    return (
        b["termination"] == s["termination"] == "ScoreTolerance"
        and math.isclose(b["value"], s["value"], rel_tol=1e-10, abs_tol=1e-9)
        and len(b["gradient"]) == len(s["gradient"])
        and len(b["estimate_free"]) == len(s["estimate_free"])
        and all(math.isclose(x, y, rel_tol=1e-8, abs_tol=1e-6) for x, y in zip(b["gradient"], s["gradient"]))
        and all(math.isclose(x, y, rel_tol=1e-5, abs_tol=1e-5) for x, y in zip(b["estimate_free"], s["estimate_free"]))
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("binary")
    p.add_argument("output")
    p.add_argument("--rows", type=int, nargs="+", default=[10, 1000, 1000000])
    p.add_argument("--runs", type=int, default=3)
    args = p.parse_args()
    if args.runs < 1 or any(n < 1 for n in args.rows):
        p.error("positive rows and runs required")
    report = {"protocol": "alea-regression-v1", "platform": platform.platform(),
              "binary_sha256": digest(args.binary), "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "toolchain": subprocess.check_output(["rustc", "-Vv"], text=True),
              "source_files": {f: digest(f) for f in subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], text=True).splitlines() if (f.startswith("crates/") or f.startswith("tools/downstream-conformance/")) and f.endswith((".rs", ".py", ".toml", ".lock"))},
              "working_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest(),
              "score_tolerance_per_observation": 1e-7,
              "planned_runs": [{"family": f, "mode": m, "rows": n, "repeat": r}
                               for f in ["normal", "bernoulli", "poisson"] for n in args.rows
                               for r in range(args.runs) for m in ["batch", "scalar"]],
              "runs": [], "comparisons": []}
    with Path(args.output).open("x") as output:
        try:
            for family in ["normal", "bernoulli", "poisson"]:
                for n in args.rows:
                    groups = {mode: [] for mode in ["batch", "scalar"]}
                    for repeat in range(args.runs):
                        # Alternate order; use separate processes for all measurements.
                        for mode in (["batch", "scalar"] if repeat % 2 == 0 else ["scalar", "batch"]):
                            command = [args.binary, family, mode, str(n), str(max(3, min(5000, 2000000 // n)))]
                            completed = subprocess.run(command, text=True, capture_output=True)
                            if completed.returncode:
                                row = {"family": family, "mode": mode, "rows": n, "repeat": repeat,
                                       "error": completed.stderr, "returncode": completed.returncode}
                            else:
                                row = json.loads(completed.stdout)
                                row["repeat"] = repeat
                                groups[mode].append(row)
                            report["runs"].append(row)
                    eligible = len(groups["batch"]) == len(groups["scalar"]) == args.runs
                    if eligible:
                        for b, s in zip(groups["batch"], groups["scalar"]):
                            eligible &= eligible_pair(b, s)
                    comparison = {"family": family, "rows": n, "eligible": bool(eligible)}
                    if eligible:
                        for field in ["construction_seconds", "evaluation_seconds", "fit_seconds", "peak_rss_kib"]:
                            if any(r[field] is None for rows in groups.values() for r in rows):
                                comparison[field + "_scalar_over_batch"] = None
                                continue
                            b = statistics.median(r[field] for r in groups["batch"])
                            s = statistics.median(r[field] for r in groups["scalar"])
                            comparison[field + "_scalar_over_batch"] = s / b
                        b = statistics.median(r["construction_seconds"] + r["fit_seconds"] for r in groups["batch"])
                        s = statistics.median(r["construction_seconds"] + r["fit_seconds"] for r in groups["scalar"])
                        comparison["setup_fit_scalar_over_batch"] = s / b
                    report["comparisons"].append(comparison)
                    print(json.dumps(comparison), flush=True)
        finally:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
