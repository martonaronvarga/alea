"""Fresh-process scalar/batch construction, fused evaluation and local-fit timings.

Run after building the release regression_benchmark example. Retain every outcome;
ratios are eligible only if both searches meet the same declared score tolerance
and free estimates agree. Memory is Linux VmHWM, not an allocation count.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time
import statistics
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprints():
    paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], text=True).splitlines()
    return {f: digest(f) for f in paths if (f.startswith("crates/") or f.startswith("tools/downstream-conformance/")) and f.endswith((".rs", ".py", ".toml", ".lock"))}


def eligible_pair(b, s):
    """Scientific gates are independent of timing; failed searches remain ineligible."""
    return (
        b["termination"] == s["termination"] == "ScoreTolerance"
        and b["score_tolerance"] == s["score_tolerance"]
        and math.isfinite(b["score_tolerance"]) and b["score_tolerance"] > 0
        and all(math.isfinite(r["final_value"]) and math.isfinite(r["value"]) and len(r["final_gradient"]) == len(r["estimate_free"]) and
                all(math.isfinite(g) and abs(g) <= r["score_tolerance"] for g in r["final_gradient"])
                for r in (b, s))
        and math.isclose(b["final_value"], s["final_value"], rel_tol=1e-10, abs_tol=1e-9)
        and math.isclose(b["value"], s["value"], rel_tol=1e-10, abs_tol=1e-9)
        and len(b["gradient"]) == len(s["gradient"])
        and len(b["estimate_free"]) == len(s["estimate_free"])
        and all(math.isclose(x, y, rel_tol=1e-8, abs_tol=1e-6) for x, y in zip(b["gradient"], s["gradient"]))
        and all(math.isclose(x, y, rel_tol=1e-5, abs_tol=1e-5) for x, y in zip(b["estimate_free"], s["estimate_free"]))
    )


def revision_comparison(before, after):
    """A whole cohort must qualify before any favorable timing ratio is emitted."""
    if not before or len(before) != len(after) or any("error" in r for r in before + after) or not all(
        eligible_pair(b, a) for b, a in zip(before, after)
    ):
        return {"eligible": False}
    result = {"eligible": True}
    for field in ["construction_seconds", "evaluation_seconds", "fit_seconds", "process_seconds", "peak_rss_kib"]:
        if all(r.get(field) is not None and r[field] > 0 for r in before + after):
            result[field + "_before_over_after"] = statistics.median(r[field] for r in before) / statistics.median(r[field] for r in after)
    result["setup_fit_before_over_after"] = statistics.median(r["construction_seconds"] + r["fit_seconds"] for r in before) / statistics.median(r["construction_seconds"] + r["fit_seconds"] for r in after)
    result["paired_outputs_identical"] = all(
        all(b[k] == a[k] for k in ["value", "gradient", "final_value", "final_gradient", "estimate_free", "termination", "iterations", "evaluations", "score_tolerance", "line_search_failure"])
        for b, a in zip(before, after)
    )
    return result


def attribution_summary(rows):
    """Unadjusted timed-call fraction; the residual includes observer overhead."""
    if not rows or any(r.get("target_calls") != r["evaluations"] or
                       not isinstance(r.get("target_seconds"), (int, float)) or
                       not math.isfinite(r["target_seconds"]) or
                       not 0 <= r["target_seconds"] <= r["fit_seconds"] or
                       r["fit_seconds"] <= 0 for r in rows):
        return {"attribution_valid": False}
    return {"attribution_valid": True,
            "target_fraction_median": statistics.median(r["target_seconds"] / r["fit_seconds"] for r in rows),
            "residual_seconds_median": statistics.median(r["fit_seconds"] - r["target_seconds"] for r in rows)}


def decode_benchmark_output(stdout, job):
    """Malformed output is an infrastructure failure, never a qualified fit."""
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    payload = json.loads(stdout, parse_constant=reject_constant)
    required = {"value", "gradient", "final_value", "final_gradient", "estimate_free",
                "termination", "iterations", "evaluations", "score_tolerance",
                "line_search_failure", "construction_seconds", "evaluation_seconds",
                "fit_seconds", "peak_rss_kib"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("incomplete benchmark output")
    if any(payload.get(key) != job[key] for key in ["family", "mode", "rows"]):
        raise ValueError("benchmark output does not match planned job")
    if job.get("revision") in ("plain", "counted", "timed") and payload.get("profile") != job["revision"]:
        raise ValueError("benchmark observer does not match planned job")
    return payload


def benchmark_revisions(args):
    rng = random.Random(args.seed)
    variants = ["plain", "counted", "timed"] if args.attribution else ["before", "after"]
    planned = []
    for family in ["normal", "bernoulli", "poisson"]:
        for n in args.rows:
            for repeat in range(args.runs):
                order = [(revision, mode) for revision in variants for mode in ["batch", "scalar"]]
                rng.shuffle(order)
                for revision, mode in order:
                    planned.append(dict(family=family, rows=n, repeat=repeat, revision=revision, mode=mode))
    report = {"protocol": "alea-regression-attribution-v1" if args.attribution else "alea-regression-revisions-v1", "platform": platform.platform(),
              "affinity": sorted(os.sched_getaffinity(0)),
              "seed": args.seed, "score_tolerance_per_observation": 1e-7,
              "toolchain": subprocess.check_output(["rustc", "-Vv"], text=True),
              "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "working_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest(),
              "baseline_revision": args.baseline_revision,
              "source_files": source_fingerprints(),
              "binaries": {v: digest(args.binary if v != "before" else args.baseline_binary) for v in variants},
              "planned_runs": planned, "runs": [], "comparisons": []}
    # Seal the declared jobs before executing, and retain every raw process record.
    path = Path(args.output)
    with path.with_suffix(".plan.json").open("x") as plan:
        json.dump(report, plan, indent=2); plan.write("\n")
    with path.open("x") as output, path.with_suffix(".jsonl").open("x") as raw:
        try:
            for job in planned:
                binary = args.baseline_binary if job["revision"] == "before" else args.binary
                command = [binary, job["family"], job["mode"], str(job["rows"]), str(max(3, min(5000, 2000000 // job["rows"])))]
                if args.attribution:
                    command.append(job["revision"])
                row = dict(job)
                started = time.monotonic()
                try:
                    process = subprocess.run(command, capture_output=True, text=True, timeout=300)
                    row.update(returncode=process.returncode, stdout=process.stdout, stderr=process.stderr)
                    if process.returncode == 0:
                        row.update(decode_benchmark_output(process.stdout, job))
                    else:
                        row["error"] = "process failure"
                except (subprocess.TimeoutExpired, ValueError, OSError) as error:
                    row["error"] = str(error)
                row["process_seconds"] = time.monotonic() - started
                report["runs"].append(row)
                raw.write(json.dumps(row, allow_nan=False) + "\n"); raw.flush()
            for family in ["normal", "bernoulli", "poisson"]:
                for n in args.rows:
                    for mode in ["batch", "scalar"]:
                        for variant in variants[1:]:
                            cohorts = [[r for r in report["runs"] if r["family"] == family and r["rows"] == n and r["mode"] == mode and r["revision"] == revision] for revision in [variants[0], variant]]
                            result = revision_comparison(*cohorts)
                            if variant == "timed" and result["eligible"]:
                                result.update(attribution_summary(cohorts[1]))
                            result.update(family=family, rows=n, mode=mode, comparison=variant)
                            report["comparisons"].append(result)
                            print(json.dumps(result), flush=True)
        finally:
            json.dump(report, output, indent=2, allow_nan=False); output.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("binary")
    p.add_argument("output")
    p.add_argument("--rows", type=int, nargs="+", default=[10, 1000, 1000000])
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--attribution", action="store_true")
    p.add_argument("--baseline-binary")
    p.add_argument("--baseline-revision")
    p.add_argument("--seed", type=int, default=271828)
    args = p.parse_args()
    if args.runs < 1 or any(n < 1 for n in args.rows):
        p.error("positive rows and runs required")
    if bool(args.baseline_binary) != bool(args.baseline_revision):
        p.error("baseline binary and revision must be supplied together")
    if args.attribution and args.baseline_binary:
        p.error("attribution and baseline comparison are separate protocols")
    if args.baseline_binary or args.attribution:
        benchmark_revisions(args)
        raise SystemExit(0)
    report = {"protocol": "alea-regression-v1", "platform": platform.platform(),
              "binary_sha256": digest(args.binary), "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "toolchain": subprocess.check_output(["rustc", "-Vv"], text=True),
              "source_files": source_fingerprints(),
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
