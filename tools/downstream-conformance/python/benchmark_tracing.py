"""Paired fixed-work overhead experiment using separately saved release binaries.

Optimization terminations, including failed line searches, are retained unchanged;
matching execution costs are not successful-inference performance claims.
No timings become correctness assertions. Failed processes and mismatched numerical
fingerprints are retained and block a qualified overhead result for that case.
"""
import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(records, rounds):
    result = {}
    for case in sorted({r["case"] for r in records}):
        rows = [r for r in records if r["case"] == case]
        fingerprints = {v["fingerprint"] for r in rows for v in r["measurements"]}
        planned = {(index, mode) for index in range(rounds) for mode in ("off", "none", "filtered", "info")}
        actual = {(r["round"], r["mode"]) for r in rows}
        qualified = actual == planned and len(fingerprints) == 1 and len(rows) == 4 * rounds and all(r["status"] == 0 and len(r["measurements"]) == 3 for r in rows)
        result[case] = {"identity_passed": qualified, "planned_processes": 4 * rounds,
                        "completed_processes": sum(r["status"] == 0 for r in rows),
                        "workload_outcomes": sorted({v.get("outcome", "unknown") for r in rows for v in r["measurements"]}),
                        "all_fits_score_tolerance": case not in ("bernoulli", "poisson") or all(v.get("outcome") == "ScoreTolerance" for r in rows for v in r["measurements"]), "modes": {}}
        if not qualified:
            continue
        baseline = {r["round"]: statistics.median(v["nanoseconds"] for v in r["measurements"][1:]) for r in rows if r["mode"] == "off"}
        for mode in ("none", "filtered", "info"):
            selected = [r for r in rows if r["mode"] == mode]
            ratios = [statistics.median(v["nanoseconds"] for v in r["measurements"][1:]) / baseline[r["round"]] for r in selected]
            rng = random.Random(20261029)
            boot = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(10000))
            entry = {"median_paired_ratio": statistics.median(ratios), "paired_bootstrap_95": [boot[249], boot[9749]],
                     "median_ns": statistics.median(v["nanoseconds"] for r in selected for v in r["measurements"][1:]),
                     "median_cold_ns": statistics.median(r["measurements"][0]["nanoseconds"] for r in selected),
                     "event_counts": sorted({v["events"] for r in selected for v in r["measurements"]})}
            median_limit, upper_limit = (1.02, 1.05) if mode == "info" else (1.01, 1.02)
            entry["overhead_objective_passed"] = entry["median_paired_ratio"] <= median_limit and entry["paired_bootstrap_95"][1] <= upper_limit
            result[case]["modes"][mode] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--cases", nargs="+", default=["stream", "hmc", "nuts", "rwmh", "parallel50", "bernoulli", "poisson"])
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root, text=True).splitlines()
    sources = {p: digest(root / p) for p in paths if (root / p).is_file() and (p.endswith(".rs") or p.endswith("Cargo.toml") or p.endswith("Cargo.lock") or p.endswith("benchmark_tracing.py"))}
    manifest = {"schema_version": 1, "rounds": args.rounds, "cases": args.cases, "host": platform.platform(),
                "cpu": Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else None,
                "binaries": {"off": digest(args.off), "on": digest(args.on)}, "sources": sources,
                "protocol": "3 repetitions/process; first cold, then 2 steady; 15 randomized paired rounds by default; count-only subscriber; no event buffering",
                "clock": "Rust Instant, end-to-end workload including setup and output collection/fingerprinting; startup/import excluded",
                "memory": "subscriber buffers no event records; RSS and allocator counts not measured by this runner"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rng = random.Random(20261029)
    records = []
    with (args.output / "runs.jsonl").open("x") as raw:
        for round_index in range(args.rounds):
            cases = args.cases.copy()
            rng.shuffle(cases)
            for case in cases:
                modes = ["off", "none", "filtered", "info"]
                rng.shuffle(modes)
                for mode in modes:
                    binary = args.off if mode == "off" else args.on
                    process = subprocess.run([str(binary.resolve()), case, mode, "3"], capture_output=True, text=True)
                    try:
                        measurements = [json.loads(line) for line in process.stdout.splitlines()]
                    except (ValueError, TypeError):
                        measurements = []
                    record = {"round": round_index, "case": case, "mode": mode, "status": process.returncode,
                              "measurements": measurements, "stdout": process.stdout, "stderr": process.stderr}
                    records.append(record)
                    raw.write(json.dumps(record) + "\n")
                    raw.flush()
    summary = summarize(records, args.rounds)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not all(case["identity_passed"] for case in summary.values()):
        raise SystemExit("failed/incomplete or numerically mismatched runs; see retained records")


if __name__ == "__main__":
    main()
