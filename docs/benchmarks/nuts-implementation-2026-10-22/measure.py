"""Serial paired work timings; freeze inputs before launching measured processes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    binaries = {name: getattr(args, name).resolve() for name in ("baseline", "current")}
    output = args.output.resolve()
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    cases = [
        ["flat", "2", "16384"], ["flat", "16", "16384"], ["flat", "64", "16384"],
        ["fit", "16", "1024"], ["fit", "64", "128"], ["fit", "128", "32"],
    ]
    paths = subprocess.check_output(
        ["rg", "--files", "crates", "-g", "*.rs", "-g", "Cargo.toml", "-g", "Cargo.lock"],
        cwd=root, text=True,
    ).splitlines()
    paths += ["flake.nix", "flake.lock"]
    paths += [str(p.relative_to(root)) for p in sorted(
        (root / "tools/nuts-reference/implementation-profile").rglob("*")
    ) if p.is_file()]
    protocol = {
        "schema_version": 1, "host": platform.uname()._asdict(), "cpu": cpu,
        "cpu_models": sorted({line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")}),
        "rounds": 7, "cases": cases,
        "binaries": {name: {"path": str(path), "sha256": sha(path)} for name, path in binaries.items()},
        "current_sources_sha256": {p: sha(root / p) for p in sorted(paths)},
        "baseline_source_archive_sha256": sha(root / ".agent/nuts-implementation/baseline-source.tar.gz"),
        "timers": "Rust Instant nanoseconds inside each process; Python perf_counter_ns around blocking subprocess.run, no timeout polling",
        "build": "Both standalone consumers use Rust 1.95, release fat LTO, codegen-units=1 and faer, with identical Cargo.lock and harness source",
        "limits": "One host; repeated timing seeds, all rounds retained; flat is an improper mechanical fixture, not an inference target; no promotion threshold",
        "adaptation": {"targets": ["correlated", "banana", "logistic", "funnel", "rotated"],
                       "methods": ["cov_diag", "fisher_diag80", "fisher_diag40", "fisher_rank280", "fisher_rank240"],
                       "chains": 4, "seeds": [23001, 23002, 23003, 23004], "warmup": 512, "retained": 2048,
                       "acceptance_target": 0.8, "initial_step": 0.1, "depth": 8, "history": 160,
                       "limits": "Only period differs within a Fisher pair; covariance/Fisher also differ in estimator and schedule; rank2 is full rank in these 2D cases"},
    }
    with (output / "protocol.json").open("x") as sink:
        json.dump(protocol, sink, indent=2)
        sink.write("\n")
    records = []
    with (output / "timings.jsonl").open("x") as raw:
        for case in cases:
            for repetition in range(7):
                order = ("baseline", "current") if repetition % 2 == 0 else ("current", "baseline")
                for name in order:
                    start = time.perf_counter_ns()
                    result = subprocess.run([str(binaries[name]), *case], capture_output=True, text=True)
                    record = {"case": case, "repetition": repetition, "binary": name,
                              "process_nanoseconds": time.perf_counter_ns() - start,
                              "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
                    if result.returncode == 0:
                        record["measurement"] = json.loads(result.stdout)
                    raw.write(json.dumps(record, allow_nan=False) + "\n")
                    raw.flush()
                    records.append(record)
            print("completed paired", *case, flush=True)
    summary = []
    for case in cases:
        pairs = []
        for repetition in range(7):
            values = {r["binary"]: r for r in records if r["case"] == case and r["repetition"] == repetition}
            if any(r["returncode"] != 0 for r in values.values()):
                pairs.append({"repetition": repetition, "status": "failed"})
                continue
            before, after = (values[name]["measurement"] for name in ("baseline", "current"))
            if case[0] == "flat":
                assert before["edges"] == after["edges"] == before["calls"] == after["calls"] == 63 * int(case[2])
            pairs.append({"repetition": repetition, "status": "completed",
                          "current_over_baseline": after["nanoseconds"] / before["nanoseconds"],
                          "baseline_nanoseconds": before["nanoseconds"], "current_nanoseconds": after["nanoseconds"]})
        valid = [p for p in pairs if p["status"] == "completed"]
        summary.append({"case": case, "pairs": pairs, "completed_pairs": len(valid),
                        "median_paired_current_over_baseline": median(p["current_over_baseline"] for p in valid) if valid else None})
    (output / "timing-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    start = time.perf_counter_ns()
    result = subprocess.run([str(binaries["current"]), "adaptation", str(output / "adaptation")], capture_output=True, text=True)
    (output / "adaptation-process.json").write_text(json.dumps({
        "returncode": result.returncode, "process_nanoseconds": time.perf_counter_ns() - start,
        "stdout": result.stdout, "stderr": result.stderr,
    }, indent=2) + "\n")
    for p, digest in protocol["current_sources_sha256"].items():
        assert sha(root / p) == digest, p
    for name, path in binaries.items():
        assert sha(path) == protocol["binaries"][name]["sha256"], name
    print("timings and adaptation finished; source and executable hashes unchanged", flush=True)


if __name__ == "__main__":
    main()
