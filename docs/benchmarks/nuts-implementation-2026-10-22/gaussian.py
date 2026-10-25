"""Supplemental matched-policy native Gaussian comparison, sealed separately."""
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[3]
    output = Path(__file__).resolve().parent
    binaries = {name: root / f".agent/nuts-implementation/{name}-gaussian" for name in ("baseline", "current")}
    source = "crates/alea-runtime/examples/overhead_profile.rs"
    assert sha(root / source) == sha(root / ".agent/nuts-implementation/baseline" / source)
    primary = json.loads((output / "protocol.json").read_text())
    for path, digest in primary["current_sources_sha256"].items():
        assert sha(root / path) == digest, path
    cpu = primary["cpu"]
    os.sched_setaffinity(0, {cpu})
    cases = [["kernel", str(d), "32768", "1"] for d in (2, 16, 64)]
    protocol = {"cpu": cpu, "cases": cases, "rounds": 7,
                "harness_sha256": sha(root / source),
                "current_source_identities": "protocol.json/current_sources_sha256",
                "build": "Both use the unchanged workspace release/fat-LTO profile, default features and stable Rust1.95; distinct from the CGU1/Faer flat/fit fixture",
                "binaries": {name: {"path": str(path), "sha256": sha(path)} for name, path in binaries.items()},
                "limits": "Native isotropic Gaussian, fixed identity/step0.3/depth8/start0.1, one repeated seed21001; clocks include construction and draw collection; different RNG consumption permits different actual calls; no ESS or mature-library speed claim"}
    with (output / "gaussian-protocol.json").open("x") as sink:
        json.dump(protocol, sink, indent=2)
        sink.write("\n")
    records = []
    with (output / "gaussian.jsonl").open("x") as raw:
        for case in cases:
            for repetition in range(7):
                order = ("baseline", "current") if repetition % 2 == 0 else ("current", "baseline")
                for name in order:
                    start = time.perf_counter_ns()
                    result = subprocess.run([str(binaries[name]), *case], capture_output=True, text=True)
                    row = {"case": case, "repetition": repetition, "binary": name,
                           "process_nanoseconds": time.perf_counter_ns() - start,
                           "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
                    if result.returncode == 0:
                        row["measurement"] = json.loads(result.stdout)
                    records.append(row)
                    raw.write(json.dumps(row, allow_nan=False) + "\n")
                    raw.flush()
            print("completed native Gaussian", case[1], flush=True)
    summary = []
    for case in cases:
        pairs = []
        for repetition in range(7):
            values = {r["binary"]: r for r in records if r["case"] == case and r["repetition"] == repetition}
            if any(r["returncode"] != 0 for r in values.values()):
                pairs.append({"repetition": repetition, "status": "failed"})
                continue
            before, after = (values[name]["measurement"] for name in ("baseline", "current"))
            pairs.append({"repetition": repetition, "status": "completed",
                          "phase_ratio": after["seconds"] / before["seconds"],
                          "call_ratio": after["calls"] / before["calls"],
                          "seconds_per_call_ratio": (after["seconds"] / after["calls"]) / (before["seconds"] / before["calls"]),
                          "baseline_seconds": before["seconds"], "current_seconds": after["seconds"]})
        complete = [p for p in pairs if p["status"] == "completed"]
        summary.append({"case": case, "pairs": pairs, "completed_pairs": len(complete),
                        **{f"median_{field}": median(p[field] for p in complete) if complete else None for field in ("phase_ratio", "call_ratio", "seconds_per_call_ratio")}})
    (output / "gaussian-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, path in binaries.items():
        assert sha(path) == protocol["binaries"][name]["sha256"]
    print("native Gaussian results retained; executable identities verified", flush=True)


if __name__ == "__main__":
    main()
