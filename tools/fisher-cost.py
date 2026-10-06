#!/usr/bin/env python3
"""Measure the fixed Fisher profiling workloads on Linux, serially.

Run inside the pinned profiling shell after building fisher_profile in release.
GNU time measures each child's peak RSS separately; it is not heap allocation.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parent.parent
WORKLOADS = (
    ("fit64", ("fit", "50", "64", "64")),
    ("fit256", ("fit", "20", "256", "160")),
    ("covariance64", ("covariance", "10", "64")),
    ("diagonal64", ("diagonal", "10", "64")),
    ("lowrank64", ("lowrank", "10", "64")),
)
SOURCES = (
    "flake.lock",
    "crates/Cargo.lock",
    "crates/Cargo.toml",
    "crates/alea-math/src/fisher.rs",
    "crates/alea-mcmc/src/adapt/fisher.rs",
    "crates/alea-mcmc/examples/fisher_profile.rs",
    "crates/alea-mcmc/benches/support/warmup.rs",
    "tools/fisher-cost.py",
)


def fingerprint(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def measure(command, timer):
    with tempfile.TemporaryDirectory(prefix="alea-fisher-cost-") as directory:
        usage = Path(directory) / "usage"
        start = time.perf_counter()
        subprocess.run(
            [timer, "-f", "%U %S %M", "-o", str(usage), *command],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        wall = time.perf_counter() - start
        user, system, rss = usage.read_text().split()
    return {
        "wall_seconds": wall,
        "user_seconds": float(user),
        "system_seconds": float(system),
        "peak_rss_kib": int(rss),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()
    if args.runs < 2 or args.warmups < 0:
        parser.error("at least two runs and nonnegative warmups are required")
    if platform.system() != "Linux":
        parser.error("Linux is required for the declared GNU time RSS units")
    if args.output.exists():
        parser.error("output already exists; choose a new artifact path")
    timer = shutil.which("time")
    if timer is None:
        parser.error("GNU time is required; use nix develop .#profiling")
    timer_version = subprocess.check_output([timer, "--version"], text=True).splitlines()[0]
    if "GNU" not in timer_version:
        parser.error("the time executable must be GNU time")
    binary = args.binary.resolve(strict=True)
    hashes = {name: fingerprint(ROOT / name) for name in SOURCES}
    binary_hash = fingerprint(binary)
    results = {name: {"arguments": list(arguments), "samples": []} for name, arguments in WORKLOADS}
    # Alternate direction each round to reduce consistent ordering bias. No
    # concurrent benchmarks, compilation, or instrumentation in this runner.
    for round_index in range(args.warmups + args.runs):
        order = WORKLOADS if round_index % 2 == 0 else reversed(WORKLOADS)
        for name, arguments in order:
            sample = measure([str(binary), *arguments], timer)
            if round_index >= args.warmups:
                sample["round"] = round_index - args.warmups
                results[name]["samples"].append(sample)
            print(f"round {round_index + 1}: {name} {sample['wall_seconds']:.6f}s", flush=True)
    if fingerprint(binary) != binary_hash or any(
        fingerprint(ROOT / name) != digest for name, digest in hashes.items()
    ):
        raise RuntimeError("binary or measured sources changed during measurement")
    for result in results.values():
        walls = [sample["wall_seconds"] for sample in result["samples"]]
        result["wall_mean_seconds"] = statistics.mean(walls)
        result["wall_stdev_seconds"] = statistics.stdev(walls)
    artifact = {
        "schema": 1,
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "cpuinfo": Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0],
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "gnu_time": timer_version,
        "binary": str(binary),
        "binary_sha256": binary_hash,
        "source_sha256": hashes,
        "environment": {key: os.environ.get(key) for key in (
            "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_PROFILE_RELEASE_DEBUG",
            "RAYON_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
        )},
        "warmup_rounds": args.warmups,
        "measured_rounds": args.runs,
        "results": results,
    }
    # Failed children abort before publication; never overwrite an earlier run.
    with args.output.open("x") as stream:
        json.dump(artifact, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
