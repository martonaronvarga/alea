#!/usr/bin/env python3
"""Measure predeclared Fisher rank/history cold-fit time and process peak RSS."""

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
CASES = tuple(
    (f"d{dimension}_h{history}_r{rank}", repetitions, dimension, history, rank)
    for dimension, history, ranks, repetitions in (
        (64, 40, (1, 2, 4), 20),
        (64, 160, (2, 4, 8), 20),
        (256, 40, (2, 4, 8), 3),
        (256, 160, (2, 4, 8), 3),
    )
    for rank in ranks
)
SOURCES = (
    "flake.lock",
    "crates/Cargo.lock",
    "crates/alea-math/src/fisher.rs",
    "crates/alea-mcmc/src/adapt/fisher.rs",
    "crates/alea-mcmc/examples/fisher_profile.rs",
    "tools/m4f-budgets.py",
)


def fingerprint(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def measure(command, timer):
    with tempfile.TemporaryDirectory(prefix="alea-m4f-budget-") as directory:
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
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()
    if args.runs < 2 or args.warmups < 0:
        parser.error("at least two runs and nonnegative warmups are required")
    if platform.system() != "Linux":
        parser.error("Linux is required for GNU time RSS units")
    if args.output.exists():
        parser.error("output already exists; choose a new artifact path")
    timer = shutil.which("time")
    if timer is None or "GNU" not in subprocess.check_output(
        [timer, "--version"], text=True
    ).splitlines()[0]:
        parser.error("GNU time is required; use nix develop .#profiling")
    binary = args.binary.resolve(strict=True)
    binary_hash = fingerprint(binary)
    hashes = {name: fingerprint(ROOT / name) for name in SOURCES}
    results = {
        name: {
            "arguments": ["fit", str(repetitions), str(dimension), str(history), str(rank)],
            "dimension": dimension,
            "history": history,
            "rank_cap": rank,
            "repetitions": repetitions,
            "minimum_ring_bytes": 2 * dimension * history * 8,
            "samples": [],
        }
        for name, repetitions, dimension, history, rank in CASES
    }
    for round_index in range(args.warmups + args.runs):
        order = CASES if round_index % 2 == 0 else tuple(reversed(CASES))
        for name, repetitions, dimension, history, rank in order:
            arguments = ["fit", str(repetitions), str(dimension), str(history), str(rank)]
            sample = measure([str(binary), *arguments], timer)
            if round_index >= args.warmups:
                sample["round"] = round_index - args.warmups
                results[name]["samples"].append(sample)
            print(
                f"round {round_index + 1}: {name} {sample['wall_seconds']:.6f}s",
                flush=True,
            )
    if fingerprint(binary) != binary_hash or any(
        fingerprint(ROOT / name) != digest for name, digest in hashes.items()
    ):
        raise RuntimeError("binary or measured sources changed during measurement")
    for result in results.values():
        walls = [sample["wall_seconds"] for sample in result["samples"]]
        rss = [sample["peak_rss_kib"] for sample in result["samples"]]
        result["wall_mean_seconds"] = statistics.mean(walls)
        result["wall_stdev_seconds"] = statistics.stdev(walls)
        result["per_fit_mean_seconds"] = result["wall_mean_seconds"] / result["repetitions"]
        result["peak_rss_kib_range"] = [min(rss), max(rss)]
    artifact = {
        "schema": 1,
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "cpuinfo": Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0],
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "binary": str(binary),
        "binary_sha256": binary_hash,
        "source_sha256": hashes,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "RUSTFLAGS",
                "CARGO_ENCODED_RUSTFLAGS",
                "RAYON_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
            )
        },
        "warmup_rounds": args.warmups,
        "measured_rounds": args.runs,
        "results": results,
    }
    with args.output.open("x") as stream:
        json.dump(artifact, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
