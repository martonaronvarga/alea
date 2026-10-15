#!/usr/bin/env python3
"""Build and exactly replay the pinned upstream Walnutpie reference probe."""

import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
WALNUTPIE_REV = "5710b0334ba69cf0caa35fc02a2338b61d756953"
NUTS_RS_REV = "a762aae513bf7bb5ccb27ffc9a938a923fc837ec"
WALNUTPIE = ROOT / "target/walnutpie-reference-upstream"
NUTS_RS = ROOT / "target/nuts-reference-upstream"
PROBE = Path(__file__).with_name("probe.cpp")
FIXTURE = Path(__file__).with_name("fixtures") / "walnutpie-trace.csv"
FIXTURE_SHA256 = "4fdaf9751c982c4cb96a516b16ad8c790d87c59d4871d742fa1939e129b3e259"


def revision(checkout):
    return subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()


def require_checkout(path, expected):
    if not path.is_dir():
        raise SystemExit(f"missing checkout: {path}; see README.md")
    actual = revision(path)
    if actual != expected:
        raise SystemExit(f"wrong revision for {path}: {actual}, expected {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="also write the verified trace")
    args = parser.parse_args()
    require_checkout(WALNUTPIE, WALNUTPIE_REV)
    require_checkout(NUTS_RS, NUTS_RS_REV)

    fixture = FIXTURE.read_bytes()
    digest = hashlib.sha256(fixture).hexdigest()
    if digest != FIXTURE_SHA256:
        raise SystemExit(f"fixture digest changed: {digest}")

    eigen_flags = subprocess.check_output(
        ["pkg-config", "--cflags", "eigen3"], text=True
    ).split()
    with tempfile.TemporaryDirectory(prefix="alea-walnutpie-") as directory:
        binary = Path(directory) / "probe"
        subprocess.run(
            [
                "c++",
                "-std=c++20",
                "-O2",
                "-I",
                str(WALNUTPIE / "include"),
                *eigen_flags,
                str(PROBE),
                "-o",
                str(binary),
            ],
            check=True,
        )
        first = subprocess.check_output([str(binary)])
        second = subprocess.check_output([str(binary)])

    if first != second:
        raise SystemExit("upstream probe was not exactly deterministic")
    if first != fixture:
        raise SystemExit("upstream trace differs from the checked fixture")
    if args.output is not None:
        args.output.write_bytes(first)
    print(f"verified Walnutpie {WALNUTPIE_REV} trace ({digest})")
    print(f"verified nuts-rs provenance revision {NUTS_RS_REV}")


if __name__ == "__main__":
    main()
