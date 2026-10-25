"""Verify the reviewed API break for first publication, without a semver claim."""
import argparse
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/ci"))
from api_compatibility import check, digest

PACKAGES = ("alea-core", "alea-math", "alea-mcmc", "alea-runtime", "alea-math+faer", "alea-mcmc+faer")


def validate(root: Path, baseline: Path) -> None:
    if root.resolve() == baseline.resolve():
        raise ValueError("first publication requires the independent unpublished baseline")
    check(root)
    review = json.loads((root / "tools/ci/api-review.json").read_text())
    if review["decision"] != "pre_release_break":
        raise ValueError("first publication requires the deliberate unpublished API review")
    with (root / "crates/Cargo.toml").open("rb") as stream:
        if tomllib.load(stream)["workspace"]["package"]["version"] != "0.1.0":
            raise ValueError("initial-publication exception is only for 0.1.0")
    for name in PACKAGES:
        path = baseline / "tools/ci/public-api" / f"{name}.txt"
        if digest(path.read_text().splitlines()) != review["packages"][name]["before"]:
            raise ValueError(f"{name}: baseline is not the reviewed unpublished API")
    print("first-publication API endpoints verified; no semver compatibility claim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    args = parser.parse_args()
    validate(ROOT, args.baseline)


if __name__ == "__main__":
    main()
