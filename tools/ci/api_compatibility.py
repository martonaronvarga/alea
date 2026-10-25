"""Conservative, digest-bound pre-release public API compatibility adjudication.

Removals/signature replacements and additions to existing types/traits require a
potentially-breaking decision. New standalone declarations are additive. This
check does not promise language-level semver completeness: it requires an explicit
review of every diff from the preceding accepted API.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def digest(lines):
    content = "\n".join(sorted(lines)) + "\n"
    return hashlib.sha256(content.encode()).hexdigest()


def difference(before, after):
    old, new = set(before), set(after)
    added, removed = sorted(new - old), sorted(old - new)
    declarations = (
        "pub fn ", "pub mod ", "pub struct ", "pub enum ", "pub trait ",
        "pub type ", "pub const ", "pub static ", "pub use ",
    )
    # Members, impls and variants can affect inference or exhaustive matching.
    member_addition = any(
        line.startswith("impl")
        or line.count("::") > 2
        or (line.startswith("pub ") and not line.startswith(declarations))
        for line in added
    )
    if removed or member_addition:
        level = "potentially_breaking"
    elif added:
        level = "additive"
    else:
        level = "unchanged"
    return {
        "before": digest(old),
        "after": digest(new),
        "added": len(added),
        "removed": len(removed),
        "delta": digest(["+ " + line for line in added] + ["- " + line for line in removed]),
        "classification": level,
        "added_lines": added,
        "removed_lines": removed,
    }


def check(root):
    review = json.loads((root / "tools/ci/api-review.json").read_text())
    fields = {"schema_version", "base_revision", "decision", "rationale", "packages"}
    if set(review) != fields or review["schema_version"] != 1:
        raise ValueError("invalid API compatibility review schema")
    if review["decision"] not in {"pre_release_break", "compatible"} or not review["rationale"].strip():
        raise ValueError("API changes require an explicit compatibility decision and rationale")
    snapshots = root / "tools/ci/public-api"
    if set(review["packages"]) != {path.stem for path in snapshots.glob("alea-*.txt")}:
        raise ValueError("compatibility package inventory mismatch")
    for package, accepted in review["packages"].items():
        after = set((snapshots / f"{package}.txt").read_text().splitlines())
        added, removed = set(accepted["added_lines"]), set(accepted["removed_lines"])
        if not added.issubset(after) or removed & after or added & removed:
            raise ValueError(f"{package}: compatibility delta does not reconstruct the accepted APIs")
        # Shallow CI/downstream archives need no historical Git objects. The
        # reviewed delta reconstructs the old endpoint and verifies both hashes.
        before = (after - added) | removed
        actual = difference(before, after)
        if actual != accepted:
            raise ValueError(f"{package}: API delta changed without compatibility adjudication")
        if actual["classification"] == "potentially_breaking" and review["decision"] != "pre_release_break":
            raise ValueError(f"{package}: potentially breaking API delta mislabeled compatible")
    print("API compatibility: digest-bound pre-release adjudication verified")


if __name__ == "__main__":
    try:
        check(Path(__file__).resolve().parents[2])
    except ValueError as error:
        print(error, file=sys.stderr)
        sys.exit(1)
