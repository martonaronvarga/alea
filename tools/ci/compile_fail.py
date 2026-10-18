#!/usr/bin/env python3
"""Verify downstream compiler-negative ownership and composition contracts."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tools/compile-fail/Cargo.toml"


@dataclass(frozen=True)
class Case:
    name: str
    code: str
    needles: tuple[str, ...]

    @property
    def source(self) -> Path:
        return ROOT / f"tools/compile-fail/cases/{self.name}.rs"


CASES = (
    Case("phase_reuse", "E0382", ("phase", "moved")),
    Case("adaptation_after_freeze", "E0382", ("warmup", "moved")),
    Case(
        "incompatible_walnuts_integrator",
        "E0308",
        ("Leapfrog", "TwoStage"),
    ),
    Case(
        "approximate_into_exact_nuts",
        "E0308",
        ("ApproximateMclmc", "Hmc"),
    ),
)


def local_errors(output: str, source: Path) -> list[dict[str, object]]:
    errors = []
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("reason") != "compiler-message":
            continue
        message = event.get("message")
        if not isinstance(message, dict) or message.get("level") != "error":
            continue
        spans = message.get("spans", [])
        if any(is_primary_source_span(span, source) for span in spans):
            errors.append(message)
    return errors


def is_primary_source_span(span: object, source: Path) -> bool:
    if not isinstance(span, dict) or span.get("is_primary") is not True:
        return False
    file_name = Path(str(span.get("file_name")))
    if not file_name.is_absolute():
        file_name = MANIFEST.parent / file_name
    return file_name.resolve() == source


def verify(case: Case) -> None:
    command = [
        "cargo",
        "check",
        "--manifest-path",
        str(MANIFEST),
        "--locked",
        "--offline",
        "--example",
        case.name,
        "--message-format=json",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        raise AssertionError(f"{case.name} unexpectedly compiled")

    errors = local_errors(result.stdout, case.source.resolve())
    matching = [
        error
        for error in errors
        if isinstance(error.get("code"), dict)
        and error["code"].get("code") == case.code
    ]
    rendered = "\n".join(str(error.get("rendered", "")) for error in matching)
    if not matching or any(needle not in rendered for needle in case.needles):
        sys.stderr.write(result.stderr)
        sys.stderr.write(result.stdout)
        raise AssertionError(
            f"{case.name} did not emit source-local {case.code} with {case.needles}"
        )


def main() -> None:
    for case in CASES:
        verify(case)
    print(f"compile-fail contracts: {len(CASES)} cases")


if __name__ == "__main__":
    main()
