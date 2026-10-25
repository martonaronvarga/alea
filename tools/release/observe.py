#!/usr/bin/env python3
"""Pair a fixed native workload with an accepted source; never gate on timing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import tomllib


def source_identity(root):
    files = [root / "crates/Cargo.toml", root / "crates/Cargo.lock"]
    for name in ("alea-core", "alea-math", "alea-mcmc", "alea-runtime"):
        files.append(root / "crates" / name / "Cargo.toml")
        files.extend((root / "crates" / name / "src").rglob("*.rs"))
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source, baseline, output = args.source.resolve(), args.baseline.resolve(), args.output.resolve()
    if source == baseline:
        parser.error("candidate cannot be its own accepted baseline")
    output.mkdir(parents=True, exist_ok=False)
    probe = (source / "tools/release/examples/probe.rs").read_bytes()
    env = dict(os.environ, CARGO_INCREMENTAL="0")
    env.setdefault("CARGO_TARGET_DIR", str(source / "crates/target"))
    env["CARGO_TARGET_DIR"] = str(Path(env["CARGO_TARGET_DIR"]).resolve())
    binaries = {}
    with tempfile.TemporaryDirectory(prefix="alea-observe-") as temporary:
        for label, root in (("baseline", baseline), ("candidate", source)):
            project = Path(temporary) / label
            (project / "src").mkdir(parents=True)
            (project / "src/main.rs").write_bytes(probe)
            dependencies = "\n".join(f'{name} = {{path = {json.dumps(str(root / "crates" / name))}}}'
                                      for name in ("alea-core", "alea-math", "alea-mcmc", "alea-runtime"))
            name = "alea-observe-" + label
            (project / "Cargo.toml").write_text(f'[package]\nname="{name}"\nversion="0.0.0"\nedition="2024"\nrust-version="1.95"\n[workspace]\n[dependencies]\n{dependencies}\n', encoding="utf-8")
            subprocess.run(["cargo", "generate-lockfile"], cwd=project, env=env, check=True)
            subprocess.run(["cargo", "fetch", "--locked"], cwd=project, env=env, check=True)
            subprocess.run(["cargo", "build", "--release", "--locked", "--offline"], cwd=project, env=env, check=True)
            (output / (label + ".lock")).write_bytes((project / "Cargo.lock").read_bytes())
            binaries[label] = Path(env["CARGO_TARGET_DIR"]) / "release" / (name + (".exe" if os.name == "nt" else ""))
        locks = [tomllib.loads((output / (label + ".lock")).read_text()) for label in ("baseline", "candidate")]
        external = [[p for p in lock["package"] if "source" in p] for lock in locks]
        if external[0] != external[1]:
            raise ValueError("paired probes resolved different external dependency graphs")
        rows = []
        for repetition in range(5):
            for label in (("baseline", "candidate") if repetition % 2 == 0 else ("candidate", "baseline")):
                raw = subprocess.check_output([str(binaries[label])], env=env, text=True)
                rows.extend(dict(json.loads(line), variant=label, repetition=repetition) for line in raw.splitlines())
    summary = []
    for dimension in (2, 16, 64):
        ratios = []
        for repetition in range(5):
            pair = {r["variant"]:r for r in rows if r["dimension"] == dimension and r["repetition"] == repetition}
            for field in ("gradient_calls", "draw_checksum"):
                if pair["baseline"][field] != pair["candidate"][field]:
                    raise ValueError("unchanged fixed kernel control differs: " + field)
            ratios.append(pair["candidate"]["elapsed_ns"] / pair["baseline"]["elapsed_ns"])
        summary.append({"dimension":dimension,"median_candidate_over_baseline":statistics.median(ratios),
                        "observed_ratio_range":[min(ratios),max(ratios)]})
    value = {"schema_version":1,"purpose":"descriptive regression observations; no timing threshold",
             "host":{"node":platform.node(),"system":platform.system(),"machine":platform.machine(),"processor":platform.processor()},
             "compiler":subprocess.check_output(["rustc","-vV"],text=True),
             "protocol":{"burn":33,"draws":5000,"repetitions":5,"dimensions":[2,16,64],"step":0.25,"depth":8,
                         "jitter":0.1,"position_turning":True,"scope":"transition plus checksum; builds excluded; generic release opt-level3"},
             "probe_sha256":hashlib.sha256(probe).hexdigest(),"rows":rows,"summary":summary}
    value["source_inputs"] = {"baseline":source_identity(baseline),"candidate":source_identity(source)}
    (output / "report.json").write_text(json.dumps(value,indent=2)+'\n',encoding="utf-8")
    print(json.dumps({"host":value["host"],"summary":summary}))


if __name__ == "__main__":
    main()
