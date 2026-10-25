#!/usr/bin/env python3
"""Build pinned Stan headers and Alea workers; retain operation-matched observations.

Run inside the selected Alea Nix shell. The Stan worktree is read only: all Stan
headers come from git archive at the supplied commit, not its dirty checkout.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import statistics
import subprocess
import tarfile


def run(command, **kwargs):
    return subprocess.check_output([str(x) for x in command], text=True, **kwargs)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stan-repo", type=Path, required=True)
    parser.add_argument("--stan-revision", default="c8b33c42cc5c222802579084e7096358a0a3423a")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--features", default="")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--cpu", type=int, help="pin measurement workers to one allowed Linux CPU")
    args = parser.parse_args()
    if args.pairs < 3:
        parser.error("at least three alternating process pairs are required")
    root = Path(__file__).resolve().parents[2]
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    stan = args.stan_repo.resolve()
    revision = run(["git", "-C", stan, "rev-parse", args.stan_revision + "^{commit}"]).strip()
    raw = subprocess.check_output(["git", "-C", str(stan), "archive", revision, "stan", "LICENSE.md", "licenses"])
    headers = output / "stan-source"
    headers.mkdir()
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        archive.extractall(headers, filter="data")
    with gzip.GzipFile(filename=str(output / "stan-source.tar.gz"), mode="wb", mtime=0) as stream:
        stream.write(raw)
    compiler = os.environ.get("CXX", "c++")
    library = stan / "lib/tbb"
    flags = ["-O3", "-DNDEBUG", "-std=c++17", "-pthread", "-DSTAN_THREADS",
             "-I" + str(headers), "-I" + str(stan / "lib/eigen_3.4.0"),
             "-I" + str(stan / "lib/boost_1.87.0"), "-I" + str(stan / "lib/tbb_2020.3/include"),
             "-I" + str(stan / "lib/sundials_6.1.1/include")]
    command = [compiler, *flags, "-MMD", "-MF", str(output / "stan.d"),
               str(root / "tools/stan-reference/wiener_benchmark.cpp"), "-L" + str(library),
               "-Wl,-rpath," + str(library), "-ltbb", "-o", str(output / "stan-worker")]
    with (output / "stan-build.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    build = ["cargo", "build", "--release", "--locked", "--manifest-path", str(root / "crates/Cargo.toml"),
             "-p", "alea-distributions", "--example", "wiener_benchmark"]
    if args.features:
        build += ["--features", args.features]
    with (output / "rust-build.log").open("w") as log:
        subprocess.run(build, cwd=root, check=True, stdout=log, stderr=subprocess.STDOUT)
    target = Path(os.environ.get("CARGO_TARGET_DIR", str(root / "crates/target")))
    alea = output / "alea-worker"
    shutil.copyfile(target / "release/examples/wiener_benchmark", alea)
    alea.chmod(0o755)
    cpp_dependencies = shlex.split((output / "stan.d").read_text().replace("\\\n", " ").split(":", 1)[1])
    dependencies = {str(Path(p).resolve()): digest(Path(p)) for p in cpp_dependencies}
    sources = sorted({p for p in (root / "crates").glob("alea-*/src/**/*.rs")}
                     | {p for p in (root / "crates").glob("alea-*/Cargo.toml")}
                     | {root / "crates/Cargo.toml", root / "crates/Cargo.lock",
                        root / "flake.nix", root / "flake.lock",
                        root / "crates/alea-distributions/examples/wiener_benchmark.rs",
                        Path(__file__), root / "tools/stan-reference/wiener_benchmark.cpp"})
    sources = sorted(set(sources)
                     | set((root / "crates").glob("alea-*/build.rs"))
                     | {p for p in (root / "crates/alea-ffi/vendor/cubature").rglob("*")
                        if p.is_file() and (p.suffix in {".c", ".h"} or p.name.startswith("COPYING"))})
    with tarfile.open(output / "alea-inputs.tar.gz", "w:gz") as archive:
        for path in sources:
            archive.add(path, arcname=path.relative_to(root), recursive=False)
    rows = []
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    for family in (4, 5, 7):
        for operation in ("logp", "fused", "batch_direct", "batch_buckets", "batch_vectorized"):
            if family == 7 and operation == "batch_buckets":
                continue
            repeats = (20 if family == 7 else 1000) if operation in ("logp", "fused") else (2 if family == 7 else 20)
            size = 16 if family == 7 else 1000
            for pair in range(args.pairs):
                for label in (("stan", "alea") if pair % 2 == 0 else ("alea", "stan")):
                    worker = output / "stan-worker" if label == "stan" else alea
                    row = json.loads(run([worker, family, operation, repeats, size]))
                    row.update(engine=label, pair=pair)
                    rows.append(row)
                    print(label, family, operation, pair, flush=True)
    summaries = []
    for family, operation in sorted({(r["family"], r["operation"]) for r in rows}):
        ratios, value_errors, score_errors = [], [], []
        for pair in range(args.pairs):
            both = {r["engine"]: r for r in rows if r["family"] == family and r["operation"] == operation and r["pair"] == pair}
            a, s = both["alea"], both["stan"]
            assert a["repetitions"] == s["repetitions"] and a["observations"] == s["observations"]
            ratios.append(a["elapsed_ns"] / s["elapsed_ns"])
            value_errors.append(abs(a["logp"] - s["logp"]) / a["observations"])
            score_errors.append(max(abs(x-y)/max(1.0,abs(y)) for x,y in zip(a["scores"],s["scores"])))
        quality = max(value_errors) < 1e-5 and max(score_errors) < 5e-4
        summaries.append(dict(family=family, operation=operation, median_alea_over_stan=statistics.median(ratios),
                              ratio_range=[min(ratios), max(ratios)], max_logp_error_per_observation=max(value_errors),
                              max_scaled_score_error=max(score_errors), quality_check=quality, timing_eligible=quality))
    diagnostics = []
    for family in (5, 7):
        size = 1000 if family == 5 else 16
        base = [1.6, .2, .45, .7, .1, .05, .1]
        fd = []
        for coordinate in range(family):
            high, low = base.copy(), base.copy()
            h = 1e-4
            high[coordinate] += h
            low[coordinate] -= h
            values = [json.loads(run([output / "stan-worker", family, "batch_logp", 1, size, *p]))["logp"] for p in (high, low)]
            fd.append((values[0] - values[1]) / (2*h))
        control = {r["engine"]: r for r in rows if r["family"] == family and r["operation"] == "batch_direct" and r["pair"] == 0}
        vectorized = next(r for r in rows if r["family"] == family and r["operation"] == "batch_vectorized" and r["engine"] == "stan" and r["pair"] == 0)
        diagnostics.append(dict(family=family, observations=size, stan_primal_finite_difference=fd,
                                stan_scalar_sum=control["stan"]["scores"][:family],
                                alea=control["alea"]["scores"][:family], stan_vectorized=vectorized["scores"][:family]))
    report = dict(schema_version=1, purpose="descriptive operation-matched timing, not a promotion gate",
                  host=dict(system=platform.system(), machine=platform.machine(), processor=platform.processor()),
                  cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  stan_revision=revision, stan_source_archive_sha256=digest(output / "stan-source.tar.gz"),
                  alea_input_archive_sha256=digest(output / "alea-inputs.tar.gz"),
                  rustc=run(["rustc", "-vV"]), compiler=run([compiler, "--version"]), compiler_command=command,
                  cargo_command=build, features=args.features, rustflags=os.environ.get("RUSTFLAGS", ""),
                  native_environment={key: os.environ.get(key) for key in ("CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS")},
                  binaries={"alea": digest(alea), "stan": digest(output / "stan-worker")},
                  inputs_sealed_before_measurement=True,
                  tbb_sha256=digest(library / "libtbb.so.2"), cpp_inputs=dependencies,
                  rust_inputs={str(p.relative_to(root)): digest(p) for p in sources},
                  precision_policy={"alea4_5": 1e-12, "alea7": 1e-4,
                                    "stan4": "fixed density bound 1e-6", "stan5": "density 1e-6; derivatives requested 1e-12", "stan7": "derivatives 1e-4; internal density bounds differ"},
                  notes=["Builds/imports excluded; repeated fresh process pairs alternate order.",
                         "Batch Stan shares AD parameters and uses one reverse sweep. batch_vectorized separately tests its upper/lower vectorized API; other batch rows use scalar lpdf sums.",
                         "batch_buckets selects Alea's strategy; Stan uses the same scalar-sum batch target as batch_direct.",
                         "Failed vectorized gradient controls remain in the report and are ineligible for favorable timing claims; independent primal finite differences adjudicate them.",
                         "Precision policies differ by implementation; quality checks accompany timing, not matched-error certification.",
                         "Wiener4 Alea logp currently computes its fused score internally.",
                         "No derivative is taken with respect to observations; lower boundaries reflect beta/drift and their scores."],
                  rows=rows, summary=summaries, vectorization_diagnostics=diagnostics)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
