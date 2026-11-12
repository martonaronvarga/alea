#!/usr/bin/env bash
set -euo pipefail

# Criterion owns samples and baseline comparisons. Build every cell before any
# timings; separate target directories prevent feature rebuilds between runs.
# A single tee writer owns each shared-filesystem log; pipefail retains failures.
manifest=crates/Cargo.toml
output=target/criterion_features
no_run=false
smoke=false
threads=1
packages=()
benches=()
tests=()
feature_sets=()
criterion_args=()
usage() {
  printf '%s\n' 'Usage: tools/bench-features.sh [--no-run|--smoke] [--manifest PATH] [--output PATH]' \
    '       [--crate NAME]... [--bench NAME|--test NAME]... [--feature-set "FEATURES"]... [--threads N]' \
    '       [-- TARGET_ARGS...]' \
    'Default: all native Criterion targets, default feature cell, one thread.' \
    'Timings require a Slurm compute allocation. --smoke runs fixtures once without timing.' \
    '--test runs release correctness tests and requires --smoke or --no-run.'
}
while (($#)); do
  case "$1" in
    --no-run) no_run=true; shift ;;
    --smoke) smoke=true; shift ;;
    --manifest) manifest=${2:?}; shift 2 ;;
    --output) output=${2:?}; shift 2 ;;
    --crate) packages+=(-p "${2:?}"); shift 2 ;;
    --bench) benches+=(--bench "${2:?}"); shift 2 ;;
    --test) tests+=(--test "${2:?}"); shift 2 ;;
    --feature-set) feature_sets+=("${2?feature set required (may be empty)}"); shift 2 ;;
    --threads) threads=${2:?}; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; criterion_args=("$@"); break ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
command_kind=bench
if ((${#tests[@]})); then
  ((${#benches[@]} == 0)) || { printf 'choose either bench or test targets\n' >&2; exit 2; }
  $smoke || $no_run || { printf 'test targets require --smoke or --no-run\n' >&2; exit 2; }
  command_kind=test
fi
cargo_mode=("$command_kind")
[[ $command_kind != test ]] || cargo_mode+=(--release)
[[ $threads =~ ^[1-9][0-9]*$ ]] || { printf 'positive thread count required\n' >&2; exit 2; }
run_prefix=()
cpu_list=not-timed
if ! $no_run && ! $smoke; then
  [[ -n ${SLURM_JOB_ID:-} && -n ${SLURMD_NODENAME:-} && ${SLURMD_NODENAME} != login && $(hostname -s) == "$SLURMD_NODENAME" ]] || {
    printf 'Timing requires a Slurm compute node; use --no-run or --smoke locally.\n' >&2; exit 2;
  }
  ((threads <= ${SLURM_CPUS_PER_TASK:-1})) || { printf 'thread budget exceeds allocation\n' >&2; exit 2; }
  cpu_list=$("${PYTHON:-python3}" -c 'import os,sys; a=sorted(os.sched_getaffinity(0)); n=int(sys.argv[1]); sys.exit("thread budget exceeds affinity") if n>len(a) else None; print(",".join(map(str,a[:n])))' "$threads")
  run_prefix=(taskset -c "$cpu_list")
fi
((${#feature_sets[@]})) || feature_sets=("")
((${#packages[@]})) || packages=(--workspace)
targets=("${benches[@]}")
if [[ $command_kind == test ]]; then targets=("${tests[@]}"); fi
((${#targets[@]})) || targets=(--benches)
mkdir -p "$output"
output=$(cd "$output" && pwd)
export OPENBLAS_NUM_THREADS=$threads OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads
export RAYON_NUM_THREADS=$threads STAN_NUM_THREADS=$threads ALEA_BENCH_THREADS=$threads
succeeded=()
failed=0
for i in "${!feature_sets[@]}"; do
  features=${feature_sets[i]}
  cell="$output/cell-$i"
  mkdir "$cell" # Never overwrite another run's raw samples.
  printf '%s\n' "$features" > "$cell/features.txt"
  { uname -a; rustc -Vv; cargo -V; printf 'RUSTFLAGS=%s\nTHREADS=%s\nSLURM_JOB_ID=%s\n' "${RUSTFLAGS:-}" "$threads" "${SLURM_JOB_ID:-}"; } > "$cell/environment.txt"
  printf 'MEASUREMENT_CPUS=%s\n' "$cpu_list" >> "$cell/environment.txt"
  if command -v taskset >/dev/null; then taskset -pc $$ >> "$cell/environment.txt"; fi
  if command -v lscpu >/dev/null; then lscpu >> "$cell/environment.txt"; fi
  extra=()
  [[ -z $features ]] || extra=(--features "$features")
  target_dir="$cell/build"
  # Correctness-only cells may reuse Cargo's normal feature-keyed cache. Timed
  # benchmarks retain isolated builds, so no rebuild enters a measurement phase.
  if [[ $command_kind == test && -n ${CARGO_TARGET_DIR:-} ]]; then target_dir=$CARGO_TARGET_DIR; fi
  printf '%s\n' "$target_dir" > "$cell/build-target.txt"
  if CARGO_TARGET_DIR="$target_dir" cargo "${cargo_mode[@]}" --manifest-path "$manifest" --locked \
       "${packages[@]}" "${targets[@]}" --no-default-features "${extra[@]}" --no-run \
       2>&1 | tee "$cell/build.log" > /dev/null; then
    succeeded+=("$i")
  else
    printf 'build failed: cell-%s (%s); see %s/build.log\n' "$i" "$features" "$cell" >&2
    touch "$cell/BUILD_FAILED"
    failed=1
  fi
done
$no_run && exit "$failed"
for i in "${succeeded[@]}"; do
  cell="$output/cell-$i"
  features=${feature_sets[i]}
  extra=()
  [[ -z $features ]] || extra=(--features "$features")
  args=("${criterion_args[@]}")
  if $smoke; then
    if [[ $command_kind == test ]]; then args+=(--show-output); else args+=(--test); fi
  fi
  target_dir=$(cat "$cell/build-target.txt")
  if ! CARGO_TARGET_DIR="$target_dir" CRITERION_HOME="$cell/criterion" \
       "${run_prefix[@]}" cargo "${cargo_mode[@]}" --manifest-path "$manifest" --locked "${packages[@]}" "${targets[@]}" \
       --no-default-features "${extra[@]}" -- "${args[@]}" 2>&1 | tee "$cell/run.log" > /dev/null; then
    touch "$cell/RUN_FAILED"
    failed=1
  fi
done
exit "$failed"
