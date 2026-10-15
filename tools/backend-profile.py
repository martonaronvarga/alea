"""Build isolated backends with one compiler, then measure serially; no build overlaps timing.

Run in nix develop .#blas. All four rows use this same nightly compiler and flags.
Output directory must be new. Inner seconds exclude construction, outer time and
native per-process RSS include it. RSS is not heap allocation or retained-draw
storage. Low-rank rows are controls, not BLAS dispatch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time

ROWS = ('scalar', 'simd', 'faer', 'openblas')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--timer', default='time', help='GNU time executable (from the profiling shell)')
    args = parser.parse_args()
    timer = shutil.which(args.timer)
    if timer is None or 'GNU' not in subprocess.check_output([timer, '--version'], text=True):
        parser.error('GNU time is required for isolated child RSS')
    args.output.mkdir(parents=True, exist_ok=False)
    if platform.system() != 'Linux':
        raise RuntimeError('Linux is required for RSS units')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    result = dict(compiler=subprocess.check_output(['rustc', '-Vv'], text=True),
                  platform=platform.platform(), cpu=Path('/proc/cpuinfo').read_text(),
                  rustflags=env.get('RUSTFLAGS', ''), rows={}, samples=[],
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                      Path(__file__), Path('flake.lock'), Path('crates/Cargo.lock'),
                      Path('crates/alea-math/src/metric.rs'),
                      Path('crates/alea-math/src/metric/low_rank.rs'),
                      Path('crates/alea-mcmc/examples/metric_profile.rs'))})
    for name in ROWS:
        cmd = ['cargo', 'build', '--manifest-path', 'crates/Cargo.toml', '--locked',
               '-p', 'alea-mcmc', '--release', '--example', 'metric_profile']
        if name != 'scalar':
            cmd += ['--features', name]
        subprocess.run(cmd, check=True, env=env)
        binary = args.output / name
        shutil.copy2('crates/target/release/examples/metric_profile', binary)
        result['rows'][name] = dict(command=cmd, sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    # Two discarded warmups, six retained rounds; alternate execution order.
    for round_index in range(8):
        for name in (ROWS if round_index % 2 == 0 else ROWS[::-1]):
            for kind in ('dense', 'lowrank'):
                for dim in (8, 64, 256):
                    repetitions = 100000 if kind == 'lowrank' else max(1000, 1000000 // dim)
                    command = [str((args.output / name).resolve()), kind, str(dim), str(repetitions)]
                    start = time.perf_counter()
                    # GNU time forks after exec: Python's inherited high-water
                    # mark must not become the benchmark child's reported RSS.
                    with tempfile.TemporaryDirectory(prefix='alea-backend-rss-') as directory:
                        usage = Path(directory) / 'usage'
                        stdout = subprocess.check_output(
                            [timer, '-f', '%M', '-o', str(usage), *command], env=env, text=True)
                        rss = int(usage.read_text())
                    sample = dict(backend=name, kind=kind, dimension=dim, repetitions=repetitions,
                                  round=round_index-2, inner_seconds=float(stdout),
                                  wall_seconds=time.perf_counter()-start, peak_rss_kib=rss)
                    if round_index >= 2:
                        result['samples'].append(sample)
        print(f'completed round {round_index+1}/8', flush=True)
    with (args.output / 'results.json').open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
