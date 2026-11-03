#!/usr/bin/env python3
"""Source-sealed repeated CPU-plan candidates; no fitted production dispatch."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tarfile
import time

from bench import REPO, command_output, describe, finite_number, read_metrics, sha, host_metadata

sys.path.insert(0, str(REPO / 'tools/release'))
import source as source_archive

GRIDS = {'calibration': ([3, 1000, 10000], [0, 1, 4, 16, 64]),
         'validation': ([1, 7, 127, 2049], [2, 7, 31]), 'smoke': ([1000], [4])}
MODES = ('Scalar', 'Auto')
PHASES = ('build', 'evaluate')
VERSION = 2
THREADS = {key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                               'MKL_NUM_THREADS', 'RAYON_NUM_THREADS')}


def cases(suite):
    rows, columns = GRIDS[suite]
    return [(family, n, p) for n in rows for p in columns
            for family in ('normal', 'logit', 'poisson')]


def construction_scope(stderr):
    scopes = [line.split(' ', 1)[1] for line in stderr.splitlines()
              if line.startswith('ALEA_CPU_PLAN_BUILD_SCOPE ')]
    if not scopes:
        return None  # Historical receipts/reports remain byte-for-byte unchanged.
    if scopes != ['shared-prepared-data-v2']:
        raise ValueError('invalid or duplicate construction scope')
    return scopes[0]


def receipts(stderr, suite):
    result = {}
    begun = set()
    for line in stderr.splitlines():
        if line.startswith('ALEA_CPU_PLAN_BEGIN '):
            _, family, n, p = line.split()
            case = (family, int(n), int(p))
            if case in begun or case not in cases(suite):
                raise ValueError('duplicate or unexpected case start')
            begun.add(case)
            continue
        if not line.startswith('ALEA_CPU_PLAN_VALUE '):
            continue
        _, family, n, p, mode, value, score = line.split(maxsplit=6)
        key = (family, int(n), int(p), mode)
        if key in result or key[:3] not in begun or mode not in MODES:
            raise ValueError('duplicate or unexpected numerical receipt')
        gradient = read_metrics('{"gradient":' + score + '}')['gradient']
        logp = read_metrics('{"value":' + value + '}')['value']
        if (not finite_number(logp) or not isinstance(gradient, list)
                or len(gradient) != int(p) + (family == 'normal')
                or not all(finite_number(v) for v in gradient)):
            raise ValueError('nonfinite or dimensionally incomplete numerical receipt')
        result[key] = [logp, *gradient]
    return begun, result


def numeric_pair(values, case):
    a, b = (values.get((*case, mode)) for mode in MODES)
    if a is None or b is None:
        raise ValueError('missing numerical candidate')
    if not all(abs(x - y) <= 3e-12 * (1 + abs(x)) for x, y in zip(a, b)):
        raise ValueError('value/score candidates disagree')


def estimate(folder, phase, case, mode, samples):
    family, n, p = case
    path = folder / 'raw' / f'regression_plan_{phase}_{family}_{n}_{p}' / mode / 'new'
    metadata = read_metrics((path / 'benchmark.json').read_text())
    expected = f'regression_plan/{phase}/{family}/{n}/{p}/{mode}'
    if (metadata['full_id'] != expected or metadata['group_id'] != expected.rsplit('/', 1)[0]
            or metadata['function_id'] != mode):
        raise ValueError('Criterion identity mismatch')
    if phase == 'evaluate' and metadata['throughput'] != {'Elements': 1}:
        raise ValueError('incorrect evaluation throughput unit')
    sample = read_metrics((path / 'sample.json').read_text())
    if (not isinstance(sample['iters'], list) or not isinstance(sample['times'], list)
            or len(sample['iters']) != len(sample['times']) or len(sample['iters']) != samples
            or not all(finite_number(v) and float(v).is_integer() for v in sample['iters'])
            or not all(finite_number(v) and v > 0 for v in sample['iters'] + sample['times'])):
        raise ValueError('invalid Criterion work samples')
    mean = read_metrics((path / 'estimates.json').read_text())['mean']
    ci = mean['confidence_interval']
    point, low, high = (mean['point_estimate'], ci['lower_bound'], ci['upper_bound'])
    if (not all(finite_number(v) and v > 0 for v in (point, low, high))
            or not low <= point <= high or ci['confidence_level'] != 0.95):
        raise ValueError('invalid Criterion estimate')
    return dict(ns=point, within_process_ci95=[low, high])


def run_files(folder):
    return {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and p != folder / 'run.json'}


def collect(folder, suite, rep, manifest):
    """Admit complete cases independently, preserving earlier cases on child failure."""
    error = None
    try:
        run = read_metrics((folder / 'run.json').read_text())
        if (run['schema_version'] != VERSION or type(run['schema_version']) is not int
                or type(run['rep']) is not int or run['rep'] != rep or run['suite'] != suite
                or type(run['returncode']) is not int
                or run.get('sealed') is not True or run.get('inputs_verified') is not True
                or any(run.get(key) != manifest[key] for key in ('binary_sha256', 'source_archive_sha256'))
                or run['files'] != run_files(folder)):
            raise ValueError('unsealed, changed or misidentified run')
        stderr = (folder / 'stderr.txt').read_text()
        scope = construction_scope(stderr)
        begun, values = receipts(stderr, suite)
    except (OSError, ValueError, KeyError, TypeError) as cause:
        error = str(cause)
        run = {}
        values = {}
        begun = set()
        scope = None
    rows = []
    for case in cases(suite):
        row = dict(family=case[0], rows=case[1], columns=case[2], rep=rep,
                   success=False, attempted=error is not None or case in begun,
                   process_returncode=run.get('returncode'))
        if scope is not None:
            row['construction_scope'] = scope
        try:
            if error is not None:
                raise ValueError(error)
            if case not in begun:
                raise ValueError('case not started before child stopped')
            numeric_pair(values, case)
            row['estimates'] = {phase: {mode: estimate(folder, phase, case, mode, manifest['criterion']['samples'])
                                      for mode in MODES} for phase in PHASES}
            row['values'] = {mode: values[(*case, mode)] for mode in MODES}
            row['success'] = True
        except (OSError, ValueError, KeyError, TypeError) as cause:
            row['failure_reason'] = str(cause)
        rows.append(row)
    return rows


def summarize(rows, suite, reps):
    output = []
    for case in cases(suite):
        attempts = [r for r in rows if (r['family'], r['rows'], r['columns']) == case]
        if len({r['rep'] for r in attempts}) != len(attempts):
            raise ValueError('duplicate repetition')
        valid = [r for r in attempts if r['success']]
        item = dict(family=case[0], rows=case[1], columns=case[2], planned=reps,
                    successful=len(valid), failed=sum(r['attempted'] and not r['success'] for r in attempts),
                    unstarted=reps - sum(r['attempted'] for r in attempts), phases={}, amortized={})
        for phase in PHASES:
            def ns(row, mode):
                return row['estimates'][phase][mode]['ns']
            item['phases'][phase] = dict(
                ns={mode: describe([ns(r, mode) for r in valid]) for mode in MODES},
                auto_over_scalar=describe([ns(r, 'Auto') / ns(r, 'Scalar') for r in valid]),
                auto_regret=describe([ns(r, 'Auto') / min(ns(r, m) for m in MODES) - 1 for r in valid]))
        item['evaluations_per_second'] = {
            mode: describe([1e9 / r['estimates']['evaluate'][mode]['ns'] for r in valid]) for mode in MODES}
        for calls in (1, 10, 1000):
            costs = [{mode: r['estimates']['build'][mode]['ns']
                      + calls * r['estimates']['evaluate'][mode]['ns'] for mode in MODES} for r in valid]
            item['amortized'][str(calls)] = dict(
                auto_over_scalar=describe([v['Auto'] / v['Scalar'] for v in costs]),
                auto_regret=describe([v['Auto'] / min(v.values()) - 1 for v in costs]))
        output.append(item)
    return output


def rebuild(output):
    manifest = read_metrics((output / 'manifest.json').read_text())
    if (type(manifest['schema_version']) is not int or manifest['schema_version'] != VERSION
            or manifest['suite'] not in GRIDS or type(manifest['reps']) is not int or manifest['reps'] < 1
            or sha(output / 'benchmark') != manifest['binary_sha256']
            or sha(output / 'sources.tar.gz') != manifest['source_archive_sha256']):
        raise ValueError('invalid campaign or changed source/binary')
    rows = []
    for rep in range(manifest['reps']):
        folder = output / f'rep-{rep:02}'
        if folder.exists():
            rows.extend(collect(folder, manifest['suite'], rep, manifest))
    report = dict(schema_version=VERSION, scope='native kernel candidates; not old/new Alea, Stan or inference efficiency',
                estimator='Criterion mean ns/call; median/min/max/stdev across independent processes',
                records=rows, summary=summarize(rows, manifest['suite'], manifest['reps']))
    scopes = {r.get('construction_scope', 'private-pack-per-descriptor-v1') for r in rows if r['success']}
    if len(scopes) > 1:
        raise ValueError('mixed construction scopes in one campaign')
    if scopes == {'shared-prepared-data-v2'}:
        report['construction_scope'] = 'shared-prepared-data-v2; excludes fresh input allocation/packing'
    return report


def compare_campaigns(before, after):
    """Recompute both endpoints; require exact value/score bits, not cached flags."""
    reports = [rebuild(path) for path in (before, after)]
    manifests = [read_metrics((path / 'manifest.json').read_text()) for path in (before, after)]
    for key in ('suite', 'reps', 'cpu', 'rustc', 'flags', 'threads', 'criterion'):
        if manifests[0].get(key) != manifests[1].get(key):
            raise ValueError(f'incompatible campaign setting: {key}')
    # Scaling MHz is transient; compare the stable CPU identity lines only.
    def identity(manifest):
        return [line for line in manifest.get('cpuinfo', '').splitlines()
                if line.split(':', 1)[0].strip() in ('vendor_id', 'model name', 'cpu family', 'model', 'stepping', 'flags')]
    if identity(manifests[0]) != identity(manifests[1]):
        raise ValueError('incompatible CPU identity')
    paired = []
    for a, b in zip(reports[0]['records'], reports[1]['records']):
        if any(a[k] != b[k] for k in ('family', 'rows', 'columns', 'rep')):
            raise ValueError('incompatible or partial repetition identities')
        if a['success'] and b['success'] and a.get('construction_scope', 'private-pack-per-descriptor-v1') != b.get('construction_scope', 'private-pack-per-descriptor-v1'):
            raise ValueError('incompatible construction scope; shared-data binding is not fresh packing')
        success = a['success'] and b['success']
        if success:
            success = all([struct.pack('>d', v) for v in a['values'][mode]]
                          == [struct.pack('>d', v) for v in b['values'][mode]] for mode in MODES)
        paired.append(dict(family=a['family'], rows=a['rows'], columns=a['columns'], rep=a['rep'],
                           success=success, before=a, after=b))
    if len(reports[0]['records']) != len(reports[1]['records']):
        raise ValueError('incompatible or partial repetition identities')
    output = []
    for case in cases(manifests[0]['suite']):
        rows = [r for r in paired if (r['family'], r['rows'], r['columns']) == case]
        valid = [r for r in rows if r['success']]
        phases = {}
        for phase in PHASES:
            phases[phase] = {mode: dict(
                before_ns=describe([r['before']['estimates'][phase][mode]['ns'] for r in valid]),
                after_ns=describe([r['after']['estimates'][phase][mode]['ns'] for r in valid]),
                paired_after_over_before=describe([r['after']['estimates'][phase][mode]['ns']
                    / r['before']['estimates'][phase][mode]['ns'] for r in valid])) for mode in MODES}
        output.append(dict(family=case[0], rows=case[1], columns=case[2],
                           qualified=len(valid), withheld=len(rows) - len(valid), phases=phases))
    return dict(schema_version=VERSION,
                scope='sequential source-sealed cohorts, not interleaved old/new processes; no inference claim',
                summary=output)


def unchanged(entries):
    paths = {p.relative_to(REPO).as_posix() for p in source_archive.source_files(REPO)}
    return (paths == {e['path'] for e in entries}
            and all(sha(REPO / e['path']) == e['sha256']
                    and (0o755 if (REPO / e['path']).stat().st_mode & 0o111 else 0o644) == e['mode']
                    for e in entries))


def campaign(args):
    output = args.output.resolve()
    if output.is_relative_to(REPO):
        raise ValueError('store campaigns outside the source checkout')
    if args.cpu not in os.sched_getaffinity(0):
        raise ValueError('CPU is outside permitted affinity')
    output.mkdir(parents=True, exist_ok=False)
    archive_sha = source_archive.archive(REPO, output / 'sources.tar.gz')
    with tarfile.open(output / 'sources.tar.gz') as archive:
        entries = json.load(archive.extractfile('alea/SOURCE-MANIFEST.json'))['files']
    build_command = ['cargo', 'bench', '--manifest-path', str(REPO / 'crates/Cargo.toml'),
                     '-p', 'alea-distributions', '--bench', 'regression_execution',
                     '--locked', '--no-run', '--message-format=json']
    start = time.perf_counter()
    build = subprocess.run(build_command, text=True, capture_output=True, check=False)
    (output / 'build.stdout.txt').write_text(build.stdout)
    (output / 'build.stderr.txt').write_text(build.stderr)
    if build.returncode != 0 or not unchanged(entries):
        raise ValueError('build failed or source changed during compilation')
    executables = {v['executable'] for line in build.stdout.splitlines()
                   if (v := json.loads(line)).get('reason') == 'compiler-artifact'
                   and v['target']['name'] == 'regression_execution' and v.get('executable')}
    if len(executables) != 1:
        raise ValueError('expected exactly one Criterion executable')
    shutil.copy2(executables.pop(), output / 'benchmark')
    manifest = dict(schema_version=VERSION, suite=args.suite, reps=args.reps, cpu=args.cpu, host=host_metadata(),
                    cpuinfo=Path('/proc/cpuinfo').read_text(), rustc=command_output(['rustc', '-vV']),
                    commit=command_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD']),
                    binary_sha256=sha(output / 'benchmark'), source_archive_sha256=archive_sha,
                    build_command=build_command, build_seconds=time.perf_counter() - start,
                    criterion=dict(samples=args.samples, measure=args.measure, warmup=args.warmup, resamples=1000),
                    flags={k: os.environ.get(k) for k in ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS')},
                    threads=THREADS, protocol='alternating candidate order; fixed synthetic states; separate build/eval; no fitted threshold')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    os.sched_setaffinity(0, {args.cpu})
    process_failure = False
    for rep in range(args.reps):
        if not unchanged(entries) or sha(output / 'benchmark') != manifest['binary_sha256']:
            raise ValueError('source/binary changed during campaign')
        folder = output / f'rep-{rep:02}'
        folder.mkdir()
        env = {**os.environ, **THREADS, 'CRITERION_HOME': str(folder / 'raw')}
        for key in ('ALEA_REGRESSION_PLAN_QUICK', 'ALEA_REGRESSION_PLAN_HELD_OUT', 'ALEA_REGRESSION_PLAN_REVERSE'):
            env.pop(key, None)
        if args.suite == 'smoke': env['ALEA_REGRESSION_PLAN_QUICK'] = '1'
        if args.suite == 'validation': env['ALEA_REGRESSION_PLAN_HELD_OUT'] = '1'
        if rep % 2: env['ALEA_REGRESSION_PLAN_REVERSE'] = '1'
        command = [str(output / 'benchmark'), '--bench', '--noplot', '--sample-size', str(args.samples),
                   '--measurement-time', str(args.measure), '--warm-up-time', str(args.warmup), '--nresamples', '1000']
        run = dict(schema_version=VERSION, rep=rep, suite=args.suite, command=command, sealed=False,
                   binary_sha256=manifest['binary_sha256'], source_archive_sha256=manifest['source_archive_sha256'])
        (folder / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
        interrupted = False
        start = time.perf_counter()
        with (folder / 'stdout.txt').open('x') as stdout, (folder / 'stderr.txt').open('x') as stderr:
            try:
                process = subprocess.run(command, stdout=stdout, stderr=stderr, env=env, check=False, timeout=args.timeout)
                run['returncode'] = process.returncode
            except (OSError, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                run.update(returncode=124, error=repr(error))
                interrupted = isinstance(error, KeyboardInterrupt)
        run.update(seconds=time.perf_counter() - start, sealed=True, files=run_files(folder),
                   inputs_verified=unchanged(entries) and sha(output / 'benchmark') == manifest['binary_sha256'])
        process_failure |= run['returncode'] != 0
        (folder / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
        report = rebuild(output)
        (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'{args.suite} repetition {rep}: exit={run["returncode"]}, {run["seconds"]:.2f}s', flush=True)
        if interrupted:
            break
    if not unchanged(entries):
        raise ValueError('source changed during campaign; keep raw data, withhold results')
    return 0 if not process_failure and all(r['successful'] == args.reps for r in report['summary']) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--replay', action='store_true', help='derive a NEW report on stdout; never modify stored artifacts')
    parser.add_argument('--compare-before', type=Path, help='compare sealed cohorts on stdout; requires identical numerical bits')
    parser.add_argument('--suite', choices=GRIDS, default='calibration')
    parser.add_argument('--reps', type=int, default=7)
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--measure', type=float, default=0.12)
    parser.add_argument('--warmup', type=float, default=0.03)
    parser.add_argument('--timeout', type=float, default=300)
    args = parser.parse_args()
    if (args.reps < 1 or args.samples < 10
            or not all(math.isfinite(v) and v > 0 for v in (args.measure, args.warmup, args.timeout))):
        parser.error('positive finite budgets, repetitions and at least ten samples required')
    if args.compare_before:
        print(json.dumps(compare_campaigns(args.compare_before, args.output), indent=2))
        return 0
    if args.replay:
        print(json.dumps(rebuild(args.output), indent=2))
        return 0
    return campaign(args)


if __name__ == '__main__':
    sys.exit(main())
