#!/usr/bin/env python3
"""Serial fresh-process native posterior benchmark; every failure is retained.

Run in the repository's lock-pinned Nix stable shell. Compilation is a separate
step. Use fresh output directories; timings include startup and engine-specific
storage. Counts/timing instrumentation is explicit and costs are not subtracted.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
CASES = dict(mean="mean", linear="linear", logistic="logistic",
             linear_large="linear", logistic_large="logistic", funnel="funnel")
DIMENSIONS = dict(mean=2, linear=9, logistic=8, linear_large=9, logistic_large=8, funnel=9)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def run_process(command, **kwargs):
    """Retain launch failures as failed child attempts, not lost journal rows."""
    try:
        return subprocess.run(command, **kwargs)
    except OSError as error:
        return subprocess.CompletedProcess(command, 127, '', f'process launch failed: {error!r}')


def read_metrics(text):
    """Reject malformed/nonfinite child JSON before it can enter a journal."""
    def constant(value):
        raise ValueError(f'nonfinite JSON constant: {value}')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate metric: {key}')
            result[key] = value
        return result
    metrics = json.loads(text, parse_constant=constant, object_pairs_hook=unique)
    def validate(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('nonfinite JSON number')
        if isinstance(value, dict):
            for child in value.values(): validate(child)
        if isinstance(value, list):
            for child in value: validate(child)
    validate(metrics)
    if not isinstance(metrics, dict):
        raise ValueError('metrics must be an object')
    return metrics


def completed_run(row, chains):
    metrics = row.get('metrics', {})
    return (isinstance(metrics, dict) and row.get('success') is True and type(metrics.get('completed')) is int
            and metrics['completed'] == 1000 * chains
            and metrics.get('chains', chains) == chains
            and not metrics.get('failures')
            and finite_number(row.get('wall_seconds')) and row['wall_seconds'] > 0)


def statistical_qualification(row, plan=None):
    """Recompute eligibility from evidence, never trust a cached pass flag."""
    metrics = row.get('metrics', {})
    chains = (plan or {}).get('chains', metrics.get('chains'))
    if type(chains) is not int or chains < 2:
        return False, 'multiple chains required'
    if not completed_run(row, chains):
        return False, 'incomplete or failed run'
    dimension = DIMENSIONS.get(row.get('case'))
    diagnostics = metrics.get('diagnostics')
    if (not isinstance(diagnostics, list) or len(diagnostics) != dimension
            or metrics.get('dimension', dimension) != dimension):
        return False, 'missing or dimensionally incomplete diagnostics'
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict) or not all(
                finite_number(diagnostic.get(k)) for k in ['rhat', 'bulk_ess', 'tail_ess']):
            return False, 'missing or nonfinite diagnostics'
        if not (0 < diagnostic['rhat'] <= 1.01 and diagnostic['bulk_ess'] >= 100
                and diagnostic['tail_ess'] >= 100):
            return False, 'diagnostic thresholds failed'
    if type(metrics.get('retained_divergences')) is not int or metrics['retained_divergences'] != 0:
        return False, 'divergences or missing divergence count'
    return True, None


def command_output(command):
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError as error:
        return f'unavailable: {error}'
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(values):
    if not values:
        return None
    return dict(n=len(values), median=statistics.median(values),
                minimum=min(values), maximum=max(values),
                stdev=statistics.stdev(values) if len(values) > 1 else None)


def summarize(rows, plan=None):
    groups = {(case,engine):[] for case in plan['cases'] for engine in ['alea','stan']} if plan else {}
    for row in rows:
        groups.setdefault((row['case'], row['engine']), []).append(row)
    result = []
    for (case, engine), group in groups.items():
        valid = [r for r in group if r['success'] and
                 (not plan or 'chains' not in plan or completed_run(r,plan['chains']))]
        eligible = [r for r in valid if statistical_qualification(r, plan)[0]]
        result.append(dict(case=case, engine=engine, planned=plan['reps'] if plan else len(group), attempted=len(group),
                           unstarted=plan['reps']-len(group) if plan else 0,
                           successful=len(valid), failed=len(group)-len(valid),
                           diagnostic_passes=len(eligible),
                           statistical_ineligible_reasons=[statistical_qualification(r, plan)[1]
                               for r in group if not statistical_qualification(r, plan)[0]],
                           timing_scope='computational completion; not statistical efficiency',
                           wall_seconds=describe([r['wall_seconds'] for r in valid]),
                           evaluations_per_second=describe([r['metrics']['evaluations']/r['wall_seconds']
                              for r in valid if isinstance(r.get('metrics', {}).get('evaluations'), int)]),
                           evaluations_per_evaluation_second=describe([r['metrics']['evaluations']/r['metrics']['evaluation_seconds']
                              for r in valid if r.get('metrics', {}).get('evaluation_seconds')]),
                           iterations_per_second=describe([r['metrics']['completed']/r['wall_seconds']
                              for r in valid if 'completed' in r.get('metrics', {})]),
                           min_bulk_ess_per_second=describe([min(d['bulk_ess'] for d in r['metrics']['diagnostics'])/r['wall_seconds']
                              for r in eligible])))
    return result


def parse_stan(paths):
    steps = divergences = completed = 0
    depths = []
    phase_seconds = []
    for path in paths:
        text = path.read_text()
        lines = [l for l in text.splitlines() if l and not l.startswith('#')]
        columns = lines[0].split(',')
        if len(columns) != len(set(columns)):
            raise ValueError('duplicate Stan CSV column')
        if len(lines) - 1 != 1000:
            raise ValueError('incomplete Stan chain: expected 1000 retained draws')
        for line in lines[1:]:
            if len(line.split(',')) != len(columns):
                raise ValueError('Stan CSV row dimension mismatch')
            values = dict(zip(columns, map(float, line.split(','))))
            steps += int(values['n_leapfrog__'])
            divergences += int(values['divergent__'])
            depths.append(int(values['treedepth__']))
            completed += 1
        phase_seconds.extend(l for l in text.splitlines() if 'seconds (' in l)
    evaluations = 0
    evaluation_seconds = 0.
    for path in paths:
        with path.with_suffix('.profile.csv').open() as stream:
            for row in csv.DictReader(stream):
                if row['name'] == 'posterior':
                    evaluations += int(row['autodiff_calls'])
                    evaluation_seconds += float(row['total_time'])
    return dict(completed=completed, retained_leapfrog_steps=steps,
                retained_divergences=divergences, retained_mean_depth=statistics.mean(depths),
                retained_max_depth=max(depths), phase_times_csv_inclusive=phase_seconds,
                evaluations=evaluations, evaluation_seconds=evaluation_seconds,
                evaluation_scope="Stan profile posterior autodiff calls; total_time includes any primal calls and excludes transform/Jacobian outside the profile")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cmdstan', type=Path, required=True)
    p.add_argument('--alea', type=Path, default=ROOT/'target/release/alea-stan-bench')
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--cases', nargs='+', choices=list(CASES), default=list(CASES))
    p.add_argument('--reps', type=int, default=7)
    p.add_argument('--chains', type=int, default=4)
    p.add_argument('--cpu', type=int)
    p.add_argument('--timed', action='store_true')
    p.add_argument('--alea-csv', action='store_true')
    args = p.parse_args()
    if min(args.reps,args.chains) < 1:
        p.error('positive reps/chains required')
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    results = args.results.resolve()
    results.mkdir(parents=True, exist_ok=False)
    alea = args.alea.resolve()
    if not alea.is_file() or any(not (ROOT/'stan'/CASES[c]).is_file() for c in args.cases):
        p.error('build Alea and all requested Stan models before measurement')
    lock = json.loads((REPO/'flake.lock').read_text())
    manifest = dict(alea_commit=command_output(['git','-C',str(REPO),'rev-parse','HEAD']),
                    source_diff=command_output(['git','-C',str(REPO),'diff','--binary']),
                    source_sha256={str(f.relative_to(REPO)):sha(f) for directory in [REPO/'crates',ROOT]
                        for f in directory.rglob('*') if f.is_file() and 'target' not in f.parts
                        and f.suffix in ('.rs','.toml','.py','.stan')},
                    binary_sha256=sha(alea), cargo_lock_sha256=sha(ROOT/'Cargo.lock'),
                    stan_binary_sha256={model:sha(ROOT/'stan'/model) for model in set(CASES[c] for c in args.cases)},
                    pins=json.loads((ROOT/'PINNED.json').read_text()),
                    flake_lock_sha256=sha(REPO/'flake.lock'), nixpkgs=lock['nodes']['nixpkgs']['locked'],
                    cmdstan_path=str(args.cmdstan.resolve()), cmdstan_version=command_output([str(args.cmdstan/'bin/stanc'),'--version']),
                    platform=platform.platform(), machine=platform.machine(), cpuinfo=Path('/proc/cpuinfo').read_text(),
                    affinity=sorted(os.sched_getaffinity(0)), rustc=command_output(['rustc','-vV']),
                    runtime_cxx=command_output(['g++','--version']),
                    stan_binary_compiler_comments={model:command_output(['readelf','-p','.comment',str(ROOT/'stan'/model)])
                        for model in set(CASES[c] for c in args.cases)},
                    flags={k:os.environ.get(k) for k in ['RUSTFLAGS','CXXFLAGS','CARGO_ENCODED_RUSTFLAGS']},
                    chains=args.chains, reps=args.reps, cases=args.cases,total_planned=2*args.reps*len(args.cases), warmup=1000, draws=1000, max_depth=10,
                    target_acceptance=0.8, step_size=0.2, data_seed=20261029, alea_seed_bytes=[17]*32,
                    stan_seed=17, initial_free_coordinates=0, timed=args.timed,
                    storage='Alea collects in memory, no serialization; Stan writes retained CSV (preserved). Process comparison includes this asymmetry.',
                    schedule='Alea sequential chains in one process; Stan sequential chain processes; one CPU',
                    data={c:sha(ROOT/'data'/f'{c}.txt') for c in args.cases})
    (results/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    env = dict(os.environ, OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
               RAYON_NUM_THREADS='1',STAN_NUM_THREADS='1',ALEA_BENCH_CHAINS=str(args.chains))
    if args.timed:
        env['ALEA_BENCH_TIMED'] = '1'
    rows = []
    (results/'summary.json').write_text(json.dumps(summarize(rows,manifest),indent=2)+'\n')
    with (results/'records.jsonl').open('x') as output:
        for case in args.cases:
            model = CASES[case]
            for rep in range(args.reps):
                for engine in (('alea','stan') if rep%2==0 else ('stan','alea')):
                    label = f'{case}-{rep}-{engine}'
                    commands = [[str(alea),model,str(ROOT/'data'/f'{case}.txt')]]
                    paths = []
                    if engine == 'stan':
                        commands = []
                        for chain in range(args.chains):
                            path = results/f'{label}-{chain}.csv'
                            paths.append(path)
                            commands.append([str(ROOT/'stan'/model),'sample','num_warmup=1000','num_samples=1000',
                                'adapt','delta=0.8','algorithm=hmc','engine=nuts','max_depth=10',
                                'metric=diag_e','stepsize=0.2','data',f'file={ROOT/"data"/f"{case}.json"}',
                                'random','seed=17',f'id={chain+1}','init=0','output',f'file={path}',f'profile_file={path.with_suffix(".profile.csv")}','sig_figs=17','refresh=0'])
                    started = time.perf_counter_ns()
                    run_env = {**env,'ALEA_BENCH_CSV':str(results/f'{label}.csv')} if engine=='alea' and args.alea_csv else env
                    processes = [run_process(c,env=run_env,text=True,capture_output=True,check=False) for c in commands]
                    wall = (time.perf_counter_ns()-started)/1e9
                    for i,process in enumerate(processes):
                        (results/f'{label}-{i}.stdout').write_text(process.stdout)
                        (results/f'{label}-{i}.stderr').write_text(process.stderr)
                    success = all(c.returncode==0 for c in processes)
                    row = dict(case=case,engine=engine,rep=rep,wall_seconds=wall,success=success,
                               commands=commands,returncodes=[c.returncode for c in processes])
                    try:
                        if engine == 'alea':
                            row['metrics'] = read_metrics(processes[0].stdout.splitlines()[-1])
                            row['success'] &= row['metrics']['completed']==1000*args.chains and not row['metrics']['failures']
                        elif success:
                            row['metrics'] = parse_stan(paths)
                            diag = run_process([str(alea),model,str(ROOT/'data'/f'{case}.txt')],
                                env={**env,'ALEA_BENCH_STAN_CSVS':':'.join(map(str,paths))},text=True,capture_output=True,check=False)
                            if diag.returncode==0:
                                row['metrics'].update(read_metrics(diag.stdout))
                            else:
                                row['diagnostic_error'] = diag.stderr
                        row['success'] = completed_run(row, args.chains)
                    except (ValueError,KeyError,IndexError,OSError,TypeError) as error:
                        row['success'] = False
                        row['parse_error'] = repr(error)
                    row['diagnostic_pass'], row['diagnostic_reason'] = statistical_qualification(row, manifest)
                    if not row['success']:
                        row['failure_reason'] = row.get('parse_error', 'nonzero child exit or incomplete retained draws')
                    rows.append(row)
                    output.write(json.dumps(row)+'\n')
                    output.flush()
                    (results/'summary.json').write_text(json.dumps(summarize(rows,manifest),indent=2)+'\n')
                    print(label, 'complete' if row['success'] else 'FAILED', f'{wall:.6f}s',flush=True)


if __name__ == '__main__':
    main()
