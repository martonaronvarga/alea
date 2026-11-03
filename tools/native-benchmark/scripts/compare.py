#!/usr/bin/env python3
"""Alternating paired native before/after runs with work/replay checks.

Fixed-work evaluation and inference are different experiments. Failed fits never
enter timing ratios. This is a reproducibility/performance regression harness;
CI should assert work/checksums, not a universal hardware timing threshold.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time
from bench import (CASES, ROOT, describe, sha, completed_run, finite_number,
                   read_metrics, statistical_qualification, run_process)


def performance_regressions(summary, max_slowdown):
    """Optional hardware-specific alert; never infer a time from a failed fit."""
    if max_slowdown is None:
        return []
    return [dict(mode=row['mode'],case=row['case'],slowdown=1/row['baseline_over_current']['median'])
            for row in summary if row['baseline_over_current']
            and 1/row['baseline_over_current']['median'] > max_slowdown]


def compare_work(a, b, mode, allow_trajectory_change=False):
    keys = (['checksum', 'evaluations'] if mode == 'evaluation' else
            ['trajectory_hash', 'evaluations', 'retained_leapfrog_steps', 'retained_divergences'])
    equal = {k: a[k] == b[k] for k in keys}
    if not allow_trajectory_change:
        if not all(equal.values()):
            raise AssertionError((mode, a, b))
    elif mode == 'evaluation':
        if not (equal['evaluations'] and all(finite_number(r['checksum']) for r in [a,b])
                and math.isclose(a['checksum'], b['checksum'], rel_tol=1e-12, abs_tol=1e-7)):
            raise AssertionError((a, b))
    return equal


def successful(row, chains):
    if row.get('returncode') != 0 or not finite_number(row.get('wall_seconds')) or row['wall_seconds'] <= 0:
        return False
    metrics = row.get('metrics', {})
    if row['mode'] == 'sampling':
        return (completed_run(dict(row, success=True), chains)
                and type(metrics.get('evaluations')) is int and metrics['evaluations'] > 0
                and all(type(metrics.get(k)) is int and metrics[k] >= 0
                        for k in ['retained_leapfrog_steps','retained_divergences'])
                and isinstance(metrics.get('trajectory_hash'), str))
    return (type(metrics.get('evaluations')) is int and metrics['evaluations'] == 10000
            and finite_number(metrics.get('evaluation_seconds')) and metrics['evaluation_seconds'] > 0
            and finite_number(metrics.get('checksum')))


def summarize(records, plan, decisions):
    """Fold the append-only journals; unvalidated pairs never yield ratios."""
    summary = []
    for mode in plan['modes']:
        for case in plan['cases']:
            if mode == 'evaluation' and case in ['linear_large', 'logistic_large', 'funnel']:
                continue
            group = [r for r in records if (r['case'], r['mode']) == (case, mode)]
            engines = {}
            for engine in ['baseline', 'current']:
                attempts = [r for r in group if r['engine'] == engine]
                valid = [r for r in attempts if successful(r, plan['chains'])]
                eligible = [r for r in valid if mode == 'sampling' and
                    statistical_qualification(dict(r, success=True), plan)[0]]
                engines[engine] = dict(planned=plan['reps'], attempts=len(attempts),
                    unstarted=plan['reps']-len(attempts), successes=len(valid), failed=len(attempts)-len(valid),
                    diagnostic_passes=len(eligible),
                    statistical_ineligible_reasons=[statistical_qualification(dict(r,success=successful(r,plan['chains'])),plan)[1]
                        for r in attempts if mode == 'sampling' and not statistical_qualification(dict(r,success=successful(r,plan['chains'])),plan)[0]],
                    seconds=describe([r['metrics']['evaluation_seconds'] if mode == 'evaluation' else r['wall_seconds'] for r in valid]),
                    evaluations_per_second=describe([r['metrics']['evaluations'] / (r['metrics']['evaluation_seconds'] if mode == 'evaluation' else r['wall_seconds']) for r in valid]),
                    min_bulk_ess_per_second=describe([min(d['bulk_ess'] for d in r['metrics']['diagnostics'])/r['wall_seconds'] for r in eligible]))
            ratios, ess_ratios = [], []
            for rep in range(plan['reps']):
                pair = {r['engine']: r for r in group if r['rep'] == rep}
                validation = [d for d in decisions if (d['mode'],d['case'],d['rep']) == (mode,case,rep)]
                if len(pair) != 2 or len(validation) != 1 or validation[0].get('compatible') is not True:
                    continue
                if not all(successful(r, plan['chains']) for r in pair.values()):
                    continue
                times = {k:r['metrics']['evaluation_seconds'] if mode == 'evaluation' else r['wall_seconds'] for k,r in pair.items()}
                ratios.append(times['baseline']/times['current'])
                if mode == 'sampling' and all(statistical_qualification(dict(r,success=True),plan)[0] for r in pair.values()):
                    ess = {k:min(d['bulk_ess'] for d in r['metrics']['diagnostics'])/r['wall_seconds'] for k,r in pair.items()}
                    ess_ratios.append(ess['current']/ess['baseline'])
            summary.append(dict(mode=mode,case=case,engines=engines,
                timing_scope='individual computational completion; paired ratios require work validation; not statistical efficiency',
                baseline_over_current=describe(ratios), current_over_baseline_ess_per_second=describe(ess_ratios)))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--current',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reps',type=int,default=7)
    p.add_argument('--cpu',type=int,default=2)
    p.add_argument('--chains',type=int,default=1)
    p.add_argument('--modes',nargs='+',choices=['evaluation','sampling'],default=['evaluation','sampling'])
    p.add_argument('--cases',nargs='+',choices=list(CASES),default=list(CASES))
    p.add_argument('--allow-trajectory-change',action='store_true',help='Record changed arithmetic/work explicitly; still validate fixed-work checksums')
    p.add_argument('--max-slowdown',type=float,help='Optional calibrated timing alert; separate from correctness CI')
    args = p.parse_args()
    if min(args.reps,args.chains) < 1 or (args.max_slowdown is not None and
                       (not math.isfinite(args.max_slowdown) or args.max_slowdown<=0)):
        p.error('positive repetitions and finite positive slowdown limit required')
    args.output.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,{args.cpu})
    binaries = dict(baseline=args.baseline.resolve(), current=args.current.resolve())
    plan = dict(schema_version=2,reps=args.reps,chains=args.chains,modes=args.modes,cpu=args.cpu,max_slowdown=args.max_slowdown,allow_trajectory_change=args.allow_trajectory_change,cases=args.cases)
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (args.output/'binaries.json').write_text(json.dumps({k:dict(path=str(v),sha256=sha(v)) for k,v in binaries.items()},indent=2)+'\n')
    records = []
    comparisons = []
    def save_summary():
        (args.output/'summary.json').write_text(json.dumps(summarize(records,plan,comparisons),indent=2)+'\n')
    save_summary()
    with (args.output/'records.jsonl').open('x') as output:
        for mode in args.modes:
            for case in args.cases:
                model = CASES[case]
                if mode=='evaluation' and case in ['linear_large','logistic_large','funnel']:
                    continue
                for rep in range(args.reps):
                    pair = {}
                    for engine in (['baseline','current'] if rep%2==0 else ['current','baseline']):
                        env = dict(os.environ, ALEA_BENCH_CHAINS=str(args.chains),ALEA_BENCH_TIMED='1',
                                   OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',RAYON_NUM_THREADS='1')
                        if mode=='evaluation':
                            env['ALEA_BENCH_EVALUATIONS'] = '10000'
                        start = time.perf_counter_ns()
                        result = run_process([str(binaries[engine]),model,str(ROOT/'data'/f'{case}.txt')],
                                                env=env,text=True,capture_output=True,check=False)
                        row = dict(mode=mode,case=case,engine=engine,rep=rep,returncode=result.returncode,
                                   wall_seconds=(time.perf_counter_ns()-start)/1e9,stdout=result.stdout,stderr=result.stderr)
                        try:
                            row['metrics'] = read_metrics(result.stdout)
                        except ValueError as error:
                            row['parse_error'] = str(error)
                        row['success'] = successful(row,args.chains)
                        if not row['success']:
                            row['failure_reason'] = row.get('parse_error', 'nonzero child exit or invalid/incomplete work metrics')
                        if mode == 'sampling':
                            row['diagnostic_pass'],row['diagnostic_reason'] = statistical_qualification(row,plan)
                        records.append(row)
                        pair[engine] = row
                        output.write(json.dumps(row)+'\n')
                        output.flush()
                        save_summary()
                    if all(r['success'] for r in pair.values()):
                        a,b = pair['baseline']['metrics'],pair['current']['metrics']
                        decision = dict(mode=mode,case=case,rep=rep,compatible=False)
                        try:
                            decision['equal'] = compare_work(a,b,mode,args.allow_trajectory_change)
                            decision['compatible'] = True
                        except (AssertionError,KeyError,TypeError) as error:
                            decision['error'] = repr(error)
                        comparisons.append(decision)
                        with (args.output/'validation.jsonl').open('a') as validation:
                            validation.write(json.dumps(decision)+'\n')
                        (args.output/'work-comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
                        save_summary()
                        if not decision['compatible']:
                            raise AssertionError(decision)
                    print(mode,case,rep,{k:r['returncode'] for k,r in pair.items()},flush=True)
    summary = summarize(records,plan,comparisons)
    alerts = performance_regressions(summary,args.max_slowdown)
    (args.output/'performance-alerts.json').write_text(json.dumps(alerts,indent=2)+'\n')
    if alerts:
        raise SystemExit('calibrated performance regression alert: '+json.dumps(alerts))


if __name__=='__main__':
    main()
