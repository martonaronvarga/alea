#!/usr/bin/env python3
"""Recheck native kernels against sealed Stan Math/JAX workers.

Internal full-posterior timers only; not compilation/process or ESS comparisons.
The reference workers, data, source identities and every raw attempt are retained.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

from bench import CASES, describe, sha
from regression_compare import compatible, report as paired_report
from three_engine import THREADS, child, save, verify


def report(out):
    manifest = json.loads((out/'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if sha(out/name) != digest:
            raise ValueError('changed reference input: '+name)
    rows = []
    identities = set()
    for line in (out/'records.jsonl').read_text().splitlines():
        row = json.loads(line)
        identity = row['case'], row['rep'], row['engine']
        if identity in identities:
            raise ValueError('duplicate attempt')
        identities.add(identity)
        for name, digest in row['files'].items():
            if sha(out/name) != digest:
                raise ValueError('changed reference output: '+name)
        try:
            row['metrics'] = json.loads((out/row['folder']/'stdout.txt').read_text().splitlines()[-1])
            row['success'] = (row['returncode'] == 0 and row['metrics']['evaluations'] == manifest['evaluations']
                              and row['metrics']['evaluation_seconds'] > 0)
        except (KeyError, ValueError, IndexError):
            row['success'] = False
        rows.append(row)
    summary = []
    for case in manifest['cases']:
        qualified = []
        incompatible = []
        for rep in range(manifest['reps']):
            group = {r['engine']:r for r in rows if (r['case'],r['rep']) == (case,rep)}
            if not all(e in group and group[e]['success'] for e in manifest['engines']):
                continue
            if all(compatible(group['alea']['metrics'], group[e]['metrics'], 'numerical') for e in manifest['engines']):
                qualified.append(group)
            else:
                incompatible.append(rep)
        entry = dict(case=case, qualified=len(qualified), incompatible=incompatible)
        for engine in manifest['engines']:
            entry[engine+'_ns'] = describe([g[engine]['metrics']['evaluation_seconds']*1e9/manifest['evaluations'] for g in qualified])
            if engine != 'alea':
                entry[engine+'_over_alea'] = describe([g[engine]['metrics']['evaluation_seconds']/g['alea']['metrics']['evaluation_seconds'] for g in qualified])
        summary.append(entry)
    return dict(summary=summary, attempted=len(rows), failed=sum(not r['success'] for r in rows),
                unrecorded_or_unstarted=manifest['reps']*len(manifest['cases'])*len(manifest['engines'])-len(rows))


def run(args):
    paired = args.paired.resolve()
    paired_report(paired)
    origin = args.reference.resolve()
    baseline = json.loads((origin/'manifest.json').read_text())
    verify(origin, baseline)
    out = args.output.resolve()
    out.mkdir(exist_ok=False)
    (out/'bin').mkdir()
    (out/'data').mkdir()
    shutil.copy2(paired/'bin/after', out/'bin/alea')
    shutil.copy2(origin/'bin/stan-evaluate', out/'bin/stan-evaluate')
    shutil.copy2(origin/'scripts/jax_worker.py', out/'jax_worker.py')
    for case in args.cases:
        if sha(paired/'data'/f'{case}.txt') != sha(origin/'data'/f'{case}.txt'):
            raise ValueError('different reference data')
        for extension in ('txt', 'json'):
            shutil.copy2(origin/'data'/f'{case}.{extension}', out/'data'/f'{case}.{extension}')
    for name in ('gaussian_reference.py', 'regression_compare.py', 'bench.py', 'three_engine.py'):
        shutil.copy2(Path(__file__).with_name(name), out/name)
    for label, source in [('native', paired), ('reference', origin)]:
        shutil.copy2(source/'manifest.json', out/f'{label}-manifest.json')
    files = [p for p in out.rglob('*') if p.is_file()]
    manifest = dict(cases=args.cases,reps=args.reps,evaluations=args.evaluations,cpu=args.cpu,
                    python=baseline['python'],threads=THREADS,engines=['alea','stan_glm','jax'],
                    policy='internal complete posterior value/score timer; original runtime positions and 3e-9 admission; no inference ranking',
                    files={str(p.relative_to(out)):sha(p) for p in files})
    save(out/'manifest.json',manifest)
    os.sched_setaffinity(0,{args.cpu})
    env = {**os.environ, **THREADS, 'ALEA_BENCH_EVALUATIONS':str(args.evaluations)}
    with (out/'records.jsonl').open('x') as journal:
        for rep in range(args.reps):
            for case in args.cases:
                engines = manifest['engines']
                for engine in engines[rep%3:]+engines[:rep%3]:
                    folder = out/'runs'/f'{case}-{rep}-{engine}'
                    if engine == 'alea':
                        command = [out/'bin/alea',CASES[case],out/'data'/f'{case}.txt']
                    elif engine == 'stan_glm':
                        command = [out/'bin/stan-evaluate',CASES[case],out/'data'/f'{case}.txt',args.evaluations,'glm']
                    else:
                        command = [manifest['python'],out/'jax_worker.py',CASES[case],out/'data'/f'{case}.json',
                                   'evaluate','--evaluations',args.evaluations,'--output',folder/'jax']
                    row = child(command,folder,env)
                    row.update(case=case,rep=rep,engine=engine,folder=str(folder.relative_to(out)),
                               load_after=Path('/proc/loadavg').read_text().strip(),
                               files={str(p.relative_to(out)):sha(p) for p in folder.rglob('*') if p.is_file()})
                    journal.write(json.dumps(row,allow_nan=False)+'\n')
                    journal.flush()
                    save(out/'summary.json',report(out))
                    print(case,rep,engine,row['returncode'],flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','report'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--paired',type=Path)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--cases',nargs='+',choices=list(CASES),default=list(CASES))
    parser.add_argument('--reps',type=int,default=7)
    parser.add_argument('--evaluations',type=int,default=10000)
    parser.add_argument('--cpu',type=int,default=2)
    args = parser.parse_args()
    if args.action == 'report':
        print(json.dumps(report(args.output.resolve()),indent=2,allow_nan=False))
    else:
        if not args.paired or not args.reference or args.reps < 1 or args.evaluations < 10 or args.evaluations%10 or len(set(args.cases)) != len(args.cases):
            parser.error('requires paired/reference capsules, positive reps, unique cases and a positive multiple of ten calls')
        run(args)
