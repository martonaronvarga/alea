#!/usr/bin/env python3
"""Paired native regression optimization evidence, using saved executables only."""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import sys

from bench import CASES, REPO, command_output, describe, sha, statistical_qualification
from three_engine import THREADS, child, kernel_compatible, save, verify


def compatible(a, b, policy='exact'):
    # This optimization must not change analytic scores or their accumulation.
    bits = lambda values: [struct.pack('>d', value) for value in values]
    try:
        if policy == 'numerical':
            return kernel_compatible(a, b)
        if policy != 'exact':
            raise ValueError('unknown comparison policy')
        return (kernel_compatible(a, b) and bits(a['gradient']) == bits(b['gradient'])
                and bits([a['gradient_checksum']]) == bits([b['gradient_checksum']]))
    except (KeyError, TypeError, ValueError, struct.error):
        return False


def report(out):
    manifest = json.loads((out/'manifest.json').read_text())
    if 'scope' in manifest:
        return screen_report(out,manifest)
    for name, digest in manifest['files'].items():
        if sha(out/name) != digest: raise ValueError('changed input: '+name)
    rows = []; seen = set()
    for line in (out/'records.jsonl').read_text().splitlines():
        row = json.loads(line)
        identity = row['case'], row['rep'], row['engine']
        if identity in seen: raise ValueError('duplicate attempt')
        seen.add(identity)
        for name, digest in row['files'].items():
            if sha(out/name) != digest: raise ValueError('changed output: '+name)
        try:
            row['metrics'] = json.loads((out/row['folder']/'stdout.txt').read_text().splitlines()[-1])
            m = row['metrics']
            row['success'] = (row['returncode'] == 0 and m['evaluations'] == manifest['evaluations']
                              and m['evaluation_seconds'] > 0)
        except (ValueError, KeyError, IndexError): row['success'] = False
        rows.append(row)
    summary = []
    for case in manifest['cases']:
        pairs = []; failures = []
        for rep in range(manifest['reps']):
            both = {r['engine']:r for r in rows if (r['case'],r['rep']) == (case,rep)}
            if not all(e in both and both[e]['success'] for e in ('before','after')): continue
            if compatible(both['before']['metrics'],both['after']['metrics'],manifest.get('score_policy','exact')): pairs.append(both)
            else: failures.append(rep)
        entry = dict(case=case, planned=manifest['reps'], qualified=len(pairs), incompatible=failures)
        for engine in ('before','after'):
            entry[engine] = describe([p[engine]['metrics']['evaluation_seconds']*1e9/manifest['evaluations'] for p in pairs])
            entry[engine+'_construction_seconds'] = describe([p[engine]['metrics']['construction_seconds'] for p in pairs])
        entry['after_over_before'] = describe([p['after']['metrics']['evaluation_seconds']/p['before']['metrics']['evaluation_seconds'] for p in pairs])
        summary.append(entry)
    return dict(summary=summary, attempted=len(rows), failed=sum(not r['success'] for r in rows),
                unrecorded_or_unstarted=manifest['reps']*len(manifest['cases'])*2-len(rows))


def screen_report(out, manifest):
    for name,digest in manifest['inputs'].items():
        if sha(Path(name))!=digest: raise ValueError('changed screen input')
    for name,digest in manifest['drivers'].items():
        if sha(out/name)!=digest: raise ValueError('changed screen driver')
    rows=[];seen=set()
    for line in (out/'records.jsonl').read_text().splitlines():
        row=json.loads(line);identity=row['case'],row.get('rep',0),row['engine']
        if identity in seen: raise ValueError('duplicate screen attempt')
        seen.add(identity)
        for name,digest in row['files'].items():
            if sha(out/name)!=digest: raise ValueError('changed screen output')
        try:
            row['metrics']=json.loads((out/row['folder']/'stdout.txt').read_text().splitlines()[-1])
            row['success']=row['returncode']==0 and row['metrics']['completed']==4000 and not row['metrics']['failures']
            row['qualified'],row['quality_reason']=statistical_qualification(row,dict(chains=4))
        except (ValueError,KeyError,IndexError): row.update(success=False,qualified=False)
        rows.append(row)
    return rows


def run(args):
    out = args.output.resolve(); out.mkdir(exist_ok=False)
    (out/'bin').mkdir(); (out/'data').mkdir()
    origin = args.baseline_manifest.resolve()
    baseline = json.loads(origin.read_text())
    if 'source_archive_sha256' in baseline:
        verify(origin.parent, baseline)
        binary = 'bin/alea'
    else:
        report(origin.parent)
        binary = 'bin/after'
    if sha(args.before) != baseline['files'][binary]: raise ValueError('wrong baseline executable')
    for engine in ('before','after'): shutil.copy2(getattr(args,engine),out/'bin'/engine)
    for case in args.cases:
        name = f'data/{case}.txt'
        source = args.data/f'{case}.txt'
        if sha(source) != baseline['files'][name]: raise ValueError('changed original data')
        shutil.copy2(source,out/name)
    sys.path.insert(0,str(REPO/'tools/release'))
    import source
    source.archive(REPO,out/'sources-after.tar.gz')
    shutil.copy2(__file__,out/'driver.py')
    for name in ('bench.py','three_engine.py'):
        shutil.copy2(Path(__file__).with_name(name),out/name)
    save(out/'baseline-manifest.json',baseline)
    files = [p for p in out.rglob('*') if p.is_file()]
    save(out/'manifest.json',dict(reps=args.reps,evaluations=args.evaluations,cases=args.cases,
        cpu=args.cpu,threads=THREADS,baseline_manifest_sha256=sha(origin),score_policy=args.score_policy,
        cpuinfo=Path('/proc/cpuinfo').read_text(),uname=list(os.uname()),
        rustc=command_output(['rustc','-vV']),rustflags=os.environ.get('RUSTFLAGS'),
        policy='alternating saved native executables; internal full value/gradient timer; explicit score policy and tolerance-checked densities; no sampler change',
        files={str(p.relative_to(out)):sha(p) for p in files}))
    os.sched_setaffinity(0,{args.cpu}); env={**os.environ,**THREADS,'ALEA_BENCH_EVALUATIONS':str(args.evaluations)}
    with (out/'records.jsonl').open('x') as journal:
        for rep in range(args.reps):
            for case in args.cases:
                for engine in (('before','after') if rep%2 == 0 else ('after','before')):
                    folder=out/'runs'/f'{case}-{rep}-{engine}'
                    load_before=Path('/proc/loadavg').read_text().strip()
                    row=child([out/'bin'/engine,CASES[case],out/'data'/f'{case}.txt'],folder,env)
                    row.update(case=case,rep=rep,engine=engine,folder=str(folder.relative_to(out)),
                        load_before=load_before,load_after=Path('/proc/loadavg').read_text().strip(),
                        files={str(p.relative_to(out)):sha(p) for p in folder.rglob('*') if p.is_file()})
                    journal.write(json.dumps(row,allow_nan=False)+'\n');journal.flush()
                    save(out/'summary.json',report(out))
                    print(case,rep,engine,row['returncode'],flush=True)


def sample(args):
    """Repeated four-chain timing screens; repetitions reuse the original seeds."""
    parent=args.input.resolve(); report(parent)  # Verify the paired capsule first.
    out=args.output.resolve();out.mkdir(exist_ok=False)
    for name, origin in [('driver.py',Path(__file__)),('bench.py',Path(__file__).with_name('bench.py')),
                         ('three_engine.py',Path(__file__).with_name('three_engine.py'))]:
        shutil.copy2(origin,out/name)
    inputs=[parent/'manifest.json',parent/'sources-after.tar.gz']
    inputs += [parent/'bin'/e for e in ('before','after')]
    inputs += [parent/'data'/f'{case}.txt' for case in args.cases]
    save(out/'manifest.json',dict(scope='repeated four-chain native timing screens; identical seeds, not extra independent chains',reps=args.reps,
        inputs={str(p):sha(p) for p in inputs},cases=args.cases,cpu=args.cpu,
        drivers={p.name:sha(p) for p in out.glob('*.py')}))
    os.sched_setaffinity(0,{args.cpu});rows=[]
    with (out/'records.jsonl').open('x') as journal:
        for rep, case in ((r,c) for r in range(args.reps) for c in args.cases):
            for engine in (('before','after') if rep%2 == 0 else ('after','before')):
                folder=out/f'{case}-{rep}-{engine}'
                env={**os.environ,**THREADS,'ALEA_BENCH_CHAINS':'4','ALEA_BENCH_CSV':str(folder/'draws.csv')}
                row=child([parent/'bin'/engine,CASES[case],parent/'data'/f'{case}.txt'],folder,env)
                row.update(case=case,rep=rep,engine=engine,phase='sampling',folder=str(folder.relative_to(out)))
                try:
                    row['metrics']=json.loads((folder/'stdout.txt').read_text().splitlines()[-1])
                    row['success']=row['returncode']==0 and row['metrics']['completed']==4000 and not row['metrics']['failures']
                    row['qualified'],row['quality_reason']=statistical_qualification(row,dict(chains=4))
                except (ValueError,KeyError,IndexError): row.update(success=False,qualified=False)
                row['files']={str(p.relative_to(out)):sha(p) for p in folder.rglob('*') if p.is_file()}
                journal.write(json.dumps(row,allow_nan=False)+'\n');journal.flush();rows.append(row)
                save(out/'summary.json',screen_report(out,json.loads((out/'manifest.json').read_text())))
                print(case,rep,engine,row['returncode'],row['qualified'],row['wall_seconds'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','report','sample'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--before',type=Path)
    parser.add_argument('--after',type=Path)
    parser.add_argument('--baseline-manifest',type=Path)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--input',type=Path,help='completed paired capsule for four-chain timing screens')
    parser.add_argument('--cases',nargs='+',choices=list(CASES),default=list(CASES))
    parser.add_argument('--reps',type=int,default=7)
    parser.add_argument('--evaluations',type=int,default=10000)
    parser.add_argument('--cpu',type=int,default=2)
    parser.add_argument('--score-policy',choices=['exact','numerical'],default='exact',
                        help='numerical uses the existing three-engine 3e-9 density/score tolerances; exact remains the default')
    args=parser.parse_args()
    if args.action=='run':
        if not all((args.before,args.after,args.baseline_manifest,args.data)) or args.reps<1 or args.evaluations<10 or args.evaluations%10 or len(set(args.cases))!=len(args.cases):
            parser.error('run requires saved binaries, baseline manifest, original data, positive repetitions and a multiple-of-ten evaluation count')
        run(args)
    elif args.action=='sample':
        if not args.input: parser.error('sample requires --input')
        sample(args)
    else: print(json.dumps(report(args.output.resolve()),indent=2,allow_nan=False))


if __name__ == '__main__': main()
