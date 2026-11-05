#!/usr/bin/env python3
"""Alternating fixed-work production-kernel controls; never time failed cases."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from bench import describe, sha, REPO, read_metrics, finite_number, run_process, host_metadata, command_output

# Declared before measurements. Small dimensions test dispatch crossovers;
# NIG/standard-normal paths control unrelated codegen effects. Normal factors
# additionally measure sharing the native Normal's density/event/parameter scores.
BASE_CASES = [
    ('poisson',1000,10000),('poisson',10000,1000),
    ('norm',4,2000000),('norm',32,200000),('norm',4096,10000),
    ('velocity',4,2000000),('velocity',4096,20000),
    ('momentum',4,2000000),('momentum',4096,20000),
    ('log_det',4096,20000),('mixture',8,100000),('mixture',64,20000),
    ('normal_factors',128,10000),('standard_normal',4096,10000),
    ('nig_density',1,1000000),
]
MIXTURE_CASES = [
    ('mixture_event_score',2,100000),('mixture_event_score',8,100000),
    ('mixture_event_score',32,20000),('mixture_event_score',64,20000),
    ('mixture_parameter_score',2,100000),('mixture_parameter_score',8,100000),
    ('mixture_parameter_score',32,20000),('mixture_parameter_score',64,20000),
    ('mixture_factors',8,10000),('mixture_factors',64,10000),
]
CASES = BASE_CASES + MIXTURE_CASES
BACKEND_CASES = [('poisson',1000,10000),('poisson',10000,1000),
                 ('poisson_scalar',1000,10000),('poisson_scalar',10000,1000),
                 ('normal_factors',128,10000),('tensor',1000,10000),('tensor',10000,1000)]
SUITES = {'base':BASE_CASES,'mixture-scores':MIXTURE_CASES,'all':CASES,'backends':BACKEND_CASES}
SUITES['sufficient-statistics'] = [('poisson',1000,10000),('poisson',10000,1000),
    ('poisson_common',1000,10000),('poisson_common',10000,1000),
    ('poisson_scalar',1000,10000),('poisson_scalar',10000,1000)]
SUITES['tensor'] = [(case,size,calls) for case in ['tensor','tensor_scalar','tensor_matmul','tensor_cond']
                   for size,calls in [(1000,10000),(10000,1000)]] + [
                   ('tensor_scan',8,100000),('tensor_scan',128,10000)]
SUITES['tensor-checkpoint'] = SUITES['tensor'] + [
    ('tensor_scan',1025,1000),('tensor_scan',10000,100),
    ('tensor_scan_checkpoint',1025,1000),('tensor_scan_checkpoint',10000,100)]
SUITES['tensor-fusion'] = SUITES['tensor-checkpoint'] + [
    ('tensor_dag',1000,10000),('tensor_dag',10000,1000),('tensor_dag',100000,100)]
SUITES['tensor-storage'] = SUITES['tensor-fusion'] + [
    (case,size,calls) for case in ['tensor_vector','tensor_reuse']
    for size,calls in [(1000,1000),(10000,100),(100000,10)]]
SUITES['matrix-providers'] = [('gemm',8,10000),('gemm',64,100),('gemm',128,25)]
SUITES['normal-batching'] = [('normal_factors',size,calls) for size,calls in
    [(3,100000),(4,100000),(5,100000),(128,10000),(4096,1000)]]


def compatible(a,b,case):
    if (a['calls'],a['size'],a['case']) != (b['calls'],b['size'],b['case']):
        return False
    if not all(math.isfinite(r['checksum']) for r in [a,b]):
        return False
    return (math.isclose(a['checksum'],b['checksum'],rel_tol=1e-12,abs_tol=1e-8)
            if case.startswith('poisson') else a['checksum']==b['checksum'])


def summarize(rows, cases=CASES, decisions=(), reps=None):
    summary=[]
    for case,size,calls in cases:
        group=[r for r in rows if (r['case'],r['size'])==(case,size)]
        engines={}
        def validated(row):
            matching=[d for d in decisions if (d['case'],d['size'],d['rep'])==(case,size,row['rep'])]
            pair=[r for r in group if r['rep']==row['rep']]
            return (len(matching)==1 and matching[0].get('compatible') is True
                    and len(pair)==2 and {r['engine'] for r in pair}=={'before','after'}
                    and all(r['success'] for r in pair))
        for engine in ['before','after']:
            attempts=[r for r in group if r['engine']==engine]
            valid=[r for r in attempts if r['success'] and validated(r)]
            engines[engine]=dict(attempts=len(attempts),failed=sum(not r['success'] for r in attempts),
                unvalidated_or_incompatible=sum(r['success'] and not validated(r) for r in attempts),
                planned=reps,unstarted=reps-len(attempts) if reps is not None else None,
                seconds=describe([r['metrics']['seconds'] for r in valid]),
                construction_seconds=describe([r['metrics']['construction_seconds'] for r in valid]))
        ratios=[]
        for rep in {r['rep'] for r in group}:
            pair={r['engine']:r for r in group if r['rep']==rep}
            if len(pair)==2 and all(r['success'] and validated(r) for r in pair.values()):
                ratios.append(pair['before']['metrics']['seconds']/pair['after']['metrics']['seconds'])
        summary.append(dict(case=case,size=size,calls=calls,engines=engines,paired_gain=describe(ratios)))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reps',type=int,default=7)
    p.add_argument('--cpu',type=int,help='first allowed CPU by default')
    p.add_argument('--threads',type=int,default=1,help='equal CPU/library-thread budget for both providers')
    p.add_argument('--before-env',action='append',default=[],metavar='NAME=VALUE')
    p.add_argument('--after-env',action='append',default=[],metavar='NAME=VALUE')
    p.add_argument('--build-metadata',type=Path,help='variants.json and adjacent completed qualification receipt for saved binaries')
    p.add_argument('--suite',choices=list(SUITES),default='base',
                   help='base preserves the original 15-case protocol; mixture-scores adds derivative/native controls')
    args=p.parse_args()
    def overrides(values):
        result={}
        for item in values:
            name,separator,value=item.partition('=')
            if not separator or not name or name in result: p.error('unique NAME=VALUE overrides required')
            result[name]=value
        return result
    environments=dict(before=overrides(args.before_env),after=overrides(args.after_env))
    cases=SUITES[args.suite]
    allowed=sorted(os.sched_getaffinity(0))
    cpu=allowed[0] if args.cpu is None else args.cpu
    if args.reps<1 or not 1 <= args.threads <= len(allowed) or cpu not in allowed:
        p.error('positive repetitions and thread budget within the CPU allocation required')
    cpus=[cpu]+[c for c in allowed if c!=cpu][:args.threads-1]
    for overrides in environments.values():
        for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS']:
            if key in overrides and not 1 <= int(overrides[key]) <= args.threads:
                p.error('library thread override exceeds the declared CPU budget')
    args.output.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,set(cpus))
    binaries=dict(before=args.before.resolve(),after=args.after.resolve())
    binary_builds=None
    if args.build_metadata:
        metadata=json.loads(args.build_metadata.read_text())
        receipt=read_metrics((args.build_metadata.parent/'qualification.json').read_text())
        if receipt.get('status')!='complete' or sha(args.build_metadata)!=receipt['variants_sha256']:
            raise ValueError('incomplete or changed kernel build metadata')
        binary_builds={}
        for engine,path in binaries.items():
            label=path.name.removesuffix('-kernels')
            matching=[r for r in metadata if r['name']==label]
            if len(matching)!=1 or sha(path)!=receipt['files'].get(path.name):
                raise ValueError('kernel binary does not match the completed build receipt')
            binary_builds[engine]=matching[0]
    files=[REPO/'crates/alea-math/src/numeric.rs',REPO/'crates/alea-math/src/metric.rs',
           REPO/'crates/alea-core/src/model.rs',
           REPO/'crates/alea-core/src/model/assembly.rs',
           REPO/'crates/alea-core/src/model/assembly/reverse.rs',
           REPO/'crates/alea-core/src/model/tensor.rs',
           REPO/'crates/alea-core/src/model/tensor/execute.rs',
           REPO/'crates/alea-core/src/model/tensor/kernels.rs',
           REPO/'crates/alea-core/src/model/tensor/source.rs',
           REPO/'crates/alea-core/src/model/tensor/fusion.rs',
           REPO/'crates/alea-core/src/model/tensor/pullback.rs',
           REPO/'crates/alea-math/src/gemm.rs',
           REPO/'crates/alea-distributions/src/mixture.rs',
           REPO/'crates/alea-distributions/src/scalar.rs',REPO/'crates/alea-distributions/src/modeling.rs',
           REPO/'crates/alea-distributions/src/modeling/normal.rs',
           REPO/'crates/alea-distributions/src/modeling/normal/batch.rs',
           REPO/'crates/alea-distributions/src/modeling/regression.rs',
           REPO/'crates/alea-distributions/src/modeling/regression/links.rs',
           REPO/'crates/alea-distributions/src/modeling/regression/poisson_statistics.rs',
           REPO/'crates/alea-distributions/src/modeling/regression/kernels.rs',
           Path(__file__).resolve().parents[1]/'src/bin/kernel_audit.rs',Path(__file__).with_name('bench.py'),Path(__file__).resolve()]
    manifest=dict(schema_version=2,cases=cases,suite=args.suite,reps=args.reps,cpu=cpu,threads=args.threads,machine=platform.machine(),host=host_metadata(),
        alea_commit=command_output(['git','-C',str(REPO),'rev-parse','HEAD']),
        flake_lock_sha256=sha(REPO/'flake.lock'),
        cargo_lock_sha256=sha(Path(__file__).resolve().parents[1]/'Cargo.lock'),
        package_sha256=sha(Path(__file__).resolve().parents[1]/'Cargo.toml'),
        cpuinfo=Path('/proc/cpuinfo').read_text(),binaries={k:dict(path=str(v),sha256=sha(v)) for k,v in binaries.items()},
        source_sha256={str(f.relative_to(REPO)):sha(f) for f in files},
        rustc=command_output(['rustc','-vV']),binary_builds=binary_builds,
        flags={k:os.environ.get(k) for k in ['RUSTFLAGS','CXXFLAGS','CARGO_ENCODED_RUSTFLAGS']},
        environments=environments,
        protocol='fixed production calls, construction separate; exact checksum except Poisson relative1e-12/absolute1e-8; no compile/profile overlap')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    rows=[]
    decisions=[]
    def save_summary():
        (args.output/'summary.json').write_text(json.dumps(summarize(rows,cases,decisions,args.reps),indent=2)+'\n')
    save_summary()
    with (args.output/'records.jsonl').open('x') as stream:
        for case,size,calls in cases:
            for rep in range(args.reps):
                pair={}
                for engine in (['before','after'] if rep%2==0 else ['after','before']):
                    result=run_process([str(binaries[engine]),case,str(size),str(calls)],text=True,capture_output=True,check=False,
                        env={**os.environ,**{k:str(args.threads) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS']},**environments[engine]})
                    row=dict(case=case,size=size,calls=calls,engine=engine,rep=rep,returncode=result.returncode,
                             success=False,stdout=result.stdout,stderr=result.stderr)
                    try:
                        row['metrics']=read_metrics(result.stdout)
                        m=row['metrics']
                        row['success']=(result.returncode==0 and finite_number(m['seconds']) and m['seconds']>0
                            and finite_number(m['construction_seconds']) and m['construction_seconds']>=0
                            and finite_number(m['checksum']) and (m['case'],m['size'],m['calls'])==(case,size,calls))
                    except (ValueError,KeyError,TypeError) as error:
                        row['parse_error']=repr(error)
                    if not row['success']:
                        row['failure_reason']=row.get('parse_error','nonzero child exit or invalid/incomplete work metrics')
                    pair[engine]=row
                    rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                    save_summary()
                if all(r['success'] for r in pair.values()):
                    valid=compatible(pair['before']['metrics'],pair['after']['metrics'],case)
                    decision=dict(case=case,size=size,rep=rep,compatible=valid)
                    decisions.append(decision)
                    with (args.output/'validation.jsonl').open('a') as validation:
                        validation.write(json.dumps(decision)+'\n')
                    save_summary()
                    if not valid:
                        raise AssertionError((case,size,pair))
                print(case,size,rep,{k:r['success'] for k,r in pair.items()},flush=True)


if __name__=='__main__':main()
