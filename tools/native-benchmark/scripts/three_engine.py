#!/usr/bin/env python3
"""Source-sealed native Alea / CmdStan / JAX CPU comparison; retain every outcome.

Prepare/build/test first, then measure serially on one CPU. No production engine
or statistical policy is changed. Report can be reconstructed from raw journals.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
import time

from bench import (ROOT, REPO, CASES, DIMENSIONS, command_output, describe,
                   read_metrics, sha, statistical_qualification, parse_stan)

THREADS = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               RAYON_NUM_THREADS='1', STAN_NUM_THREADS='1', JAX_PLATFORMS='cpu', JAX_ENABLE_X64='true',
               XLA_FLAGS='--xla_cpu_multi_thread_eigen=false --xla_cpu_enable_fast_math=false --xla_force_host_platform_device_count=1')
KERNELS = ['alea', 'stan_vectorized', 'stan_glm', 'jax']


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def close(a, b, tolerance=3e-9):
    return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tolerance*(1+abs(b))


def kernel_compatible(a, b):
    return (a['evaluations'] == b['evaluations'] and len(a['gradient']) == len(b['gradient'])
            and all(close(a[k], b[k]) for k in ('logp', 'checksum', 'gradient_checksum'))
            and all(close(x, y) for x, y in zip(a['gradient'], b['gradient'], strict=True)))


def rederive(out, row):
    """Use raw child outputs, not cached success/qualification/metric fields."""
    row=dict(row); folder=out/row['folder']
    try:
        if row['engine'] in ('alea','jax') or row['phase']=='evaluate':
            metrics=read_metrics((folder/'stdout.txt').read_text().splitlines()[-1])
        else: metrics=parse_stan([Path(p) for p in row['stan_paths']])
        if row['phase']=='sampling' and row['engine']!='alea':
            metrics.update(read_metrics((folder/'diagnostics/stdout.txt').read_text()))
        row['metrics']=metrics
        row['success']=(row['returncode']==0 and (row['phase']=='evaluate' or metrics.get('completed')==4000 and not metrics.get('failures')))
        if row['success'] and row['phase']=='evaluate':
            if metrics['evaluations'] <= 0 or metrics['evaluation_seconds'] <= 0:
                raise ValueError('invalid evaluation work/time')
        if row['success'] and row['phase']=='sampling':
            dimension=DIMENSIONS[row['case']]
            if row['engine']=='alea':
                with (folder/'draws.csv').open() as stream: draws=list(csv.DictReader(stream))
                if len(draws)!=4000 or {int(r['chain']) for r in draws}!=set(range(4)):
                    raise ValueError('incomplete retained Alea CSV')
                for chain in range(4):
                    indices=[int(r['draw']) for r in draws if int(r['chain'])==chain]
                    if len(indices)!=1000 or indices!=list(range(indices[0],indices[0]+1000)):
                        raise ValueError('nonconsecutive retained Alea draws')
                if any(not math.isfinite(float(r[f'q{j}'])) for r in draws for j in range(dimension)):
                    raise ValueError('nonfinite retained Alea coordinate')
            elif row['engine']=='jax':
                if len(metrics['csv_paths'])!=4: raise ValueError('incomplete JAX chain plan')
                for path in metrics['csv_paths']:
                    with Path(path).open() as stream:
                        reader=csv.reader(stream); header=next(reader); draws=list(reader)
                    if header!=[f'q{j}' for j in range(dimension)] or len(draws)!=1000 or any(len(r)!=dimension or any(not math.isfinite(float(v)) for v in r) for r in draws):
                        raise ValueError('incomplete or invalid retained JAX CSV')
    except (OSError,ValueError,KeyError,IndexError,TypeError,StopIteration) as error:
        row.update(success=False,parse_error=repr(error))
    return row


def engines(case):
    return ['alea', 'stan', 'stan_glm', 'jax'] if CASES[case] in ('linear', 'logistic') else ['alea', 'stan', 'jax']


def compiled_inputs(files):
    return {name:digest for name,digest in files.items()
            if Path(name).suffix in ('.rs','.toml','.cpp','.c','.h','.stan') or Path(name).name=='Cargo.lock'}


def child(command, folder, env, timeout=300):
    folder.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    with (folder/'stdout.txt').open('x') as out, (folder/'stderr.txt').open('x') as err:
        try:
            process = subprocess.Popen(list(map(str, command)), env=env, stdout=out, stderr=err)
            expired = threading.Event()
            def expire():
                if process.poll() is None:
                    expired.set()
                    try: process.kill()
                    except ProcessLookupError: pass
            watchdog = threading.Timer(timeout, expire)
            watchdog.daemon = True
            watchdog.start()
            try:
                # wait(timeout=...) uses polling sleeps on POSIX, up to 50 ms.
                # That quantizes short jobs and charges four waits to CmdStan.
                code = process.wait()
            except BaseException:
                if process.poll() is None: process.kill()
                process.wait()
                raise
            finally:
                watchdog.cancel()
                watchdog.join()
            error = f'worker exceeded {timeout} seconds' if expired.is_set() else None
            if expired.is_set(): code = 124
        except OSError as cause:
            code, error = 124, repr(cause)
    return dict(command=list(map(str, command)), returncode=code, error=error,
                wall_seconds=time.perf_counter()-start)


def prepare(args):
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    (out/'bin').mkdir(); (out/'data').mkdir(); (out/'scripts').mkdir()
    commands = []
    def build(command, label):
        result = child(command, out/label, os.environ)
        commands.append(result); save(out/'builds.json', commands)
        if result['returncode']: raise ValueError(f'{label} failed; see retained build log')
    source_paths = [p for directory in (REPO/'crates', ROOT) for p in directory.rglob('*')
                    if p.is_file() and 'target' not in p.parts and '__pycache__' not in p.parts
                    and (p.suffix in ('.rs','.toml','.py','.stan','.cpp','.c','.h') or p.name=='Cargo.lock')]
    before = {str(p.relative_to(REPO)):sha(p) for p in sorted(source_paths)}
    reuse = None
    if args.reuse_build:
        reuse=args.reuse_build.resolve()
        origin=read_metrics((reuse/'manifest.json').read_text());verify(reuse,origin)
        if (compiled_inputs(before)!=compiled_inputs(origin['source_sha256'])
                or origin['cmdstan']!=str(args.cmdstan.resolve())
                or origin['cxx']!=command_output([args.cxx,'--version'])
                or origin['rustc']!=command_output(['rustc','-vV'])):
            raise ValueError('reused binaries do not match current compiled inputs/toolchains')
        shutil.copytree(reuse/'bin',out/'bin',dirs_exist_ok=True)
        shutil.copy2(reuse/'stan-evaluate.d',out/'stan-evaluate.d')
        shutil.copy2(reuse/'builds.json',out/'builds.json')
        save(out/'build-origin.json',dict(manifest=origin,manifest_sha256=sha(reuse/'manifest.json'),
            source=str(reuse),reason='only benchmark Python driver changed; compiled inputs and binary hashes identical'))
    else:
        build(['cargo','build','--release','--locked','--manifest-path',ROOT/'Cargo.toml','--bins'], 'rust-build')
        shutil.copy2(ROOT/'target/release/alea-stan-bench', out/'bin/alea')
    math_root = args.cmdstan/'stan/lib/stan_math'; lib = math_root/'lib'
    flags = ['-O3','-std=c++17','-pthread','-DBOOST_DISABLE_ASSERTS','-DEIGEN_NO_DEBUG',
             '-I'+str(math_root), *['-I'+str(lib/p) for p in ('eigen_5.0.1','boost_1.87.0','tbb_2020.3/include','sundials_6.1.1/include')]]
    if reuse is None:
        build([args.cxx,*flags,'-MMD','-MF',out/'stan-evaluate.d',ROOT/'stan/evaluate.cpp',
               '-L'+str(lib/'tbb'),'-Wl,-rpath,'+str(lib/'tbb'),'-ltbb','-o',out/'bin/stan-evaluate'], 'stan-math-build')
        for model in ('mean','linear','logistic','funnel'):
            shutil.copy2(ROOT/'stan'/model, out/'bin'/model)
        for model in ('linear_glm','logistic_glm'):
            shutil.copy2(ROOT/'stan'/f'{model}.stan', out/'bin'/f'{model}.stan')
            build(['make','-C',args.cmdstan,f'CXX={args.cxx}','PRECOMPILED_HEADERS=false',out/'bin'/model],f'{model}-build')
    for case in args.cases:
        for extension in ('txt','json'):
            shutil.copy2(ROOT/'data'/f'{case}.{extension}',out/'data'/f'{case}.{extension}')
    for name in ('bench.py','check_stan.py','jax_worker.py','three_engine.py'):
        shutil.copy2(ROOT/'scripts'/name,out/'scripts'/name)
    (out/'stan').symlink_to('bin',target_is_directory=True)  # Original correctness helper layout.
    if before != {str(p.relative_to(REPO)):sha(p) for p in sorted(source_paths)}:
        raise ValueError('source changed during build; do not admit these binaries')
    sys.path.insert(0,str(REPO/'tools/release'))
    import source
    source.archive(REPO,out/'sources.tar.gz')
    dependencies = shlex.split((out/'stan-evaluate.d').read_text().replace('\\\n',' ').split(':',1)[1])
    save(out/'stan-dependencies.json',{str(Path(p).resolve()):sha(p) for p in dependencies})
    python_versions = command_output([str(args.python),'-c',"import json,importlib.metadata as m;print(json.dumps({n:m.version(n) for n in ['jax','jaxlib','blackjax','numpy','scipy']}))"])
    files = [p for directory in ('bin','scripts','data') for p in (out/directory).rglob('*') if p.is_file()]
    manifest = dict(schema_version=1,cases=args.cases,phases=args.phases,kernel_reps=args.kernel_reps,reps=args.reps,
                    cpu=args.cpu,chains=4,warmup=1000,draws=1000,depth=10,step_size=.2,target=.8,
                    evaluations=args.evaluations,threads=THREADS,python=str(args.python.resolve()),
                    python_versions=read_metrics(python_versions),
                    cmdstan=str(args.cmdstan.resolve()),cmdstan_version=command_output([args.cmdstan/'bin/stanc','--version']),
                    rustc=command_output(['rustc','-vV']),cxx=command_output([args.cxx,'--version']),
                    pins=read_metrics((ROOT/'PINNED.json').read_text()),
                    cpuinfo=Path('/proc/cpuinfo').read_text(),uname=command_output(['uname','-a']),
                    git_commit=command_output(['git','-C',REPO,'rev-parse','HEAD']),
                    source_archive_sha256=sha(out/'sources.tar.gz'),source_sha256=before,
                    reused_build_manifest_sha256=sha(reuse/'manifest.json') if reuse else None,
                    files={str(p.relative_to(out)):sha(p) for p in files},
                    flake_lock_sha256=sha(REPO/'flake.lock'),
                    policy='10000 changing-state full-posterior calls, all scores consumed; four-chain native NUTS; no matched-adaptation claim; JAX compile/import costs separate; fresh process includes output asymmetry',
                    seeds='data20261029; Alea[17;32]/streams0..3; Stan seed17/ids1..4; BlackJAX keys17..20',
                    quality='finite all-coordinate Rhat<=1.01, bulk/tailESS>=100, zero retained divergences; pair mean agreement<=6 combined MCSE; repeats reuse seeds and are timing repeats only')
    save(out/'manifest.json',manifest)
    print(out,flush=True)


def verify(out, manifest):
    if sha(out/'sources.tar.gz') != manifest['source_archive_sha256']:
        raise ValueError('changed source capsule')
    for name,digest in manifest['files'].items():
        if sha(out/name) != digest: raise ValueError(f'changed sealed input: {name}')


def execute(out, manifest, phase, case, engine, rep, parent, env, count=None):
    model=CASES[case]; folder=parent/f'{phase}-{case}-{rep}-{engine}'
    local=dict(env); command=[]
    if phase=='evaluate':
        count=count or manifest['evaluations']
        if engine=='alea':
            local['ALEA_BENCH_EVALUATIONS']=str(count)
            command=[out/'bin/alea',model,out/'data'/f'{case}.txt']
        elif engine.startswith('stan_'):
            command=[out/'bin/stan-evaluate',model,out/'data'/f'{case}.txt',count,engine.split('_')[1]]
        else:
            command=[manifest['python'],out/'scripts/jax_worker.py',model,out/'data'/f'{case}.json','evaluate',
                     '--evaluations',count,'--output',folder/'jax']
    elif engine=='alea':
        local.update(ALEA_BENCH_CHAINS='4',ALEA_BENCH_CSV=str(folder/'draws.csv'))
        command=[out/'bin/alea',model,out/'data'/f'{case}.txt']
    elif engine=='jax':
        command=[manifest['python'],out/'scripts/jax_worker.py',model,out/'data'/f'{case}.json','sampling','--output',folder/'jax']
    if command:
        record=child(command,folder,local)
        record['commands']=[record.pop('command')]
    else:
        folder.mkdir(parents=True,exist_ok=False); record=dict(commands=[],wall_seconds=0.,returncode=0,error=None)
        paths=[]; native=model+'_glm' if engine=='stan_glm' else model
        for chain in range(4):
            path=folder/f'chain-{chain}.csv';paths.append(path)
            command=[out/'bin'/native,'sample','num_warmup=1000','num_samples=1000','adapt','delta=0.8',
                     'algorithm=hmc','engine=nuts','max_depth=10','metric=diag_e','stepsize=0.2',
                     'data',f'file={out/"data"/f"{case}.json"}','random','seed=17',f'id={chain+1}',
                     'init=0','output',f'file={path}',f'profile_file={path.with_suffix(".profile.csv")}','sig_figs=17','refresh=0']
            run=child(command,folder/f'process-{chain}',local)
            record['commands'].append(run['command']);record['wall_seconds']+=run['wall_seconds']
            if run['returncode']: record['returncode']=run['returncode']
        record['stan_paths']=list(map(str,paths))
    record.update(phase=phase,case=case,engine=engine,rep=rep,folder=str(folder.relative_to(out)))
    try:
        if engine in ('alea','jax') or phase=='evaluate':
            metrics=read_metrics((folder/'stdout.txt').read_text().splitlines()[-1])
        else: metrics=parse_stan([Path(p) for p in record['stan_paths']])
        if phase=='sampling' and engine!='alea' and record['returncode']==0:
            paths=metrics['csv_paths'] if engine=='jax' else record['stan_paths']
            diagnostic=child([out/'bin/alea',model,out/'data'/f'{case}.txt'],folder/'diagnostics',
                             {**local,'ALEA_BENCH_STAN_CSVS':':'.join(paths)})
            if diagnostic['returncode']: raise ValueError('diagnostic reader failed')
            metrics.update(read_metrics((folder/'diagnostics/stdout.txt').read_text()))
        record['metrics']=metrics
        record['success']=(record['returncode']==0 and (phase=='evaluate' or metrics.get('completed')==4000 and not metrics.get('failures')))
    except (OSError,ValueError,KeyError,IndexError,TypeError) as error:
        record.update(success=False,parse_error=repr(error))
    record['files']={str(p.relative_to(out)):sha(p) for p in folder.rglob('*') if p.is_file()}
    return record


def preflight(out,manifest,env):
    parent=out/'preflight'; parent.mkdir(exist_ok=False)
    checks=[]
    selected=[case for case in manifest['cases'] if case!='funnel']
    if selected:
        checked=child([sys.executable,out/'scripts/check_stan.py','--alea',out/'bin/alea',
                       '--output',out/'cmdstan-check','--cases',*selected],parent/'cmdstan-probes',env)
        if checked['returncode']: raise ValueError('CmdStan shared-coordinate check failed')
        for row in json.loads((out/'cmdstan-check/checks.json').read_text()):
            checks.append(dict(case=row['case'],point=row['point'],engine='cmdstan-probes',passed=row['density_ok'] and row['gradient_ok']))
    for case in manifest['cases']:
        values={}
        for engine in KERNELS:
            row=execute(out,manifest,'evaluate',case,engine,0,parent,env,count=100)
            save(parent/f'{case}-{engine}.json',row)
            if not row['success']: raise ValueError(f'preflight worker failed: {case}/{engine}')
            values[engine]=row['metrics']
        for engine in KERNELS[1:]:
            ok=kernel_compatible(values['alea'],values[engine]);checks.append(dict(case=case,engine=engine,passed=ok))
    # Additional off-loop/free-coordinate/extreme probes. CmdStan's full target
    # is checked by check_stan.py separately, including positive chart Jacobians.
    for case in manifest['cases']:
        if case=='funnel': continue
        dimension=DIMENSIONS[case];model=CASES[case]
        points=[[0.]*dimension,[.3]*dimension,[-1.5]*dimension]
        points+= [[1000.]*dimension,[-1000.]*dimension] if model=='logistic' else [[.2]*(dimension-1)+[-20.]]
        path=parent/f'{case}-points.json';save(path,points)
        a=child([out/'bin/alea',model,out/'data'/f'{case}.txt'],parent/f'{case}-alea-probes',{**env,'ALEA_BENCH_PROBES':str(path)})
        j=child([manifest['python'],out/'scripts/jax_worker.py',model,out/'data'/f'{case}.json','probes','--points',path,
                 '--output',parent/f'{case}-jax-probes'/'jax'],parent/f'{case}-jax-probes',env)
        if a['returncode'] or j['returncode']: raise ValueError('probe process failed')
        actual=json.loads((parent/f'{case}-alea-probes/stdout.txt').read_text())
        reference=read_metrics((parent/f'{case}-jax-probes/stdout.txt').read_text())['probes']
        for i,(a,b) in enumerate(zip(actual,reference,strict=True)):
            ok=(close(a['logp']-actual[0]['logp'],b['logp']-reference[0]['logp'],2e-10)
                and len(a['gradient'])==len(b['gradient']) and all(close(x,y) for x,y in zip(a['gradient'],b['gradient'],strict=True)))
            checks.append(dict(case=case,point=i,engine='jax-probes',passed=ok))
            cpp=child([out/'bin/stan-evaluate',model,out/'data'/f'{case}.txt',1,'glm',*a['q']],
                      parent/f'{case}-stan-glm-probe-{i}',env)
            if cpp['returncode']: raise ValueError('Stan GLM probe process failed')
            c=read_metrics((parent/f'{case}-stan-glm-probe-{i}/stdout.txt').read_text())
            checks.append(dict(case=case,point=i,engine='stan-glm-probes',passed=close(a['logp'],c['logp'],2e-10)
                and len(a['gradient'])==len(c['gradient']) and all(close(x,y) for x,y in zip(a['gradient'],c['gradient'],strict=True))))
    save(out/'checks.json',checks)
    if not all(c['passed'] for c in checks): raise ValueError('cross-engine correctness failed; no timing admission')


def report(out):
    manifest=read_metrics((out/'manifest.json').read_text());verify(out,manifest)
    checks=json.loads((out/'checks.json').read_text())
    if not checks or not all(c.get('passed') is True for c in checks): raise ValueError('failed preflight')
    rows=[]; seen=set()
    for line in (out/'records.jsonl').read_text().splitlines():
        row=read_metrics(line); key=tuple(row[k] for k in ('phase','case','engine','rep'))
        if key in seen: raise ValueError('duplicate journal identity')
        seen.add(key)
        for name,digest in row['files'].items():
            if sha(out/name)!=digest: raise ValueError('changed raw output')
        rows.append(rederive(out,row))
    summary=[]
    for phase in manifest.get('phases',('evaluate','sampling')):
        for case in manifest['cases']:
            for engine in KERNELS if phase=='evaluate' else engines(case):
                group=[r for r in rows if (r['phase'],r['case'],r['engine'])==(phase,case,engine)]
                valid=[r for r in group if r['success']]
                planned=manifest['kernel_reps'] if phase=='evaluate' else manifest['reps']
                item=dict(phase=phase,case=case,engine=engine,planned=planned,attempted=len(group),
                          failed=len(group)-len(valid),unstarted=planned-len(group),wall_seconds=describe([r['wall_seconds'] for r in valid]))
                if phase=='evaluate':
                    qualified=[]
                    for r in valid:
                        reference=next((a for a in rows if (a['phase'],a['case'],a['engine'],a['rep'])==('evaluate',case,'alea',r['rep']) and a['success']),None)
                        if reference and kernel_compatible(reference['metrics'],r['metrics']): qualified.append(r)
                    item.update(qualified=len(qualified),ns_per_evaluation=describe([r['metrics']['evaluation_seconds']*1e9/r['metrics']['evaluations'] for r in qualified]))
                else:
                    qualified=[r for r in valid if statistical_qualification(r,dict(chains=4))[0]]
                    item.update(diagnostic_passes=len(qualified),diagnostic_failures=[statistical_qualification(r,dict(chains=4))[1] for r in group if not statistical_qualification(r,dict(chains=4))[0]],
                                min_bulk_ess_per_process_second=describe([min(d['bulk_ess'] for d in r['metrics']['diagnostics'])/r['wall_seconds'] for r in qualified]),
                                retained_leapfrog_steps=describe([r['metrics']['retained_leapfrog_steps'] for r in valid]),
                                retained_divergences=[r['metrics']['retained_divergences'] for r in valid])
                summary.append(item)
    pairs=[]
    for case in manifest['cases']:
        for engine in [e for e in engines(case) if e!='alea']:
            eligible=[]; reasons=[]
            for rep in range(manifest['reps']):
                both={r['engine']:r for r in rows if (r['phase'],r['case'],r['rep'])==('sampling',case,rep)}
                if not all(e in both and statistical_qualification(both[e],dict(chains=4))[0] for e in ('alea',engine)):
                    reasons.append('incomplete or diagnostic-ineligible pair');continue
                a,b=both['alea'],both[engine]
                good=all(abs(x['mean']-y['mean']) <= 6*math.hypot(x['mcse_mean'],y['mcse_mean'])+1e-10*(1+abs(y['mean']))
                         for x,y in zip(a['metrics']['diagnostics'],b['metrics']['diagnostics'],strict=True))
                if not good: reasons.append('mean agreement exceeds six combined MCSE');continue
                ess=lambda r:min(d['bulk_ess'] for d in r['metrics']['diagnostics'])/r['wall_seconds']
                eligible.append((a['wall_seconds']/b['wall_seconds'],ess(a)/ess(b)))
            pairs.append(dict(case=case,reference=engine,qualified=len(eligible),withheld_reasons=reasons,
                              alea_over_reference_process=describe([x for x,y in eligible]),alea_over_reference_ess_per_process_second=describe([y for x,y in eligible])))
    return dict(schema_version=1,scope='one-host CPU/f64; full posterior kernel versus full fresh-process workflows; no equivalent adaptation claim',summary=summary,pairs=pairs)


def run(args):
    out=args.output.resolve(); manifest=read_metrics((out/'manifest.json').read_text());verify(out,manifest)
    os.sched_setaffinity(0,{manifest['cpu']});env={**os.environ,**manifest['threads']}
    preflight(out,manifest,env)
    parent=out/'runs';parent.mkdir(exist_ok=False)
    with (out/'records.jsonl').open('x') as journal:
        for phase in manifest.get('phases',('evaluate','sampling')):
            reps=manifest['kernel_reps'] if phase=='evaluate' else manifest['reps']
            for rep in range(reps):
                for case in manifest['cases']:
                    candidates=KERNELS if phase=='evaluate' else engines(case)
                    order=candidates[rep%len(candidates):]+candidates[:rep%len(candidates)]
                    for engine in order:
                        verify(out,manifest)
                        row=execute(out,manifest,phase,case,engine,rep,parent,env)
                        journal.write(json.dumps(row,allow_nan=False)+'\n');journal.flush()
                        save(out/'summary.json',report(out))
                        print(phase,case,rep,engine,'complete' if row['success'] else 'FAILED',round(row['wall_seconds'],4),flush=True)
    verify(out,manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','report'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cmdstan',type=Path)
    parser.add_argument('--python',type=Path)
    parser.add_argument('--cxx',type=Path)
    parser.add_argument('--reuse-build',type=Path,help='reuse sealed executables only when every compiled source and toolchain identity matches')
    parser.add_argument('--cases',nargs='+',choices=list(CASES),default=list(CASES))
    parser.add_argument('--phases',nargs='+',choices=['evaluate','sampling'],default=['evaluate','sampling'])
    parser.add_argument('--cpu',type=int,default=2)
    parser.add_argument('--reps',type=int,default=7)
    parser.add_argument('--kernel-reps',type=int,default=7)
    parser.add_argument('--evaluations',type=int,default=10000)
    args=parser.parse_args()
    if len(args.phases)!=len(set(args.phases)): parser.error('duplicate phases')
    if args.action=='prepare':
        if not all((args.cmdstan,args.python,args.cxx)) or min(args.reps,args.kernel_reps)<1 or args.evaluations<10 or args.evaluations%10:
            parser.error('prepare requires pinned paths, positive repetitions and a positive multiple-of-ten evaluation count')
        prepare(args)
    elif args.action=='run': run(args)
    else: print(json.dumps(report(args.output.resolve()),indent=2,allow_nan=False))


if __name__=='__main__': main()
