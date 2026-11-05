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
import itertools
import re
import tomllib

from bench import (ROOT, REPO, CASES, DEFAULT_CASES, DIMENSIONS, command_output, describe,
                   read_metrics, sha, statistical_qualification, parse_stan, host_metadata, functional_agreement)

THREADS = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               RAYON_NUM_THREADS='1', STAN_NUM_THREADS='1', JAX_PLATFORMS='cpu', JAX_ENABLE_X64='true',
               XLA_FLAGS='--xla_cpu_multi_thread_eigen=false --xla_cpu_enable_fast_math=false --xla_force_host_platform_device_count=1')
KERNELS = ['alea', 'stan_vectorized', 'stan_glm', 'jax']

# Reviewed associations, NOT claims that every function in a module is timed
# independently. Unmapped modules stay explicit gaps rather than inheriting a
# package-wide "covered" label. Internal children inherit only a workflow link.
CARGO_BENCH_SCOPES = {
    'alea-math': {'storage':['buffer.rs'], 'quasi_newton':['quasi_newton.rs'],
        'primitives':['numeric.rs','random.rs','metric.rs','matrix.rs','affine.rs','exact.rs','interpolation.rs']},
    'alea-core': {'transforms':['transform'], 'tensor_tape':['model/tensor'],
        'model_pipeline':['model/assembly','target.rs','density.rs','model.rs']},
    'alea-distributions': {'scalar_scores':['scalar','multivariate','directional.rs','mixture.rs'], 'regression_execution':['modeling/regression'],
        'model_semantics':['modeling.rs'], 'wiener7':['wiener']},
    'alea-mcmc': {'hmc':['hmc','nuts','integrator.rs','hamiltonian','ghmc.rs','mams.rs','walnuts.rs','atlas_sampler.rs','atlas.rs','chees.rs','advanced.rs','canonical.rs'],
        'rwmh':['rwmh.rs'], 'warmup':['adapt']},
    'alea-runtime': {'workflow':['session','diagnostics','draws.rs','output.rs','optimize.rs','analysis']},
    'alea-autodiff': {'derivatives':['codegen.rs','gradient_check.rs']},
    'alea-ffi': {'cubature':['cubature']},
}


def cargo_bench_inventory(root):
    """Static inventory without compiling, timing, or guessing module coverage."""
    root=Path(root)
    workspace=tomllib.loads((root/'crates/Cargo.toml').read_text())['workspace']
    packages=[]
    for member in workspace['members']:
        package=root/'crates'/member
        manifest=tomllib.loads((package/'Cargo.toml').read_text())
        targets=[]
        for target in manifest.get('bench',[]):
            source=package/target.get('path',f"benches/{target['name']}.rs")
            if not source.is_file(): raise ValueError(f'missing benchmark source: {source}')
            targets.append(dict(name=target['name'],path=str(source.relative_to(root)),
                features=target.get('required-features',[]),criterion=target.get('harness') is False))
        names={t['name'] for t in targets}
        if manifest['package'].get('autobenches',True):
            automatic=list((package/'benches').glob('*.rs'))+list((package/'benches').glob('*/main.rs'))
            for source in sorted(automatic):
                name=source.stem if source.name!='main.rs' else source.parent.name
                if name not in names:
                    targets.append(dict(name=name,path=str(source.relative_to(root)),features=[],criterion=False))
                    names.add(name)
        scopes=CARGO_BENCH_SCOPES.get(member,{})
        if set(scopes)-names: raise ValueError(f'missing declared targets: {member}: {set(scopes)-names}')
        modules=[]
        for source in sorted((package/'src').rglob('*.rs')):
            name=source.relative_to(package/'src').as_posix()
            matches=[bench for bench,paths in scopes.items() if any(
                name==p or name==p+'.rs' or name.startswith(p+'/') for p in paths)]
            contracts=member in {'alea','alea-smc'} or name in {'lib.rs','prelude.rs','error.rs','state_space.rs','distribution.rs'} or name.endswith(('/tests.rs','_tests.rs'))
            modules.append(dict(path=str(source.relative_to(root)),benchmarks=matches,
                status='declaration-or-test' if contracts else ('workflow-associated' if matches else 'gap')))
        packages.append(dict(package=member,targets=targets,modules=modules))
    return dict(schema_version=1,scope='native cargo bench inventory; associations are not per-function or measured coverage',
        packages=packages,gaps=[m['path'] for p in packages for m in p['modules'] if m['status']=='gap'],
        compiled_ad_manifest='tools/enzyme-conformance/Cargo.toml')


def cargo_bench_report(root):
    """Read Criterion's existing estimates/samples; no second timing engine."""
    root=Path(root)
    cells=[];rows=[]
    def number(value,positive=False):
        if type(value) not in (float,int) or not math.isfinite(value) or (value<=0 if positive else value<0):
            raise ValueError('invalid Criterion numeric estimate/sample')
        return value
    for cell in sorted(root.glob('cell-*')):
        if not cell.is_dir(): continue
        features=(cell/'features.txt').read_text().strip()
        status='build-failed' if (cell/'BUILD_FAILED').exists() else ('run-failed' if (cell/'RUN_FAILED').exists() else ('finished' if (cell/'run.log').exists() else 'not-run'))
        cells.append(dict(cell=cell.name,features=features,status=status))
        for path in sorted((cell/'criterion').rglob('new/estimates.json')):
            location=path.parent
            estimates=json.loads(path.read_text())
            identity=json.loads((location/'benchmark.json').read_text())
            sample=json.loads((location/'sample.json').read_text())
            iterations=sample['iters'];times=sample['times']
            if not iterations or len(iterations)!=len(times): raise ValueError('incomplete Criterion samples')
            for value in iterations: number(value,True)
            for value in times: number(value,True)
            median=estimates['median'];interval=median['confidence_interval']
            point=number(median['point_estimate'],True)
            lower=number(interval['lower_bound']);upper=number(interval['upper_bound'],True)
            if lower>upper: raise ValueError('inconsistent Criterion interval')
            level=number(interval['confidence_level'],True)
            if level>=1: raise ValueError('invalid Criterion confidence level')
            parts=location.relative_to(cell/'criterion').parts
            namespace='/'.join(parts[:2]) if len(parts)>=4 and (parts[0]=='alea' or parts[0].startswith('alea-')) else ''
            case=namespace+'/'+identity['full_id'] if namespace else identity['full_id']
            rows.append(dict(cell=cell.name,features=features,case=case,namespace=namespace,benchmark_id=identity['full_id'],
                nanoseconds_per_iteration=point,median_interval=[lower,upper],confidence_level=level,
                samples=len(iterations),iterations=sum(iterations),throughput=identity.get('throughput'),
                source=str(location.relative_to(root))))
    seen=set()
    for row in rows:
        key=(row['cell'],row['case'])
        if key in seen: raise ValueError('duplicate Criterion case identity')
        seen.add(key)
    comparisons=[]
    for case in sorted({r['case'] for r in rows}):
        group=[r for r in rows if r['case']==case]
        for a,b in itertools.combinations(group,2):
            if a['throughput']!=b['throughput']: raise ValueError('changed Criterion work units')
            comparisons.append(dict(case=case,reference=a['cell'],candidate=b['cell'],
                reference_over_candidate=a['nanoseconds_per_iteration']/b['nanoseconds_per_iteration']))
    return dict(schema_version=1,scope='sequential feature-cell ratios of Criterion medians, not paired estimates, cross-source qualification or ESS/sec',
        cells=cells,measurements=rows,comparisons=comparisons)


def allocation_affinity(plan):
    """Use only scheduler-allowed CPUs; CPU numbers need not start at zero."""
    allowed = set(os.sched_getaffinity(0))
    cpus = plan.get('cpus')
    if cpus is None:
        cpu = plan.get('cpu')
        cpus = [min(allowed) if cpu is None else cpu]
    if (not cpus or any(type(c) is not int for c in cpus)
            or len(set(cpus)) != len(cpus) or not set(cpus) <= allowed):
        raise ValueError('requested CPUs must be unique members of the current allocation')
    threads = plan.get('threads', 1)
    if type(threads) is not int or not 1 <= threads <= len(cpus):
        raise ValueError('thread count must fit the requested CPU budget')
    return sorted(cpus), threads


def profile_run(args):
    """Profile saved evaluation binaries; never rank instrumented wall times."""
    if not args.local_profile and (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION')=='interactive'):
        raise ValueError('profiling requires a compute allocation or explicit --local-profile')
    plan=read_metrics(args.matrix_plan.read_text())
    names={v['name'] for v in plan['variants']}
    if not set(args.cases)<=set(plan['cases']) or (args.profile_variants and not set(args.profile_variants)<=names):
        raise ValueError('unknown profiling case or variant')
    cpus,_=allocation_affinity(dict(cpu=args.cpu,threads=1))
    perf=shlex.split(args.perf_command)
    if not perf or args.evaluations<1: raise ValueError('perf command and positive evaluation count required')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for case in args.cases:
        for variant in plan['variants']:
            if args.profile_variants and variant['name'] not in args.profile_variants: continue
            if case not in variant.get('cases',plan['cases']): continue
            label=f"{case}-{variant['name']}"
            env={**os.environ,**THREADS,**variant.get('env',{})}
            for key in ['ALEA_BENCH_PROBES','ALEA_BENCH_STATES','ALEA_BENCH_KERNEL','ALEA_BENCH_STAN_CSVS']:env.pop(key,None)
            env['ALEA_BENCH_EVALUATIONS']=str(args.evaluations)
            if bank:=plan.get('state_banks',{}).get(case):
                validate_state_bank(bank,DIMENSIONS[case]);env['ALEA_BENCH_STATES']=bank
            command=[variant['binary'],CASES[case],str(ROOT/'data'/f'{case}.txt')]
            if variant.get('kind')=='stan_math' or variant['name']=='stan':
                command += [str(args.evaluations),variant.get('implementation','vectorized')]
            # Keep controlled settings on the target side of an optional sudo
            # perf prefix; sudo otherwise drops the evaluation-mode variable.
            controlled={**THREADS,**variant.get('env',{}),
                        'ALEA_BENCH_EVALUATIONS':str(args.evaluations)}
            if 'ALEA_BENCH_STATES' in env: controlled['ALEA_BENCH_STATES']=env['ALEA_BENCH_STATES']
            command=['taskset','-c',str(cpus[0]),'env',*[f'{k}={v}' for k,v in controlled.items()],*command]
            stat=child([*perf,'stat','-x',';','-o',output/f'{label}.stat','-e','cycles:u,instructions:u,branches:u,branch-misses:u','--',*command],output/f'{label}-counters',env)
            record=child([*perf,'record','-q','-m','256','-e','cycles:u','-F','997','--call-graph','dwarf,8192','-o',output/f'{label}.data','--',*command],output/f'{label}-samples',env)
            report=None
            if record['returncode']==0:
                report=child([*perf,'report','--stdio','--no-children','-g','none','-i',output/f'{label}.data'],output/f'{label}-report',env)
            rows.append(dict(case=case,variant=variant['name'],stat=stat,record=record,report=report))
            save(output/'profile.json',dict(scope='local diagnostic; not comparative timing' if args.local_profile else 'compute-node diagnostic; not comparative timing',host=host_metadata(),plan=plan,rows=rows))
    if not rows or any(r['stat']['returncode'] or r['record']['returncode'] or not r['report'] or r['report']['returncode'] for r in rows):
        raise ValueError('profiling incomplete or failed; see retained profile.json and child stderr')


def slurm_submit(args):
    """Schedule existing commands, without introducing a second benchmark runner."""
    command = args.job_command
    if command and command[0] == '--': command = command[1:]
    if not command or args.cpus < 1 or not re.fullmatch(r'\d+:\d{2}:\d{2}', args.walltime):
        raise ValueError('a command, positive CPU budget and HH:MM:SS walltime are required')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    script = out/'job.sh'
    script.write_text('#!/bin/bash\nset -euo pipefail\n'
        ': "${SLURM_JOB_ID:?benchmark commands require a Slurm allocation}"\n'
        + 'cd ' + shlex.quote(str(ROOT)) + '\n'
        + 'exec srun --ntasks=1 --cpu-bind=cores ' + shlex.join(list(map(str, command))) + '\n')
    submit = ['sbatch', '--parsable', '--partition='+args.partition, '--nodes=1',
              '--ntasks=1', '--cpus-per-task='+str(args.cpus), '--time='+args.walltime,
              '--job-name=alea-benchmark', '--chdir='+str(ROOT),
              '--output='+str(out/'slurm-%j.log')]
    if not args.shared_node: submit.append('--exclusive')
    excluded=getattr(args,'exclude_nodes',None)
    if excluded:
        if any(not re.fullmatch(r'[A-Za-z0-9_.-]+',node) for node in excluded):
            raise ValueError('excluded nodes must be explicit Slurm node names')
        submit.append('--exclude='+','.join(excluded))
    memory=getattr(args,'memory',None)
    if memory:
        if not re.fullmatch(r'[1-9]\d*[KMGTP]?',memory): raise ValueError('invalid Slurm memory budget')
        submit.append('--mem='+memory)
    elif args.shared_node:
        raise ValueError('shared-node jobs require an explicit memory budget')
    if args.account: submit.append('--account='+args.account)
    afterok=getattr(args,'afterok',None)
    afterany=getattr(args,'afterany',None)
    if afterok and afterany: raise ValueError('choose afterok or afterany, not both')
    if afterok:
        if not afterok.isdecimal(): raise ValueError('afterok requires a numeric Slurm job ID')
        submit.append('--dependency=afterok:'+afterok)
    if afterany:
        if any(not job.isdecimal() for job in afterany): raise ValueError('afterany requires numeric Slurm job IDs')
        submit.append('--dependency=afterany:'+':'.join(afterany))
    submit.append(str(script))
    save(out/'submission.json', dict(command=command, submission=submit,
         exclusive=not args.shared_node, status='prepared'))
    if not args.dry_run:
        result = subprocess.run(submit, capture_output=True, text=True)
        save(out/'submission.json', dict(command=command, submission=submit,
             exclusive=not args.shared_node, returncode=result.returncode,
             stdout=result.stdout, stderr=result.stderr,
             status='submitted' if result.returncode == 0 else 'failed'))
        if result.returncode: raise ValueError('Slurm submission failed: '+result.stderr.strip())
        print('Submitted Slurm job '+result.stdout.strip(), flush=True)


def matrix_plan(args):
    """Expand saved builds into actual native/source and scalar/auto executions."""
    variants = []
    for builds in args.builds:
        for original in json.loads(builds.read_text()):
            prefix = builds.resolve().parent.name+'-'+original['name']
            for execution in ['auto', 'scalar']:
                variant = dict(original, name=prefix+'-native-'+execution,
                    env={'ALEA_BENCH_BACKEND':'native', 'ALEA_BENCH_REGRESSION':execution},
                    execution=dict(ad='native', regression=execution,
                        note='Cargo features describe availability, not kernel selection'))
                variants.append(variant)
            if 'enzyme' in original['features']:
                variants.append(dict(original, name=prefix+'-source', cases=['covariance'],
                    env={'ALEA_BENCH_BACKEND':'enzyme'},
                    execution=dict(ad='source-enzyme', binding='frozen covariance fixture')))
    for name, implementation in [('stan','vectorized'), ('stan_glm','glm')]:
        variants.append(dict(name=name, binary=str(args.stan_evaluate.resolve()),
            kind='stan_math', implementation=implementation,
            **({'cases':['linear','linear_large','logistic','logistic_large']} if name=='stan_glm' else {})))
    if len({v['name'] for v in variants}) != len(variants):
        raise ValueError('build directories and variant names must identify unique configurations')
    if args.output.exists(): raise ValueError('plan output already exists')
    banks={case:str((args.state_bank_dir/f'{case}.txt').resolve()) for case in args.cases} if getattr(args,'state_bank_dir',None) else {}
    for case,path in banks.items(): validate_state_bank(path,DIMENSIONS[case])
    save(args.output, dict(cases=args.cases, reps=args.reps, evaluations=args.evaluations,
                          cpu=args.cpu, variants=variants,state_banks=banks))


def validate_state_bank(path, dimension):
    values=[float(v) for v in Path(path).read_text().split()]
    # Matrix receipts replay ten states per cycle; the standalone executables
    # accept any nonempty bank. Keep this measured-work contract explicit.
    if len(values)!=10*dimension or not all(math.isfinite(v) for v in values):
        raise ValueError('matrix state bank requires ten complete finite states')


def suite_run(args):
    """One entry point, reusing builders, evaluation, inference and kernel tools."""
    from types import SimpleNamespace
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    steps=[]
    def execute(command, label):
        row=child(command,out/(label+'-logs'),os.environ,timeout=args.job_timeout)
        steps.append(dict(stage=label,**row));save(out/'progress.json',steps)
        return row['returncode']==0
    if not args.builds:
        args.builds=[]
        for index,toolchain in enumerate(args.toolchains):
            flags=os.environ.get('RUSTFLAGS','')+' -Cembed-bitcode=yes'
            if args.native_cpu: flags+=' -Ctarget-cpu=native'
            directory=out/('builds-'+str(index))
            if not execute(['env','RUSTUP_TOOLCHAIN='+toolchain,'RUSTFLAGS='+flags,
                            sys.executable,__file__,'matrix-build','--output',directory,
                            '--feature-axes',*args.feature_axes], 'builds-'+str(index)):
                raise ValueError('build stage interrupted; see progress.json and retained build outcomes')
            args.builds.append(directory/'variants.json')
    reference=args.stan_evaluate
    if reference is None:
        if not args.cmdstan or not args.cxx: raise ValueError('suite requires --cmdstan and --cxx')
        if not execute([sys.executable,__file__,'matrix-stan-build','--cmdstan',args.cmdstan,
                        '--cxx',args.cxx,'--cases',*args.cases,'--output',out/'stan',
                        *(['--native-cpu'] if args.native_cpu else [])], 'stan-build'):
            raise ValueError('Stan build failed; see retained build logs')
        reference=out/'stan/stan-evaluate'
    plan_path=out/'plan.json'
    matrix_plan(SimpleNamespace(**{**vars(args),'output':plan_path,'stan_evaluate':reference}))
    plan=read_metrics(plan_path.read_text())
    for case,path in plan.get('state_banks',{}).items():
        if case not in plan['cases']: raise ValueError('unknown state bank workload')
        validate_state_bank(path,DIMENSIONS[case])
    admission_plan=plan_path
    admission_out=out/'evaluation'
    if 'evaluate' not in args.phases:
        admission_plan=out/'preflight-plan.json';admission_out=out/'sampling-preflight'
        save(admission_plan,dict(plan,reps=1,evaluations=10))
    admitted=execute([sys.executable,__file__,'matrix','--matrix-plan',admission_plan,
                     '--output',admission_out], 'evaluation' if 'evaluate' in args.phases else 'preflight')
    checks={}
    if admitted and (admission_out/'checks.json').exists():
        checks={(row['case'],row['variant']):row['passed']
                for row in json.loads((admission_out/'checks.json').read_text())}
    cpu=allocation_affinity(plan)[0][0]
    build_checks=build_admissions(plan['variants'])
    baseline=next((v for v in plan['variants'] if v.get('execution',{}).get('ad')=='native'
                   and v['execution']['regression']=='auto' and not v.get('build_returncode',0)
                   and build_checks[v['name']] is None),None)
    # Native/source executions are distinct in the evaluation plan. Source-backed
    # inference is not implemented by this worker and must not be silently timed
    # as native inference under a source label.
    for variant in plan['variants']:
        if variant.get('kind')=='stan_math': continue
        name=variant['name']
        if variant['execution']['ad']!='native':
            if 'sampling' in args.phases:
                steps.append(dict(stage=name+'-sampling',status='unsupported',
                                  reason='this source worker exposes evaluation only'))
                save(out/'progress.json',steps)
            continue
        if variant.get('build_returncode',0) or build_checks[name]:
            steps.append(dict(stage=name,status='unavailable',reason='failed or unqualified build'))
            save(out/'progress.json',steps);continue
        if 'evaluate' in args.phases and variant['execution']['regression']=='auto':
            kernels=variant['binary']+'-kernels'
            for suite in ['all','backends','sufficient-statistics','tensor-storage']:
                execute([sys.executable,ROOT/'scripts/kernel_compare.py',
                    '--before',baseline['binary']+'-kernels','--after',kernels,
                    '--suite',suite,'--cpu',str(cpu),'--reps',str(args.reps),
                    '--output',out/(name+'-'+suite)], name+'-'+suite)
            for feature,provider in [('faer','faer'),('openblas','blas')]:
                if feature in variant['features']:
                    for threads in sorted({1,len(os.sched_getaffinity(0))}):
                        label=name+'-'+provider+f'-{threads}cpu'
                        execute([sys.executable,ROOT/'scripts/kernel_compare.py',
                            '--before',kernels,'--after',kernels,'--suite','matrix-providers',
                            '--before-env','ALEA_MATRIX_KERNEL=compensated',
                            '--after-env','ALEA_MATRIX_KERNEL='+provider,
                            '--cpu',str(cpu),'--threads',str(threads),'--reps',str(args.reps),
                            '--build-metadata',Path(variant['build_receipt']).parent/'variants.json',
                            '--output',out/label],label)
        if 'sampling' in args.phases:
            cases=[case for case in args.cases if checks.get((case,name),False)]
            excluded=[case for case in args.cases if case not in cases]
            if excluded:
                steps.append(dict(stage=name+'-sampling-preflight',status='preflight_failed',cases=excluded))
                save(out/'progress.json',steps)
            if not cases: continue
            # bench.py already alternates engines, retains failures and computes
            # qualified ESS/sec. Do not duplicate its statistical implementation.
            for workers in sorted({1,min(4,len(os.sched_getaffinity(0)))}):
                label=name+f'-{workers}cpu-sampling'
                command=['env',*[k+'='+v for k,v in variant['env'].items()],sys.executable,
                    ROOT/'scripts/bench.py','--cmdstan',args.cmdstan,'--alea',variant['binary'],
                    '--stan-directory',reference.parent,'--results',out/label,
                    '--cases',*cases,'--reps',str(args.reps),'--chains','4',
                    '--chain-workers',str(workers),*(['--cpu',str(cpu)] if workers==1 else []),
                    '--minimum-ess-per-chain','100','--both-coordinate-spaces','--alea-csv','--timed']
                execute(command,label)
    save(out/'progress.json',steps)
    suite_report(out)


def suite_report(out):
    """A compact view over the existing evaluation and inference reports."""
    from bench import summarize
    from replay import journal, rebuild
    evaluation=matrix_report(out/'evaluation') if (out/'evaluation/manifest.json').exists() else []
    sampling=[];kernels=[]
    for directory in sorted(out.glob('*-sampling')):
        if not (directory/'manifest.json').exists(): continue
        manifest=read_metrics((directory/'manifest.json').read_text())
        records=list(journal(directory/'records.jsonl')) if (directory/'records.jsonl').exists() else []
        sampling.extend(dict(row,configuration=directory.name.removesuffix('-sampling'))
                        for row in summarize(records,manifest))
    for directory in sorted(out.iterdir()):
        if not directory.is_dir() or not (directory/'manifest.json').exists(): continue
        manifest=read_metrics((directory/'manifest.json').read_text())
        if 'suite' not in manifest: continue
        kernels.extend(dict(row,configuration=directory.name,threads=manifest.get('threads',1),
                            environments=manifest.get('environments',{}))
                       for row in rebuild(directory,'kernel'))
    save(out/'summary.json',dict(evaluation=evaluation,sampling=sampling,kernels=kernels,
         policy='evaluation and qualified inference ranked separately; no universal winner; partial runs retained'))
    lines=['# Alea configuration comparison','',
           'Evaluation: full posterior value + gradient, not ESS/sec. Only complete admitted repetitions enter this ranking.','',
           '| Model | Configuration | Median µs/evaluation | Median paired Stan/config |',
           '|---|---|---:|---:|']
    for case in dict.fromkeys(r['case'] for r in evaluation):
        for row in sorted((r for r in evaluation if r['case']==case and r['admitted']==r['planned']),
                          key=lambda r:r['seconds']['median']):
            ratio=row['stan_over_variant']
            lines.append(f"| {case} | {row['variant']} | {row['seconds']['median']*1e6:.3f} | {ratio['median']:.3f} |")
    lines+=['','Inference: rank only statistically qualified samples. Startup/storage and trajectory work are included.','',
            '| Model | Configuration | Engine | Qualified/planned | Median minimum bulk ESS/sec |',
            '|---|---|---|---:|---:|']
    for row in sampling:
        ess=row['min_bulk_ess_per_second']
        lines.append(f"| {row['case']} | {row['configuration']} | {row['engine']} | "
                     f"{row['diagnostic_passes']}/{row['planned']} | {ess['median'] if ess else 'unqualified'} |")
    lines+=['','Kernel/provider controls: matched work; gains are baseline/configuration, not Stan comparisons.','',
            '| Configuration | Kernel | Size | Thread budget | Median paired gain |',
            '|---|---|---:|---:|---:|']
    for row in kernels:
        gain=row['paired_gain']
        lines.append(f"| {row['configuration']} | {row['case']} | {row['size']} | {row['threads']} | "
                     f"{gain['median'] if gain else 'unqualified'} |")
    lines+=['','See summary.json for all outcomes, variability and pairwise evaluation ratios; progress.json retains stage failures.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def stan_math_flags(cmdstan):
    math_root=cmdstan/'stan/lib/stan_math';lib=math_root/'lib'
    return ['-O3','-std=c++17','-pthread','-DBOOST_DISABLE_ASSERTS','-DEIGEN_NO_DEBUG',
            '-I'+str(math_root),*['-I'+str(lib/p) for p in ('eigen_5.0.1','boost_1.87.0','tbb_2020.3/include','sundials_6.1.1/include')]]


def matrix_stan_build(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    cmdstan=args.cmdstan.resolve();lib=cmdstan/'stan/lib/stan_math/lib'
    commands=[]
    def build(command,label):
        row=child(command,out/label,os.environ,timeout=1800);commands.append(row)
        save(out/'builds.json',commands)
        if row['returncode']: raise ValueError(f'{label} failed; retain build logs')
    cpu_flags=['-march=native'] if args.native_cpu else []
    build([args.cxx,*stan_math_flags(cmdstan),*cpu_flags,'-MMD','-MF',out/'stan-evaluate.d',ROOT/'stan/evaluate.cpp',
           '-L'+str(lib/'tbb'),'-Wl,-rpath,'+str(lib/'tbb'),'-ltbb','-o',out/'stan-evaluate'],'math-build')
    # Make does not track changes in CXXFLAGS. Never link a native-CPU model
    # against an existing generic main.o: Eigen's alignment ABI can differ.
    # A per-study driver rebuilds with exactly the model's compiler and flags,
    # without touching the shared CmdStan installation or historical binaries.
    driver=out/'cmdstan_main.cpp'
    shutil.copy2(cmdstan/'src/cmdstan/main.cpp',driver)
    for model in dict.fromkeys(CASES[c] for c in args.cases):
        shutil.copy2(ROOT/'stan'/f'{model}.stan',out/f'{model}.stan')
        build(['make','-C',cmdstan,f'CXX={args.cxx}','PRECOMPILED_HEADERS=false',
               f'CMDSTAN_MAIN={driver}',
               *(['CXXFLAGS+=-march=native'] if args.native_cpu else []),out/model],model+'-build')
    deps=shlex.split((out/'stan-evaluate.d').read_text().replace('\\\n',' ').split(':',1)[1])
    save(out/'manifest.json',dict(compiler=command_output([args.cxx,'--version']),
        cmdstan_version=command_output([cmdstan/'bin/stanc','--version']),
        dependencies={str(Path(p).resolve()):sha(p) for p in deps},
        files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))


def feature_environment(base, features):
    """Enable compiler AD only when needed; preserve explicit user flag controls."""
    env=dict(base)
    if 'enzyme' not in features: return env
    encoded=env.get('CARGO_ENCODED_RUSTFLAGS')
    key='CARGO_ENCODED_RUSTFLAGS' if encoded else 'RUSTFLAGS'
    separator='\x1f' if encoded else ' '
    flags=encoded.split('\x1f') if encoded else shlex.split(env.get('RUSTFLAGS',''))
    if not any('autodiff=' in flag for flag in flags):
        env[key]=env.get(key,'')+separator+'-Zautodiff=Enable'
    return env


def matrix_build(args):
    """Compile/test the complete requested numerical-feature powerset serially."""
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    variants=[];builds=[]
    def inputs():
        return {str(p.relative_to(REPO)):sha(p) for directory in (REPO/'crates',ROOT)
                for p in directory.rglob('*') if p.is_file() and 'target' not in p.parts
                and (p.suffix in ('.rs','.toml','.c','.h') or p.name=='Cargo.lock')}
    source=inputs();save(out/'source.json',source)
    save(out/'qualification.json',dict(status='pending'))
    env={**os.environ,**{k:v for k,v in THREADS.items() if k not in ('XLA_FLAGS',)}}
    env['CARGO_BUILD_JOBS']=str(min(int(env.get('CARGO_BUILD_JOBS', '2')), len(os.sched_getaffinity(0))))
    compiler=command_output(['rustc','-vV'])
    axes=args.feature_axes
    if (len(axes)>8 or len(set(axes))!=len(axes)
            or any(not re.fullmatch(r'[A-Za-z0-9_-]+',f) for f in axes)):
        raise ValueError('zero to eight unique Cargo feature names required')
    if args.enzyme: axes=[axis for axis in axes if axis!='enzyme']
    for bits in itertools.product([False,True],repeat=len(axes)):
        features=[f for f,on in zip(axes,bits,strict=True) if on]
        if args.enzyme and 'enzyme' not in features: features.append('enzyme')
        label='-'.join(features) or 'default'
        flags=['--features',','.join(features)] if features else []
        cell_env=feature_environment(env,features)
        command=['cargo','test','--release','--locked','--manifest-path',ROOT/'Cargo.toml','--bins',*flags]
        row=child(command,out/(label+'-test'),cell_env,timeout=1800);builds.append(dict(variant=label,stage='model-tests',**row))
        if row['returncode']==0:
            row=child(['cargo','build','--release','--locked','--manifest-path',ROOT/'Cargo.toml','--bins',*flags],out/(label+'-build'),cell_env,timeout=1800)
            builds.append(dict(variant=label,stage='build',**row))
        binary=out/label
        if row['returncode']==0:
            shutil.copy2(ROOT/'target/release/alea-stan-bench',binary)
            shutil.copy2(ROOT/'target/release/kernel_audit',out/(label+'-kernels'))
        variants.append(dict(name=label,binary=str(binary),features=features,compiler=compiler,
            source_sha256=sha(out/'source.json'),
            build_receipt=str(out/'qualification.json'),
            flags={k:cell_env.get(k) for k in ['RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','PKG_CONFIG_PATH','LD_LIBRARY_PATH']},
            env={'ALEA_BENCH_BACKEND':'enzyme'} if 'enzyme' in features else {},build_returncode=row['returncode']))
        save(out/'builds.json',builds);save(out/'variants.json',variants)
        print(label,row['returncode'],flush=True)
    if inputs()!=source:
        save(out/'qualification.json',dict(status='source_changed'))
        raise ValueError('source changed during feature builds; do not admit these binaries')
    save(out/'qualification.json',dict(status='complete',
        source_sha256=sha(out/'source.json'),variants_sha256=sha(out/'variants.json'),
        variants={v['name']:dict(returncode=v['build_returncode'],
                  binary_sha256=sha(Path(v['binary'])) if Path(v['binary']).is_file() else None) for v in variants},
        files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*')
               if p.is_file() and p.name!='qualification.json'}))


def build_admission(variant, resolve=lambda p:Path(p), digest=None):
    """A builder's per-variant success cannot bypass its final source-freeze gate."""
    if 'source_sha256' not in variant: return None  # Explicit external binary, not this builder's receipt.
    digest_file=sha if digest is None else digest
    try:
        receipt_path=Path(variant['build_receipt']);root=receipt_path.parent
        receipt=read_metrics(resolve(receipt_path).read_text())
        if receipt['status']!='complete': raise ValueError('incomplete build cohort')
        if (digest_file(resolve(root/'source.json'))!=receipt['source_sha256']
                or receipt['source_sha256']!=variant['source_sha256']
                or digest_file(resolve(root/'variants.json'))!=receipt['variants_sha256']):
            raise ValueError('changed build inputs or variant manifest')
        record=receipt['variants'][Path(variant['binary']).name]
        if record['returncode'] or digest_file(resolve(Path(variant['binary'])))!=record['binary_sha256']:
            raise ValueError('failed or changed feature build')
        for name,expected in receipt['files'].items():
            if digest_file(resolve(root/name))!=expected: raise ValueError('changed build evidence')
    except (OSError,ValueError,KeyError,TypeError) as error:
        return 'unqualified build: '+str(error)
    return None


def build_admissions(variants, resolve=lambda p:Path(p)):
    """Verify a boundary once, sharing file reads across a cohort's variants.

    Cache lifetime is ONE admission pass, never process-global. A later replay
    rereads bytes and rejects mutations; execution also verifies live inputs at
    completion. Workload/rep loops must not rehash every binary in every cohort.
    """
    hashes={}
    def digest(path):
        path=Path(path)
        if path not in hashes: hashes[path]=sha(path)
        return hashes[path]
    return {v['name']:build_admission(v,resolve,digest) for v in variants}


def matrix_summary(plan, rows, include_pairwise=True):
    """Every planned identity remains visible; ratios require numerical admission."""
    result=[];groups={};valid={}
    for row in rows:
        key=(row['case'],row['variant'])
        groups.setdefault(key,[]).append(row)
        if row.get('admitted'): valid.setdefault(key,{})[row['rep']]=row
    for case in plan['cases']:
        reference=valid.get((case,'stan'),{})
        for variant in plan['variants']:
            attempts=groups.get((case,variant['name']),[])
            admitted=list(valid.get((case,variant['name']),{}).values())
            ratios=[reference[r['rep']]['metrics']['evaluation_seconds']/r['metrics']['evaluation_seconds']
                    for r in admitted if r['rep'] in reference]
            result.append(dict(case=case,variant=variant['name'],planned=plan['reps'],attempts=len(attempts),
                unstarted=plan['reps']-len(attempts),admitted=len(admitted),
                rejected=[r.get('reason') for r in attempts if not r.get('admitted')],
                seconds=describe([r['metrics']['evaluation_seconds']/r['metrics']['evaluations'] for r in admitted]),
                evaluations_per_second=describe([r['metrics']['evaluations']/r['metrics']['evaluation_seconds'] for r in admitted]),
                stan_over_variant=describe(ratios),
                pairwise_reference_over_variant={other['name']:describe([
                    valid[(case,other['name'])][a['rep']]['metrics']['evaluation_seconds']/a['metrics']['evaluation_seconds']
                    for a in admitted if a['rep'] in valid.get((case,other['name']),{})])
                    for other in plan['variants'] if include_pairwise and other['name']!=variant['name']}))
    return result


def matrix_probes(actual, references, dimension):
    if not isinstance(actual,list) or len(actual)!=len(references):
        raise ValueError('probe count mismatch')
    for a,b in zip(actual,references,strict=True):
        if any(not isinstance(r,dict) or not isinstance(r.get('gradient'),list)
               or not isinstance(r.get('logp'),(int,float)) for r in (a,b)):
            raise ValueError('invalid probe record')
        if (len(a['gradient'])!=dimension or len(b['gradient'])!=dimension or not close(a['logp'],b['logp'],2e-10)
                or not all(close(x,y,3e-9) for x,y in zip(a['gradient'],b['gradient'],strict=True))):
            raise ValueError('density or full gradient mismatch')
    for j,g in enumerate(actual[3]['gradient']):
        fd=(actual[4+2*j]['logp']-actual[5+2*j]['logp'])/2e-5
        if not close(g,fd,2e-6): raise ValueError('central finite difference mismatch')


def matrix_receipt(metrics, reference, count, dimension):
    expected=dict(reference)
    expected.update(evaluations=count,checksum=expected['checksum']*(count//10),
                    gradient_checksum=expected['gradient_checksum']*(count//10))
    return (metrics['evaluations']==count and metrics['evaluation_seconds']>0
            and math.isfinite(metrics['evaluation_seconds'])
            and len(metrics['gradient'])==dimension and kernel_compatible(metrics,expected))


def matrix_report(output):
    """Re-admit raw outputs; journal success/metrics and checks.json are not authority."""
    from replay import journal
    manifest=read_metrics((output/'manifest.json').read_text());plan=manifest['plan']
    def resolve(path):
        digest=manifest['files'].get(str(path));captured=output/'inputs'/str(digest)
        return captured if captured.is_file() else Path(path)
    for path,digest in manifest['files'].items():
        captured=output/'inputs'/digest
        if sha(captured if captured.is_file() else Path(path))!=digest: raise ValueError('changed feature input')
    admissions=build_admissions(plan['variants'],resolve)
    def raw(folder):
        process=read_metrics((folder/'process.json').read_text())
        if process['returncode']: raise ValueError('nonzero child exit')
        return read_metrics('{"payload":'+(folder/'stdout.txt').read_text()+'}')['payload']
    checks={};receipts={}
    for case in plan['cases']:
        probes=sorted((output/'preflight').glob(f'{case}-stan-[0-9]*'),key=lambda p:int(p.name.rsplit('-',1)[1]))
        references=[]
        for path in probes:
            if path.is_dir():
                try: references.append(raw(path))
                except (OSError,ValueError,KeyError,TypeError): references.append(None)
        try: receipts[case]=raw(output/'preflight'/f'{case}-stan-loop')
        except (OSError,ValueError,KeyError,TypeError): receipts[case]=None
        for variant in plan['variants']:
            name=variant['name']
            if 'cases' in variant and case not in variant['cases']:
                checks[(case,name)]='unsupported workload for this variant (declared before measurement)'
                continue
            admission=admissions[name]
            if admission:
                checks[(case,name)]=admission;continue
            try:
                if name=='stan': actual=references
                elif variant.get('kind')=='stan_math':
                    actual=[raw(output/'preflight'/f'{case}-{name}-{i}') for i in range(len(references))]
                else: actual=raw(output/'preflight'/f'{case}-{name}')
            except (OSError,ValueError,KeyError,TypeError,IndexError):
                checks[(case,name)]='unsupported or failed target preflight; see raw stderr'
                continue
            try:
                matrix_probes(actual,references,DIMENSIONS[case]);checks[(case,name)]=None
            except (ValueError,KeyError,TypeError,IndexError) as error: checks[(case,name)]=str(error)
    rows=[];seen=set()
    planned={(c,v['name'],r) for c in plan['cases'] for v in plan['variants'] for r in range(plan['reps'])}
    for original in journal(output/'records.jsonl'):
        key=original['case'],original['variant'],original['rep']
        if key in seen or key not in planned: raise ValueError('duplicate or unknown feature job')
        seen.add(key);case,name,rep=key
        row=dict(case=case,variant=name,rep=rep,admitted=False,reason=checks[(case,name)])
        if row['reason'] is None:
            try:
                m=raw(output/f'{case}-{rep}-{name}')
                if not matrix_receipt(m,receipts[case],plan['evaluations'],DIMENSIONS[case]): raise ValueError('failed measured work')
                row.update(admitted=True,metrics=m)
            except (OSError,ValueError,KeyError,TypeError): row['reason']='failed measured work'
        rows.append(row)
    return matrix_summary(plan,rows)


def matrix_run(plan_path, output):
    """Reuse child isolation/JSON/admission for arbitrary saved feature builds.

    Build/test separately. The plan records their logs and compiler identities;
    no builds, tests or profilers overlap this timing phase.
    """
    plan=read_metrics(plan_path.read_text())
    for case,path in plan.get('state_banks',{}).items():
        if case not in plan['cases']: raise ValueError('unknown state bank workload')
        validate_state_bank(path,DIMENSIONS[case])
    names=[v['name'] for v in plan['variants']]
    if (len(names)!=len(set(names)) or 'stan' not in names or plan['reps']<1
            or plan['evaluations']<10 or plan['evaluations']%10
            or len(set(plan['cases']))!=len(plan['cases'])):
        raise ValueError('invalid feature experiment matrix')
    cpus, threads = allocation_affinity(plan)
    output.mkdir(parents=True,exist_ok=False)
    setup_start=time.perf_counter()
    save(output/'setup.json',dict(stage='enumerating-inputs'))
    os.sched_setaffinity(0,set(cpus))
    env={**os.environ,**THREADS}
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS','STAN_NUM_THREADS']:
        env[key]=str(threads)
    # Discover the union BEFORE hashing. A dict comprehension with sha as its
    # value still hashes duplicate keys, including every file in a reused cohort.
    paths=matrix_input_paths(plan)
    save(output/'setup.json',dict(stage='hashing-inputs',input_files=len(paths)))
    files={str(p):sha(p) for p in sorted(paths)}
    (output/'inputs').mkdir()
    for path,digest in files.items():
        captured=output/'inputs'/digest
        if not captured.exists(): shutil.copy2(path,captured)
        if sha(captured)!=digest: raise ValueError('input changed while sealing experiment')
    save(output/'manifest.json',dict(plan=plan,host=host_metadata(),files=files,
        source_commit=plan.get('source_commit') or command_output(['git','-C',REPO,'rev-parse','HEAD']),
        source_diff=command_output(['git','-C',REPO,'diff','--binary']),
        policy='fixed-work full posterior; complete gradients and FD preflight; different reduction order allowed within declared tolerances; not ESS/sec'))
    save(output/'plan.json',plan)
    def captured_input(path):
        digest=files.get(str(path))
        captured=output/'inputs'/str(digest)
        return captured if captured.is_file() else Path(path)
    admissions=build_admissions(plan['variants'],captured_input)
    save(output/'setup.json',dict(stage='preflight',input_files=len(files),setup_seconds=time.perf_counter()-setup_start))
    def invoke(variant,case,folder,count,points=None):
        binary=Path(variant['binary']);model=CASES[case];data=ROOT/'data'/f'{case}.txt'
        local={**env,**variant.get('env',{})}
        for key in ['ALEA_BENCH_STATES','ALEA_BENCH_PROBES','ALEA_BENCH_KERNEL','ALEA_BENCH_STAN_CSVS']:
            local.pop(key,None)
        if points is None and case in plan.get('state_banks',{}):
            local['ALEA_BENCH_STATES']=plan['state_banks'][case]
        if variant.get('kind')=='stan_math' or variant['name']=='stan':
            command=[binary,model,data,count,variant.get('implementation','vectorized'),*(points or [])]
        else:
            command=[binary,model,data]
            local['ALEA_BENCH_EVALUATIONS']=str(count)
            if points is not None:
                probe=folder.parent/(folder.name+'-points.json');save(probe,points)
                local['ALEA_BENCH_PROBES']=str(probe)
        row=child(command,folder,local)
        try: row['metrics']=read_metrics('{"payload":'+(folder/'stdout.txt').read_text()+'}')['payload']
        except (ValueError,OSError): row['metrics']=None
        save(folder/'process.json',row)
        return row
    checks={}; receipts={}
    preflight=output/'preflight';preflight.mkdir()
    stan=next(v for v in plan['variants'] if v['name']=='stan')
    for case in plan['cases']:
        d=DIMENSIONS[case]
        points=[[0.]*d,[.3]*d,[-1.5]*d]
        # Every coordinate, asymmetric states, and central differences. Keep
        # extreme rejection evidence separate from valid-target timing admission.
        q=[.07*(j+1) for j in range(d)]
        points.append(q)
        for j in range(d):
            for sign in [1,-1]:
                p=q.copy();p[j]+=sign*1e-5;points.append(p)
        positive = [2,3] if case=='covariance' else [d-1] if case in ('schools','schools_nc','mean','linear','linear_large') else []
        for index in positive:
            for value in [-20.,20.]:
                p=q.copy();p[index]=value;points.append(p)
        if case.startswith('logistic'):
            points.extend([[1000.]*d,[-1000.]*d])
        references=[]
        receipt=invoke(stan,case,preflight/f'{case}-stan-loop',10)
        receipts[case]=receipt['metrics'] if receipt['returncode']==0 else None
        for i,q in enumerate(points):
            row=invoke(stan,case,preflight/f'{case}-stan-{i}',1,q)
            references.append(row.get('metrics') if row['returncode']==0 else None)
        for variant in plan['variants']:
            reason=None
            if 'cases' in variant and case not in variant['cases']:
                actual=None;reason='unsupported workload for this variant (declared before measurement)'
            elif admissions[variant['name']]:
                actual=None;reason=admissions[variant['name']]
            elif variant['name']=='stan': actual=references
            elif variant.get('kind')=='stan_math':
                actual=[]
                for i,q in enumerate(points):
                    row=invoke(variant,case,preflight/f'{case}-{variant["name"]}-{i}',1,q)
                    actual.append(row['metrics'] if row['returncode']==0 else None)
            else:
                row=invoke(variant,case,preflight/f'{case}-{variant["name"]}',1,points)
                actual=row['metrics']
                if row['returncode'] or not isinstance(actual,list): reason='unsupported or failed target preflight; see raw stderr'
            try:
                if reason: raise ValueError(reason)
                matrix_probes(actual,references,d)
            except (ValueError,TypeError,KeyError,IndexError) as error: reason=str(error)
            checks[(case,variant['name'])]=reason
    save(output/'checks.json',[dict(case=c,variant=v,reason=r,passed=r is None) for (c,v),r in checks.items()])
    rows=[]
    save(output/'summary.json',matrix_summary(plan,rows,include_pairwise=False))
    with (output/'records.jsonl').open('x') as journal:
        for case in plan['cases']:
            for rep in range(plan['reps']):
                variants=plan['variants'];order=variants[rep%len(variants):]+variants[:rep%len(variants)]
                for variant in order:
                    reason=checks[(case,variant['name'])]
                    row=dict(case=case,variant=variant['name'],rep=rep,admitted=False,reason=reason)
                    if reason is None:
                        raw=invoke(variant,case,output/f'{case}-{rep}-{variant["name"]}',plan['evaluations'])
                        row.update(raw)
                        try:
                            m=row['metrics']
                            if row['returncode'] or not matrix_receipt(m,receipts[case],plan['evaluations'],DIMENSIONS[case]):
                                raise ValueError('failed or incomplete measured work')
                            # Compare changing-state receipts to the admitted
                            # reference; do not trust preflight for loop identity.
                            row['admitted']=True
                        except (ValueError,TypeError,KeyError): row['reason']='failed measured work'
                    rows.append(row);journal.write(json.dumps(row,allow_nan=False)+'\n');journal.flush()
                    print(case,rep,variant['name'],row['admitted'],flush=True)
                save(output/'summary.json',matrix_summary(plan,rows,include_pairwise=False))
    for path,digest in files.items():
        if sha(Path(path))!=digest: raise ValueError('sealed binary/data changed during measurement')
    save(output/'summary.json',matrix_summary(plan,rows))


def matrix_input_paths(plan):
    """Union build evidence by path before any expensive content reads."""
    paths={Path(v['binary']) for v in plan['variants'] if Path(v['binary']).is_file()}
    paths.update(ROOT/'data'/f'{case}.txt' for case in plan['cases'])
    paths.update(Path(p) for p in plan.get('state_banks',{}).values())
    receipts={Path(v['build_receipt']) for v in plan['variants'] if v.get('build_receipt')}
    for receipt_path in receipts:
        if not receipt_path.is_file(): continue
        receipt=read_metrics(receipt_path.read_text())
        paths.update(p for p in [receipt_path,*[receipt_path.parent/n for n in receipt.get('files',{})]] if p.is_file())
    return paths


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
    flags = stan_math_flags(args.cmdstan)
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
                good=functional_agreement(a['metrics']['diagnostics'],b['metrics']['diagnostics'])
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
    parser.add_argument('action',choices=['prepare','run','report','matrix','matrix-build','matrix-report','matrix-stan-build','matrix-plan','profile','slurm-submit','suite','suite-report','cargo-inventory','cargo-report'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cmdstan',type=Path)
    parser.add_argument('--python',type=Path)
    parser.add_argument('--cxx',type=Path)
    parser.add_argument('--reuse-build',type=Path,help='reuse sealed executables only when every compiled source and toolchain identity matches')
    parser.add_argument('--matrix-plan',type=Path,help='saved feature builds and compiler/test receipts; no concurrent builds')
    parser.add_argument('--enzyme',action='store_true',help='matrix-build under the pinned autodiff toolchain')
    parser.add_argument('--feature-axes',nargs='*',default=['simd','faer','openblas','enzyme','branch-hints','tracing','analysis'],
                        help='full feature powerset by default; use the pinned autodiff compiler for Enzyme')
    parser.add_argument('--cases',nargs='+',choices=list(CASES))
    parser.add_argument('--phases',nargs='+',choices=['evaluate','sampling'],default=['evaluate','sampling'])
    parser.add_argument('--cpu',type=int,help='default: first CPU in the scheduler allocation for matrix plans')
    parser.add_argument('--builds',nargs='+',type=Path,help='saved variants.json files for matrix-plan')
    parser.add_argument('--stan-evaluate',type=Path,help='Stan Math reference executable for matrix-plan')
    parser.add_argument('--state-bank-dir',type=Path,help='optional ten-state <case>.txt banks; historical coordinate-zero loop remains the default')
    parser.add_argument('--perf-command',default='perf',help='perf executable/prefix, e.g. "sudo -n /path/to/perf"; never changes sysctl policy')
    parser.add_argument('--local-profile',action='store_true',help='explicit diagnostic-only exception to remote profiling policy')
    parser.add_argument('--profile-variants',nargs='+',help='saved matrix variant names to profile')
    parser.add_argument('--partition',default='hpc2019')
    parser.add_argument('--exclude-nodes',nargs='+',help='explicit compute nodes to keep profiling/build jobs off a timed host')
    parser.add_argument('--account')
    parser.add_argument('--afterok',help='run only after this preparation job succeeds')
    parser.add_argument('--afterany',nargs='+',help='wait for these Slurm jobs to finish regardless of outcome (avoid overlapping measurements)')
    parser.add_argument('--cpus',type=int,default=18,help='Slurm CPU budget')
    parser.add_argument('--walltime',default='24:00:00')
    parser.add_argument('--shared-node',action='store_true',help='explicitly permit shared-node execution; exclusive by default')
    parser.add_argument('--memory',help='Slurm memory budget, e.g. 8G; required for shared-node preparation')
    parser.add_argument('--dry-run',action='store_true',help='write submission script without contacting Slurm')
    parser.add_argument('--job-command',nargs=argparse.REMAINDER,help='existing command to execute in the allocation; specify last')
    parser.add_argument('--job-timeout',type=int,default=432000,help='suite stage watchdog seconds, not a timing subtraction')
    parser.add_argument('--toolchains',nargs='+',default=['1.95.0','nightly-2026-06-23'],help='suite compiler cohorts, kept distinct')
    parser.add_argument('--native-cpu',action='store_true',help='suite/Stan builds: specialize both engines to the build compute-node CPU')
    parser.add_argument('--reps',type=int,default=7)
    parser.add_argument('--kernel-reps',type=int,default=7)
    parser.add_argument('--evaluations',type=int,default=10000)
    args=parser.parse_args()
    if args.cases is None:
        args.cases=list(CASES) if args.action in ('matrix-plan','matrix-stan-build','suite') else DEFAULT_CASES
    if len(args.phases)!=len(set(args.phases)): parser.error('duplicate phases')
    if args.action=='cargo-inventory': save(args.output,cargo_bench_inventory(REPO))
    elif args.action=='cargo-report': save(args.output/'summary.json',cargo_bench_report(args.output))
    elif args.action=='suite-report': suite_report(args.output.resolve())
    elif args.action=='suite':
        if 'sampling' in args.phases and not args.cmdstan: parser.error('sampling requires --cmdstan')
        suite_run(args)
    elif args.action=='slurm-submit': slurm_submit(args)
    elif args.action=='profile':
        if args.matrix_plan is None: parser.error('profile requires --matrix-plan')
        profile_run(args)
    elif args.action=='matrix-plan':
        if not args.builds or not args.stan_evaluate: parser.error('matrix-plan requires --builds and --stan-evaluate')
        matrix_plan(args)
    elif args.action=='matrix-stan-build':
        if not args.cmdstan or not args.cxx: parser.error('Stan build requires --cmdstan and --cxx')
        matrix_stan_build(args)
    elif args.action=='matrix-report': print(json.dumps(matrix_report(args.output),indent=2,allow_nan=False))
    elif args.action=='matrix-build': matrix_build(args)
    elif args.action=='matrix':
        if args.matrix_plan is None: parser.error('matrix requires --matrix-plan')
        matrix_run(args.matrix_plan,args.output)
    elif args.action=='prepare':
        if args.cpu is None: args.cpu=min(os.sched_getaffinity(0))
        if any(c not in DEFAULT_CASES for c in args.cases):
            parser.error('new conformance cases use matrix; JAX ports are not supplied for them')
        if not all((args.cmdstan,args.python,args.cxx)) or min(args.reps,args.kernel_reps)<1 or args.evaluations<10 or args.evaluations%10:
            parser.error('prepare requires pinned paths, positive repetitions and a positive multiple-of-ten evaluation count')
        prepare(args)
    elif args.action=='run': run(args)
    else: print(json.dumps(report(args.output.resolve()),indent=2,allow_nan=False))


if __name__=='__main__': main()
