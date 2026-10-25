"""Fresh, serial subprocess execution with durable failure records and RSS scope."""
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import tarfile
import time
from . import models, schema
from .manifest import ROOT, file_hash, task_text, verify_sources, write_new
from .process import isolated


def host():
    cpu='unknown'
    try:cpu=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
    except (OSError,StopIteration):pass
    identity=dict(cpu=cpu,machine=platform.machine(),system=platform.system(),node_hash=hashlib.sha256(platform.node().encode()).hexdigest())
    return dict(identity,fingerprint=schema.digest(identity),kernel=platform.release(),logical_cpus=os.cpu_count(),controller_python=platform.python_version())


def jobs(manifest):
    p=manifest['protocol'];cases=manifest['corpus']['cases'];out=[]
    for rep in range(p['policy']['repetitions']):
        for i,case in enumerate(cases):
            # Deterministic rotation counterbalances engine order without tuning.
            engines=p['engines'];offset=(rep+i)%len(engines)
            for engine in engines[offset:]+engines[:offset]:
                out.append(dict(id=f'{case["id"]}-{engine}-r{rep}',case=case,engine=engine,repetition=rep,dataset=None,truth=None))
    config=p['calibration'];by_id={c['id']:c for c in cases}
    for name in config['cases']:
        for dataset in range(config['datasets']):
            case,truth=models.generate(by_id[name],config['seed']+10000*cases.index(by_id[name])+dataset)
            out.append(dict(id=f'sbc-{name}-{dataset}',case=case,engine='alea',repetition=0,dataset=dataset,truth=truth))
    return out


def gradient_preflight(case,binary,policy,directory,thresholds):
    """Check actual Rust values/forces against an independent density at 3 points."""
    checks=[]
    for index in range(3):
        c=dict(case);c['initial']=[v+.07*(index-1)*(i+1) for i,v in enumerate(case['initial'])]
        task=directory/f'probe-{index}.task';task.write_text(task_text(c,policy));output=directory/f'probe-{index}.json'
        subprocess.run([binary,'--probe',str(task),str(output)],check=True,stdout=subprocess.DEVNULL,timeout=30)
        observed=schema.load(output);expected=models.density(c,c['initial'])
        if observed['log_density'] is None or abs(observed['log_density']-expected)>1e-8*(1+abs(expected)):raise schema.Invalid('independent log-density measure mismatch')
        for i,g in enumerate(observed['gradient']):
            plus=list(c['initial']);minus=list(plus);h=1e-5*max(1.,abs(plus[i]));plus[i]+=h;minus[i]-=h
            finite=(models.density(c,plus)-models.density(c,minus))/(2*h)
            limit=thresholds['gradient_absolute_tolerance']+thresholds['gradient_relative_tolerance']*max(abs(finite),abs(g) if g is not None else 0)
            error=abs(g-finite) if g is not None else math.inf
            checks.append({'point':index,'coordinate':i,'absolute_error':error if math.isfinite(error) else None,'limit':limit,'passed':math.isfinite(error) and error<=limit})
    return dict(passed=all(c['passed'] for c in checks),checks=checks,oracle='independent Python density central differences; Wiener spectral values')


def archive_sources(manifest,directory,root):
    path=directory/'sources.tar.gz'
    with path.open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as compressed,tarfile.open(fileobj=compressed,mode='w|') as archive:
        for name in sorted(manifest['sources']):
            data=(root/name).read_bytes();info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;info.mtime=0;archive.addfile(info,io.BytesIO(data))
    return file_hash(path)


def archive_preparation(config_path,directory):
    files=sorted(set([config_path,*config_path.parent.glob('*.log'),*config_path.parent.glob('*.build.json')]))
    with (directory/'preparation.tar.gz').open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as compressed,tarfile.open(fileobj=compressed,mode='w|') as archive:
        for path in files:
            data=path.read_bytes();info=tarfile.TarInfo(path.name);info.size=len(data);info.mode=0o644;info.mtime=0;archive.addfile(info,io.BytesIO(data))
    return file_hash(directory/'preparation.tar.gz')


def environment(root):
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',STAN_NUM_THREADS='1',JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1')
    env['PYTHONPATH']=str(root/'tools/bench')+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return env


def phases(path,chains):
    with path.open() as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=chains or [int(r['chain']) for r in rows]!=list(range(chains)):raise schema.Invalid('lost/duplicate phase records')
    names=set(rows[0])-{'chain'};result={}
    for name in names:
        values=[float(r[name]) if r[name] else None for r in rows]
        if any(v is not None and (not math.isfinite(v) or v<0) for v in values):raise schema.Invalid('invalid phase time/work counter')
        result[name]=sum(values) if all(v is not None for v in values) else None
    return result


def compress_raw(path):
    # Raw content is preserved losslessly with a deterministic gzip header.
    with path.open('rb') as source,path.with_suffix(path.suffix+'.gz').open('xb') as target,gzip.GzipFile(fileobj=target,mode='wb',mtime=0,filename='') as compressed:
        shutil.copyfileobj(source,compressed)
    path.unlink()


def measure_job(manifest,job,config,directory,host_record,cpu,root):
    directory.mkdir();p=manifest['protocol'];policy=p['policy'];case=job['case'];engine=job['engine'];binary=config['binaries'][engine]
    task=directory/'task.txt';task.write_text(task_text(case,policy));write_new(directory/'case.json',case);write_new(directory/'policy.json',policy)
    record=dict(schema_version=1,manifest_sha256=manifest['manifest_sha256'],job=job,host=host_record,engine=engine,status='failed',failure=None,
                process=dict(returncode=None,wall_seconds=None,peak_rss_kib=None,cpu_affinity=[cpu],command=[],binary_sha256=config['binary_sha256'][engine]),phases=None,metrics=None,preflight=None,artifacts={},availability={})
    started=None;process=None
    try:
        verify_sources(manifest,root)
        record['preflight']=gradient_preflight(case,config['binaries']['alea'],policy,directory,p['thresholds'])
        if not record['preflight']['passed']:raise schema.Invalid('wrong gradient: independent preflight failed')
        if file_hash(Path(binary))!=config['binary_sha256'][engine]:raise schema.Invalid('prepared engine binary changed')
        if engine=='alea':command=[binary,'--worker',str(task),str(directory)]
        elif engine=='nuts-rs':command=[binary,str(task),str(directory)]
        else:
            interpreter=config['binaries']['blackjax'] if engine=='blackjax' else shutil.which('python3')
            command=[interpreter,'-m','alea_bench.adapters',engine,str(directory/'case.json'),str(directory/'policy.json'),str(directory)]
            if engine=='stan':command.append(binary)
        timer=shutil.which('time');affinity=shutil.which('taskset')
        if not timer or not affinity:raise RuntimeError('GNU time and taskset required for isolated v1 measurements')
        full=[timer,'-f','%M','-o',str(directory/'rss.txt'),affinity,'-c',str(cpu),*command]
        record['process']['command']=command;started=time.perf_counter()
        with (directory/'process.log').open('x') as log:
            measured=isolated(full,cwd=root,env=environment(root),log=log,timeout=p['resources']['timeout_seconds'])
        exitcode=measured['returncode'];record['process']['wall_seconds']=measured['wall_seconds'];record['process']['returncode']=exitcode
        rss=(directory/'rss.txt').read_text().splitlines()
        if rss and rss[-1].isdigit():record['process']['peak_rss_kib']=int(rss[-1])
        if measured['timed_out']:raise RuntimeError('worker process-group deadline exhausted; direct worker reaped')
        if exitcode:raise RuntimeError(f'worker failed with exit {exitcode}')
        if sum(f.stat().st_size for f in directory.iterdir() if f.is_file())>p['resources']['maximum_output_bytes']:raise RuntimeError('worker output byte budget exceeded')
        verify_sources(manifest,root)
        record['phases']=phases(directory/'phases.csv',policy['chains'])
        subprocess.run([config['binaries']['alea'],'--analyze',str(task),str(directory/'draws.csv'),str(directory/'metrics.json')],check=True,timeout=30,stdout=subprocess.DEVNULL)
        record['metrics']=schema.load(directory/'metrics.json')
        if (directory/'engine.json').exists():record['availability']['engine']=schema.load(directory/'engine.json')
        record['availability'].update(energy_error_scope='Alea: maximum finite signed increase including zero; nuts-rs: maximum absolute error; Stan/BlackJAX: unavailable',hvp_calls={'value':None,'reason':'not requested by this Euclidean NUTS protocol'},allocations={'value':None,'reason':'allocator calls are not instrumented; process peak RSS is measured per fresh job'},metric_actions={'scope':'Alea actual velocity/momentum operations' if engine=='alea' else 'not exposed by upstream adapter'},gradient_calls={'scope':'actual target callback calls' if engine in ('alea','nuts-rs') else 'not exposed; never inferred from integration steps'},timing_scope='retained excludes serialization for Alea/nuts-rs/BlackJAX; Stan native includes CSV; common comparison uses fresh process wall')
        record['status']='completed'
    except Exception as error:
        record['failure']={'kind':type(error).__name__,'message':str(error)}
    finally:
        if started is not None and record['process']['wall_seconds'] is None:record['process']['wall_seconds']=time.perf_counter()-started
        if process is not None and record['process']['returncode'] is None:record['process']['returncode']=process.returncode
        if (directory/'draws.csv').exists():compress_raw(directory/'draws.csv')
        # Native Stan originals also remain byte-for-byte recoverable.
        for native in directory.glob('stan-*.csv'):compress_raw(native)
        record['artifacts']={f.name:{'sha256':file_hash(f),'bytes':f.stat().st_size} for f in sorted(directory.iterdir()) if f.is_file()}
        schema.result(record);write_new(directory/'result.json',record)
    return record


def run(manifest_path,config_path,output,root=ROOT):
    manifest=schema.manifest(schema.load(manifest_path));verify_sources(manifest,root);config=schema.load(config_path)
    for name,sha in config.get('rust_sources',{}).items():
        if name not in manifest['sources'] or manifest['sources'][name]!=sha:raise schema.Invalid('prepared binary source does not match sealed experiment')
    for engine in manifest['protocol']['engines']:
        if engine not in config['binaries']:raise schema.Invalid(f'missing prepared engine: {engine}')
    output.mkdir(parents=True,exist_ok=False);write_new(output/'manifest.json',manifest);shutil.copyfile(config_path,output/'engines.json')
    source_hash=archive_sources(manifest,output,root);preparation_hash=archive_preparation(config_path,output);host_record=host();write_new(output/'host.json',host_record)
    affinity=sorted(os.sched_getaffinity(0));cpu=affinity[0];planned=jobs(manifest);records=[]
    (output/'jobs').mkdir()
    # Write intent first; interruption cannot turn missing failed jobs into passes.
    write_new(output/'plan.json',{'schema_version':1,'manifest_sha256':manifest['manifest_sha256'],'jobs':[j['id'] for j in planned],'source_archive_sha256':source_hash,'preparation_archive_sha256':preparation_hash})
    for i,job in enumerate(planned):
        record=measure_job(manifest,job,config,(output/'jobs'/job['id']).resolve(),host_record,cpu,root);records.append(record)
        with (output/'progress.jsonl').open('a') as log:log.write(json.dumps({'id':job['id'],'status':record['status'],'failure':record['failure']},allow_nan=False)+'\n')
        print(f'{i+1}/{len(planned)} {job["id"]}: {record["status"]}',flush=True)
    write_new(output/'index.json',{'schema_version':1,'manifest_sha256':manifest['manifest_sha256'],'records':{r['job']['id']:file_hash(output/'jobs'/r['job']['id']/'result.json') for r in records},'source_archive_sha256':source_hash,'preparation_archive_sha256':preparation_hash})
    return records
