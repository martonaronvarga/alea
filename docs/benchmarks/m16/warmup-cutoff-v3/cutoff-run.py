"""Serial cutoff-only warmup ablation, queued after the sealed full matrix."""
import collections,datetime,gzip,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
BASE=Path('/home/usu/alea-bench-runs/m16-posteriordb-complete-20261022')
ROOT=BASE/'source';OUT=BASE/'cutoff-ablation'
sys.path.insert(0,str(ROOT/'tools/bench'))
from alea_bench.manifest import file_hash,write_new
from alea_bench.process import isolated
from alea_bench.runner import environment,compress_raw
from alea_bench.pdb_report import coordinate_summary,reference_comparison,work_metrics

def status(stage,**kwargs):
 value=dict(stage=stage,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kwargs)
 temporary=BASE/'cutoff-status.tmp';temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(BASE/'cutoff-status.json');print(stage,kwargs,flush=True)
try:
 main=json.loads((BASE/'status.json').read_text());assert main['stage']=='complete','preserve serial measurements'
 spec=json.loads((BASE/'cutoff-spec.json').read_text());build=json.loads((BASE/'cutoff-build.json').read_text());binary=BASE/'alea-pdb-cutoff-worker'
 assert file_hash(binary)==spec['binary_sha256']==build['binary_sha256']
 assert file_hash(BASE/'cutoff-build.json')==spec['build_sha256']
 OUT.mkdir();write_new(OUT/'seal.json',dict(spec_sha256=file_hash(BASE/'cutoff-spec.json'),build_sha256=file_hash(BASE/'cutoff-build.json'),source_archive_sha256=file_hash(BASE/'cutoff-source.tar.gz')))
 cpu=min(os.sched_getaffinity(0));rows=[];index={}
 for rep in range(3):
  for case in ['earnings-earn_height','sblri-blr','arK-arK']:
   original=BASE/'ablation/jobs'/f'{case}--nuts_legacy_warmup--r{rep}'
   for method in (['nuts_legacy_warmup','nuts_legacy_downhill'] if rep%2==0 else ['nuts_legacy_downhill','nuts_legacy_warmup']):
    folder=OUT/f'{case}--{method}--r{rep}';folder.mkdir();job=json.loads((original/'job.json').read_text());job.update(method=method,directory=str(folder));write_new(folder/'job.json',job)
    command=[str(binary),'sample',str(folder/'job.json')]
    wrapped=['time','-f','%M','-o',str(folder/'rss.txt'),'taskset','-c',str(cpu),*command]
    with (folder/'process.log').open('x') as log:process=isolated(wrapped,cwd=ROOT,env=environment(ROOT),log=log,timeout=300)
    worker=json.loads((folder/'worker.json').read_text()) if (folder/'worker.json').exists() else None
    complete=process['returncode']==0 and not process['timed_out'];metrics=None
    if complete:
     subprocess.run([str(binary),'diagnose',str(folder/'job.json')],cwd=ROOT,check=True,timeout=60);metrics=json.loads((folder/'metrics.json').read_text())
    for path in folder.glob('*.csv'):compress_raw(path)
    artifacts={p.name:dict(sha256=file_hash(p),bytes=p.stat().st_size) for p in folder.iterdir() if p.is_file()}
    result=dict(job=job,process=process,worker=worker,metrics=metrics,status='completed' if complete else 'failed',artifacts=artifacts)
    write_new(folder/'result.json',result);index[folder.name]=file_hash(folder/'result.json')
    diag=coordinate_summary(metrics['unconstrained']) if metrics else {};reference=json.loads((ROOT/'tools/bench/corpus/pdb-v2'/case/'reference-summary.json').read_text());compare=reference_comparison(metrics['constrained'],reference) if metrics else []
    phases=(worker or {}).get('phases',[]);aggregate=lambda key:sum(p.get(key,0.) for p in phases)
    row=dict(case=case,method=method,repetition=rep,status=result['status'],failure=(worker or {}).get('failure'),process_seconds=process['wall_seconds'],warmup_seconds=aggregate('warmup_seconds'),retained_seconds=aggregate('retained_seconds'),diagnostics=diag,maximum_abs_mean_error_reference_sd=max((abs(c['mean_error_reference_sd']) for c in compare),default=None),maximum_abs_sd_ratio_error=max((abs(c['posterior_sd_ratio']-1) for c in compare if c['posterior_sd_ratio'] is not None),default=None))
    row.update(work_metrics(worker or {},diag.get('minimum_bulk_ess'),complete))
    if method=='nuts_legacy_warmup' and complete:
     old=json.loads((original/'result.json').read_text());row['control_draws_identical_to_original_ablation']=artifacts['draws.csv.gz']['sha256']==old['artifacts']['draws.csv.gz']['sha256']
    rows.append(row);status('sampling',completed_jobs=len(rows),last_job=folder.name,last_status=result['status'])
 write_new(OUT/'index.json',index);write_new(OUT/'report.json',dict(spec=spec,summary=dict(planned_jobs=18,status_counts=dict(collections.Counter(r['status'] for r in rows))),rows=rows))
 write_new(OUT/'checksums.json',{str(p.relative_to(OUT)):file_hash(p) for p in OUT.rglob('*') if p.is_file()})
 status('complete',summary=json.loads((OUT/'report.json').read_text())['summary'])
except BaseException as error:
 status('failed',error=type(error).__name__+': '+str(error));traceback.print_exc();raise
