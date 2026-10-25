"""Version-two descriptive experiments: complete matrices, no promotion decisions."""
import argparse,csv,gzip,io,json,math,os,re,shutil,subprocess,sys,tarfile,time
from pathlib import Path
from . import schema
from .manifest import ROOT,file_hash,source_paths,verify_sources,write_new
from .process import isolated
from .runner import host,environment,archive_sources,compress_raw

NATIVE=['nuts_cov_diag','nuts_cov_dense','nuts_fisher_diag','nuts_fisher_rank4','nuts_fisher_rm','nuts_fisher_adam','nuts_atlas_joint','nuts_pathfinder','nuts_transport','hmc_cov','hmc_omf','hmc_bcss','ghmc','drghmc','mams','mclmc','chees','walnuts_d','walnuts_r2p','rmhmc_softabs','rmhmc_manifold']
EXTERNAL=['stan_nuts_diag','stan_nuts_dense','stan_pathfinder_nuts','blackjax_nuts','nuts_rs']
NATIVE_V3=[m for m in NATIVE if m not in ('nuts_atlas_joint','nuts_pathfinder')]+['nuts_saia_joint','nuts_pathfinder_raw','nuts_pathfinder_psis','atlas_full','nuts_legacy_warmup','nuts_grad_diag','nuts_grad_diag_jitter','nuts_grad_diag_position']
METHODS=NATIVE+EXTERNAL
METHODS_V3=NATIVE_V3+EXTERNAL
WORKER=ROOT/'tools/bench/pdb-worker/target/release/alea-pdb-worker'
JAX=Path('/nix/store/9cx7xm1vzsm2whddkxz23a6wma1f99df-python3-3.13.12-env/bin/python3')

def validate_protocol(value):
 schema.keys(value,{'schema_version','id','purpose','corpus','methods','chains','seeds','warmup','draws','repetitions','depth','timeout_seconds','maximum_output_bytes','rmhmc','notes'}|({'cases'} if value.get('schema_version')==3 else set())|({key for key in ('report_work_metrics','record_warmup_trace') if key in value} if value.get('schema_version')==3 else set()),'descriptive protocol')
 if type(value['schema_version']) is not int or value['schema_version'] not in (2,3) or value['purpose']!='descriptive':raise schema.Invalid('requires descriptive version-two/three protocol')
 schema.relative(value['corpus'])
 for key in ('report_work_metrics','record_warmup_trace'):
  if key in value and type(value[key]) is not bool:raise schema.Invalid(key+' must be boolean')
 if value['schema_version']==3:
  if not isinstance(value['cases'],list) or not value['cases'] or any(not isinstance(c,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',c) for c in value['cases']) or len(set(value['cases']))!=len(value['cases']):raise schema.Invalid('invalid case subset')
 if not value['methods'] or len(set(value['methods']))!=len(value['methods']) or any(m not in (METHODS_V3 if value['schema_version']==3 else METHODS) for m in value['methods']):raise schema.Invalid('unsupported/duplicate method')
 for key,lo,hi in [('chains',2,16),('warmup',1,100000),('draws',4,100000),('repetitions',1,20),('depth',1,12),('timeout_seconds',1,86400),('maximum_output_bytes',4096,1073741824)]:schema.integer(value[key],lo,hi,key)
 if len(value['seeds'])!=value['chains'] or len(set(value['seeds']))!=value['chains']:raise schema.Invalid('seed identities')
 for s in value['seeds']:schema.integer(s,1,2**31-1,'seed')
 schema.keys(value['rmhmc'],{'maximum_dimension','warmup','draws','timeout_seconds'},'RMHMC budget')
 for key,lo,hi in [('maximum_dimension',1,128),('warmup',1,10000),('draws',4,10000),('timeout_seconds',1,86400)]:schema.integer(value['rmhmc'][key],lo,hi,key)
 if not isinstance(value['notes'],list) or not value['notes'] or any(not isinstance(n,str) or not n.strip() for n in value['notes']):raise schema.Invalid('accounting/method notes required')
 if max(value['seeds'])+10000*(value['repetitions']-1)>=2**31:raise schema.Invalid('repetition seed overflow')
 return value


def prepare_metadata(models,output,worker_env=None):
 """Bind compiled models, finite initial points and worker binary before sealing."""
 models=models.resolve();output=output.resolve();output.mkdir(parents=True,exist_ok=False)
 records=schema.load(models/'models.json');corpus=schema.load(ROOT/'tools/bench/corpus/pdb-v2.json');cases={c['id']:c for c in corpus['cases']};prepared={}
 for record in records['records']:
  case=cases[record['case']];row=dict(record)
  bridge=row['builds']['bridge']
  if bridge['returncode']==0:
   config=output/f'{case["id"]}.input.json';write_new(config,{'library':bridge['binary'],'data':str(ROOT/case['directory']/'data.json')});metadata=output/f'{case["id"]}.metadata.json'
   with (output/f'{case["id"]}.log').open('x') as log:p=subprocess.run([str(WORKER),'metadata',str(config),str(metadata)],env=worker_env,stdout=log,stderr=subprocess.STDOUT)
   row['metadata_status']=p.returncode
   if p.returncode==0:row['metadata']=schema.load(metadata)
  else:row['metadata_status']=None
  prepared[case['id']]=row
 rust={p.relative_to(ROOT).as_posix():file_hash(p) for p in source_paths(ROOT) if p.suffix in ('.rs','.toml','.lock','.nix','.c','.h','.cpp','.hpp')}
 result={'schema_version':2,'models':prepared,'worker':str(WORKER),'worker_sha256':file_hash(WORKER),'jax_python':str(JAX),'jax_python_sha256':file_hash(JAX),'build_records':records,'compiled_source_inputs':rust,'upstream':{'bridgestan':records['bridgestan_revision'],'stan':records['stan_version'],'nuts_rs':'a762aae513bf7bb5ccb27ffc9a938a923fc837ec'},'metadata_directory':str(output),'model_directory':str(models)}
 write_new(output/'prepared.json',result);return result


def seal(protocol,prepared,output):
 p=validate_protocol(schema.load(protocol));corpus=schema.load(ROOT/p['corpus']);config=schema.load(prepared)
 if corpus['schema_version']!=2 or set(config['models'])!={c['id'] for c in corpus['cases']}:raise schema.Invalid('model inventory drift')
 paths=set(source_paths(ROOT));paths.update((ROOT/'tools/bench/corpus/pdb-v2').rglob('*'));paths={x for x in paths if x.is_file()};paths.add(protocol.resolve())
 if 'native' in config:paths.update(ROOT/name for name in config['native']['dependency_sources'])
 if p['id'].startswith('m16-posteriordb-native-') and 'native' not in config:raise schema.Invalid('native protocol requires native preparation; no implicit BridgeStan fallback')
 if 'native' in config:
  for case in p['cases']:
   metadata=config['models'][case].get('metadata')
   if metadata:select_initials(metadata,p['chains'])
 sources={x.relative_to(ROOT).as_posix():file_hash(x) for x in sorted(paths)}
 if p['schema_version']==3 and config.get('third_ad_source_sha256')!=sources.get('tools/bench/third_ad.cpp'):raise schema.Invalid('third-AD compiled source drift')
 for case in corpus['cases']:
  for name,sha in case['files'].items():
   if sources.get(f'{case["directory"]}/{name}')!=sha:raise schema.Invalid('vendored model/data/reference drift')
 for name,sha in config['compiled_source_inputs'].items():
  if sources.get(name)!=sha:raise schema.Invalid('compiled source drift')
 manifest={'schema_version':p['schema_version'],'purpose':'descriptive','protocol':p,'corpus':corpus,'preparation':config,'sources':sources,'toolchains':{'rust':'1.95.0 via flake.lock','stan':'2.38.0 via flake.lock','bridgestan':'2.7.0','jax':'0.9.2','blackjax':'1.5'},'reference_defaults_changed':p['schema_version']==3}
 if 'native' in config:manifest['toolchains']['rust']=config['native']['compiler']
 manifest['manifest_sha256']=schema.digest(manifest);validate_manifest(manifest);write_new(output,manifest);return manifest


def validate_manifest(m):
 schema.keys(m,{'schema_version','purpose','protocol','corpus','preparation','sources','toolchains','reference_defaults_changed','manifest_sha256'},'descriptive manifest')
 if type(m['schema_version']) is not int or m['schema_version'] not in (2,3) or m['purpose']!='descriptive' or m['reference_defaults_changed'] is not (m['schema_version']==3):raise schema.Invalid('descriptive manifest identity')
 validate_protocol(m['protocol'])
 native=m['preparation'].get('native')
 if native:
  if m['schema_version']!=3 or native.get('rust_backend') not in ('enzyme','bridgestan') or native.get('execution') not in ('serial','parallel'):raise schema.Invalid('invalid native backend/execution identity')
  if not isinstance(native.get('conformance'),dict):raise schema.Invalid('missing native conformance inventory')
 if m['protocol']['schema_version']!=m['schema_version']:raise schema.Invalid('manifest/protocol version mismatch')
 if m['schema_version']==3 and not set(m['protocol']['cases'])<={c['id'] for c in m['corpus']['cases']}:raise schema.Invalid('unknown selected posterior')
 corpus=m['corpus'];cases=corpus.get('cases',[])
 if corpus.get('schema_version')!=2 or not cases or len(cases)>128 or len({c['id'] for c in cases})!=len(cases) or set(m['preparation']['models'])!={c['id'] for c in cases}:raise schema.Invalid('invalid posterior inventory')
 for case in cases:
  if not re.fullmatch(r'[A-Za-z0-9_-]+',case['id']):raise schema.Invalid('invalid posterior identifier')
  schema.relative(case['directory'])
  for name,sha in case['files'].items():
   schema.relative(name);schema.sha(sha)
   if m['sources'].get(case['directory']+'/'+name)!=sha:raise schema.Invalid('posterior source seal mismatch')
 if schema.digest({k:v for k,v in m.items() if k!='manifest_sha256'})!=m['manifest_sha256']:raise schema.Invalid('manifest seal mismatch')
 for name,sha in m['sources'].items():schema.relative(name);schema.sha(sha)
 return m


def selected_cases(m):
 return [c for c in m['corpus']['cases'] if m['protocol']['schema_version']==2 or c['id'] in m['protocol']['cases']]

def method_info(method):
 engine='alea' if method in set(NATIVE+NATIVE_V3) else ('stan' if method.startswith('stan_') else 'blackjax' if method=='blackjax_nuts' else 'nuts-rs')
 guarantee='approximate' if method=='mclmc' else 'exactness_pending' if method.startswith('rmhmc') else 'reference' if method.startswith(('nuts_cov','nuts_grad','stan_','blackjax','nuts_rs')) else 'research_exact_target_intent'
 component='historical eight-step HMC covariance warmup + unchanged NUTS retention; controlled ablation only' if method=='nuts_legacy_warmup' else 'full ATLAS: NoUT preliminary/delayed/failure + deterministic HVP step densities' if method=='atlas_full' else 's-AIA curvature epochs + native NUTS warmup + frozen two-stage NUTS' if method=='nuts_saia_joint' else 'initializer + covariance warmup + NUTS' if 'pathfinder' in method else 'ATLAS curvature probe + joint two-stage warmup + NUTS' if method=='nuts_atlas_joint' else 'affine transport + covariance warmup + NUTS' if method=='nuts_transport' else 'sampler + declared warmup'
 return {'engine':engine,'guarantee':guarantee,'component':component}


def jobs(m):
 result=[];p=m['protocol'];methods=p['methods']
 for rep in range(p['repetitions']):
  for i,case in enumerate(selected_cases(m)):
   order=methods[(rep+i)%len(methods):]+methods[:(rep+i)%len(methods)]
   for method in order:
    info=method_info(method);model=m['preparation']['models'][case['id']];meta=model.get('metadata');reason=None
    if not meta:reason='model compilation or finite initialization unavailable; see preparation logs'
    elif meta['dimension']>128:reason='generic worker admission limits unconstrained dimension to 128'
    elif method.startswith('rmhmc') and meta['dimension']>p['rmhmc']['maximum_dimension']:reason='declared dense Hessian/third-derivative dimension budget'
    elif method in ('nuts_atlas_joint','nuts_saia_joint','atlas_full') and not m['preparation'].get('native',{}).get('rust_backend')=='enzyme' and meta.get('derivative_validation',{}).get('hvp_error'):reason='HVP capability preflight: '+meta['derivative_validation']['hvp_error']
    elif method in ('mams','mclmc') and meta['dimension']<2:reason='implemented isokinetic kernel requires at least two dimensions'
    elif method.startswith('stan_') and model['builds']['stan']['returncode']!=0:reason='native CmdStan model compilation failed'
    native=m['preparation'].get('native')
    if native and native['execution']=='parallel' and method=='chees':reason='shared ChEES adaptation requires a cohort; independent process-per-chain mode is not equivalent'
    if native and not reason and (method in set(NATIVE+NATIVE_V3) or method in ('nuts_rs','blackjax_nuts')):
     validation=native['conformance'].get(case['id'],{})
     if validation.get('status')!='validated':reason='native conformance unavailable/failed; no backend fallback: '+validation.get('reason','missing validation')
     elif method in ('nuts_atlas_joint','nuts_saia_joint','atlas_full') and native['rust_backend']=='enzyme':reason=None
    if p['schema_version']==3 and method.startswith('rmhmc') and not reason:
     helper=model.get('third_ad',{})
     if helper.get('returncode')!=0 and not (native and native['rust_backend']=='enzyme'):reason='per-model third-AD build unavailable; see archived preparation logs'
    limited=method.startswith('rmhmc');job={'id':f'{case["id"]}--{method}--r{rep}','case':case['id'],'method':method,'repetition':rep,'family':case['family'],'classification':info,'unsupported_reason':reason,'warmup':p['rmhmc']['warmup'] if limited else p['warmup'],'draws':p['rmhmc']['draws'] if limited else p['draws'],'timeout_seconds':p['rmhmc']['timeout_seconds'] if limited else p['timeout_seconds'],'depth':p['depth'],'seeds':[s+10000*rep for s in p['seeds']]}
    if p['schema_version']==3:job['workflow_version']=3
    if 'record_warmup_trace' in p:job['record_warmup_trace']=p['record_warmup_trace']
    if native:job.update(backend_lane=native['rust_backend'],execution=native['execution'])
    result.append(job)
 return result


def archive_preparation(m,directory):
 cfg=m['preparation'];paths=[]
 for key in ('metadata_directory','model_directory','build_directory','third_ad_directory','native_directory','native_conformance_directory'):
  if key not in cfg:continue
  base=Path(cfg[key]);paths.extend((x,f'{key}/{x.relative_to(base)}') for x in base.rglob('*') if x.is_file() and x.suffix in (('.log','.json','.cpp','.hpp','.stan','.rs') if m['schema_version']==3 else ('.log','.json')))
 with (directory/'preparation.tar.gz').open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as zipped,tarfile.open(fileobj=zipped,mode='w|') as tar:
  for path,name in paths:
   data=path.read_bytes();info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;tar.addfile(info,io.BytesIO(data))


def p_version(m):return m['protocol']['schema_version']

def select_initials(metadata,chains):
 """Use the same declared prefix for every engine, without changing preparation."""
 points=metadata.get('initials',[])
 if len(points)<chains:raise schema.Invalid('not enough sealed initial points for requested chain count')
 return [list(q) for q in points[:chains]]


def run(manifest_path,directory):
 m=validate_manifest(schema.load(manifest_path));verify_sources(m);directory=directory.resolve();directory.mkdir(parents=True,exist_ok=False);write_new(directory/'manifest.json',m);write_new(directory/'host.json',host());archive_sources(m,directory,ROOT);archive_preparation(m,directory)
 planned=jobs(m);write_new(directory/'plan.json',{'schema_version':m['schema_version'],'manifest_sha256':m['manifest_sha256'],'jobs':planned,'host_sha256':file_hash(directory/'host.json'),'sources_sha256':file_hash(directory/'sources.tar.gz'),'preparation_sha256':file_hash(directory/'preparation.tar.gz')});(directory/'jobs').mkdir();index={};cpu=min(os.sched_getaffinity(0));cfg=m['preparation'];bycase={c['id']:c for c in m['corpus']['cases']}
 for n,job in enumerate(planned):
  path=directory/'jobs'/job['id'];path.mkdir();record={'schema_version':m['schema_version'],'manifest_sha256':m['manifest_sha256'],'job':job,'status':'unsupported' if job['unsupported_reason'] else 'failed','failure':None,'process':None,'worker':None,'metrics':None,'artifacts':{}}
  if not job['unsupported_reason']:
   case=bycase[job['case']];model=cfg['models'][case['id']];metadata=model['metadata'];task=dict(job,**{k:v for k,v in metadata.items() if k not in job},library=model['builds']['bridge']['binary'],data=str(ROOT/case['directory']/'data.json'),directory=str(path),stan_binary=model['builds']['stan']['binary'])
   task['initials']=select_initials(metadata,len(job['seeds']))
   if 'initial_constrained' in task:task['initial_constrained']=task['initial_constrained'][:len(job['seeds'])]
   native=cfg.get('native')
   if native:
    task.update(target_backend=native['rust_backend'],case=case['id'],require_native_jax=True)
   if p_version(m)==3 and job['method'].startswith('rmhmc') and not (native and native['rust_backend']=='enzyme'):task['third_ad_library']=model['third_ad']['binary']
   write_new(path/'job.json',task)
   try:
    verify_sources(m)
    for binary,expected in [(cfg['worker'],cfg['worker_sha256']),(task['library'],model['builds']['bridge']['binary_sha256'])]:
     if file_hash(Path(binary))!=expected:raise schema.Invalid('compiled binary drift')
    if 'third_ad_library' in task and file_hash(Path(task['third_ad_library']))!=model['third_ad']['binary_sha256']:raise schema.Invalid('third-AD executable drift')
    if job['method'].startswith('stan_') and file_hash(Path(task['stan_binary']))!=model['builds']['stan']['binary_sha256']:raise schema.Invalid('Stan executable drift')
    if job['method']=='blackjax_nuts' and file_hash(Path(cfg['jax_python']))!=cfg['jax_python_sha256']:raise schema.Invalid('JAX interpreter drift')
    if job['method'] in set(NATIVE+NATIVE_V3) or job['method']=='nuts_rs':command=[cfg['worker'],'sample',str(path/'job.json')]
    else:command=[cfg['jax_python'] if job['method']=='blackjax_nuts' else sys.executable,'-m','alea_bench.pdb_external',str(path/'job.json')]
    affinity=[cpu]
    if native and native['execution']=='parallel':
     affinity=sorted(os.sched_getaffinity(0))[:len(job['seeds'])]
     if len(affinity)!=len(job['seeds']):raise RuntimeError('parallel protocol requires one available CPU per chain')
     task['parallel_command']=command[:-1];task['parallel_cpus']=affinity;write_path=path/'parallel-job.json';write_new(write_path,task)
     command=[sys.executable,'-m','alea_bench.native_parallel',str(write_path)]
    wrapped=[shutil.which('time'),'-f','%M','-o',str(path/'rss.txt'),shutil.which('taskset'),'-c',','.join(map(str,affinity)),*command]
    process_env=environment(ROOT)
    if native:
     process_env.update(native['runtime_environment'])
     if file_hash(Path(native['tbb_library']))!=native['tbb_sha256']:raise schema.Invalid('native TBB runtime drift')
    with (path/'process.log').open('x') as log:record['process']=isolated(wrapped,cwd=ROOT,env=process_env,log=log,timeout=job['timeout_seconds'])
    record['process'].update(command=command,cpu_affinity=affinity,peak_rss_kib=None)
    rss=(path/'rss.txt').read_text().splitlines()
    if rss and rss[-1].isdigit():record['process']['peak_rss_kib']=int(rss[-1])
    if (path/'worker.json').exists():record['worker']=schema.load(path/'worker.json')
    elif (path/'progress-worker.json').exists():record['worker']=dict(schema.load(path/'progress-worker.json'),status='partial')
    if record['process']['timed_out']:raise RuntimeError('declared job deadline exhausted; process group killed and direct worker reaped')
    if record['process']['returncode']!=0:
     with (path/'process.log').open('rb') as worker_log:admission_rejected=b'job shape/resource admission failed' in worker_log.read(16384)
     if admission_rejected:raise schema.Invalid('worker rejected job shape/resource admission; runner failure, not a sampler outcome')
     raise RuntimeError(f'worker exit {record["process"]["returncode"]}: {record["worker"].get("failure") if record["worker"] else "see process.log"}')
    if sum(x.stat().st_size for x in path.iterdir() if x.is_file())>m['protocol']['maximum_output_bytes']:raise RuntimeError('declared output byte budget exceeded')
    subprocess.run([cfg['worker'],'diagnose',str(path/'job.json')],check=True,timeout=60)
    record['metrics']=schema.load(path/'metrics.json');record['status']='completed';verify_sources(m)
   except Exception as error:record['failure']={'kind':type(error).__name__,'message':str(error)}
  if cfg.get('native',{}).get('execution')=='parallel' and not (path/'chain-artifacts.tar.gz').exists():
   with tarfile.open(path/'chain-artifacts.tar.gz','x:gz') as archive:
    for child in sorted(path.glob('chain-*')):
     if child.is_dir():archive.add(child,arcname=child.name)
  for raw in list(path.glob('*.csv')):compress_raw(raw)
  record['artifacts']={x.name:{'sha256':file_hash(x),'bytes':x.stat().st_size} for x in sorted(path.iterdir()) if x.is_file()};write_new(path/'result.json',record);index[job['id']]=file_hash(path/'result.json')
  with (directory/'progress.jsonl').open('a') as log:log.write(json.dumps({'job':job['id'],'status':record['status'],'failure':record['failure']})+'\n')
  print(f'{n+1}/{len(planned)} {job["id"]}: {record["status"]}',flush=True)
 write_new(directory/'index.json',{'schema_version':m['schema_version'],'manifest_sha256':m['manifest_sha256'],'records':index});return m


def load(directory):
 directory=Path(directory);m=validate_manifest(schema.load(directory/'manifest.json'));plan=schema.load(directory/'plan.json');index=schema.load(directory/'index.json');expected=jobs(m)
 if plan['jobs']!=expected or index['manifest_sha256']!=m['manifest_sha256'] or plan['manifest_sha256']!=m['manifest_sha256'] or set(index['records'])!={j['id'] for j in expected}:raise schema.Invalid('incomplete/mixed experiment; failures and unsupported jobs cannot be dropped')
 for name,sha in [('host.json',plan['host_sha256']),('sources.tar.gz',plan['sources_sha256']),('preparation.tar.gz',plan['preparation_sha256'])]:
  if file_hash(directory/name)!=sha:raise schema.Invalid('archive drift')
 seen=set()
 with tarfile.open(directory/'sources.tar.gz','r:gz') as source:
  import hashlib
  for member in source:
   if not member.isfile() or member.name not in m['sources'] or member.name in seen or member.size>64*1024*1024:raise schema.Invalid('invalid source archive')
   if hashlib.sha256(source.extractfile(member).read()).hexdigest()!=m['sources'][member.name]:raise schema.Invalid('archived source digest drift')
   seen.add(member.name)
 if seen!=set(m['sources']):raise schema.Invalid('incomplete sources')
 rows=[]
 for job in expected:
  folder=directory/'jobs'/job['id'];path=folder/'result.json'
  if file_hash(path)!=index['records'][job['id']]:raise schema.Invalid('result digest drift')
  row=schema.load(path)
  if row['schema_version']!=m['schema_version'] or row['job']!=job or row['manifest_sha256']!=m['manifest_sha256']:raise schema.Invalid('result identity drift')
  if row['status'] not in ('completed','failed','unsupported') or (row['status']=='failed')!=(row['failure'] is not None) or (row['status']=='unsupported')!=(job['unsupported_reason'] is not None):raise schema.Invalid('invalid completion/failure status')
  for name,info in row['artifacts'].items():
   schema.relative(name);p=folder/name
   if p.stat().st_size!=info['bytes'] or file_hash(p)!=info['sha256']:raise schema.Invalid('raw artifact drift')
  if row['status']=='completed':
   if row['process']['returncode']!=0 or row['process']['timed_out'] or row['metrics']['unconstrained']['draws']!=len(job['seeds'])*job['draws'] or row['metrics']['constrained']['draws']!=len(job['seeds'])*job['draws']:raise schema.Invalid('lost retained draws or false success')
   if 'metrics.json' not in row['artifacts'] or schema.load(folder/'metrics.json')!=row['metrics']:raise schema.Invalid('diagnostic artifact mismatch')
   meta=m['preparation']['models'][job['case']]['metadata']
   for name,skip,names in [('draws.csv.gz',8,[f'q{k}' for k in range(meta['dimension'])]),('constrained.csv.gz',2,meta['parameter_names'].split(','))]:
    if name not in row['artifacts']:raise schema.Invalid('missing retained samples')
    verify_draws(folder/name,job,skip,names)
  rows.append(row)
 return m,rows


def verify_draws(path,job,skip,names):
 """Offline completeness and finite-coordinate validation, without any engine."""
 counts=[0]*len(job['seeds'])
 with gzip.open(path,'rt',newline='') as stream:
  rows=csv.reader(stream);header=next(rows,None)
  if header is None or len(header)!=skip+len(names) or header[skip:]!=names:raise schema.Invalid('retained coordinate header drift')
  for row in rows:
   if len(row)!=len(header):raise schema.Invalid('ragged retained samples')
   c,i=int(row[0]),int(row[1])
   if c<0 or c>=len(counts) or i!=counts[c] or i>=job['draws']:raise schema.Invalid('lost/duplicate retained draw')
   if any(not math.isfinite(float(v)) for v in row[skip:]):raise schema.Invalid('nonfinite retained coordinate')
   counts[c]+=1
 if counts!=[job['draws']]*len(counts):raise schema.Invalid('incomplete retained chain set')


def main(argv=None):
 parser=argparse.ArgumentParser(prog='alea-bench pdb');commands=parser.add_subparsers(dest='command',required=True)
 p=commands.add_parser('prepare');p.add_argument('output',type=Path);p.add_argument('--models',type=Path);p.add_argument('--workflow-version',type=int,choices=(2,3),default=2)
 p=commands.add_parser('experiment');p.add_argument('protocol',type=Path);p.add_argument('output',type=Path)
 p=commands.add_parser('pack');p.add_argument('directory',type=Path);p.add_argument('output',type=Path)
 p=commands.add_parser('unpack');p.add_argument('bundle',type=Path);p.add_argument('output',type=Path)
 p=commands.add_parser('metadata');p.add_argument('models',type=Path);p.add_argument('output',type=Path)
 p=commands.add_parser('seal');p.add_argument('protocol',type=Path);p.add_argument('prepared',type=Path);p.add_argument('output',type=Path)
 p=commands.add_parser('run');p.add_argument('manifest',type=Path);p.add_argument('directory',type=Path)
 p=commands.add_parser('report');p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True)
 p=commands.add_parser('verify');p.add_argument('directory',type=Path)
 args=parser.parse_args(argv)
 if args.command in ('pack','unpack'):
  from . import pdb_bundle
  if args.command=='pack':pdb_bundle.pack(args.directory,args.output)
  else:pdb_bundle.unpack(args.bundle,args.output)
 elif args.command=='prepare':
  from .pdb_prepare import prepare
  prepare(args.output,args.models,args.workflow_version)
 elif args.command=='experiment':
  from .pdb_prepare import prepare
  args.output.mkdir(parents=True,exist_ok=False);prepared=prepare(args.output/'preparation',workflow_version=validate_protocol(schema.load(args.protocol))['schema_version']);manifest=args.output/'manifest.json';seal(args.protocol,prepared,manifest);run(manifest,args.output/'run')
  from .pdb_report import write_report
  write_report(args.output/'run')
 elif args.command=='metadata':prepare_metadata(args.models,args.output)
 elif args.command=='seal':seal(args.protocol,args.prepared,args.output)
 elif args.command=='run':
  run(args.manifest,args.directory)
  from .pdb_report import write_report
  write_report(args.directory)
 elif args.command=='report':
  from .pdb_report import build_report
  write_new(args.output,build_report(args.directory))
 else:m,rows=load(args.directory);print(f'{len(rows)} complete planned records verified; descriptive experiment')
if __name__=='__main__':
 import sys
 main()
