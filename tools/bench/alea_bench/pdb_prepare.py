"""Pinned PosteriorDB/BridgeStan builds, entirely outside measured jobs."""
import concurrent.futures
import json,os,shutil,subprocess,time
from pathlib import Path
from .manifest import ROOT,file_hash,write_new
from . import schema
BRIDGE_REV='ec1d4d319f434d85b07193df2d81736568c42920'
CMDSTAN=Path('/nix/store/jzmq1wj1yzxj3hiqypp5ng8z8pzwlg4c-cmdStan-2.38.0/opt/cmdstan')

def checkout(path,url,revision):
 if not path.exists():
  path.parent.mkdir(parents=True,exist_ok=True)
  subprocess.run(['git','init','-q',str(path)],check=True)
  subprocess.run(['git','-C',str(path),'fetch','-q','--depth=1',url,revision],check=True)
  subprocess.run(['git','-C',str(path),'checkout','-q','--detach','FETCH_HEAD'],check=True)
 if subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()!=revision or subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=no'],text=True).strip():raise ValueError('upstream checkout must be clean and pinned')

def prepare(output,models=None,workflow_version=2):
 """Fetch/build everything before measurements, optionally reuse verified models."""
 from .pdb import prepare_metadata
 from .manifest import source_paths
 output=output.resolve();output.mkdir(parents=True,exist_ok=False)
 bridge=ROOT/'target/m16b-source/bridgestan'
 checkout(bridge,'https://github.com/roualdes/bridgestan.git',BRIDGE_REV)
 checkout(ROOT/'target/nuts-reference-upstream','https://github.com/pymc-devs/nuts-rs.git','a762aae513bf7bb5ccb27ffc9a938a923fc837ec')
 records=[]
 def build(name,command,env=None):
  start=time.perf_counter_ns()
  with (output/(name+'.log')).open('x') as log:p=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  records.append({'name':name,'command':command,'returncode':p.returncode,'seconds':(time.perf_counter_ns()-start)/1e9,'log_sha256':file_hash(output/(name+'.log'))})
  write_new(output/(name+'.json'),records[-1])
  if p.returncode:raise RuntimeError(name+' build failed; see preserved log')
 # Resolve pinned environments instead of assuming this machine's Nix store paths.
 global CMDSTAN
 CMDSTAN=Path(subprocess.check_output(['nix','build','.#cmdStan','--no-link','--print-out-paths'],cwd=ROOT,text=True).strip().splitlines()[-1])/'opt/cmdstan'
 from . import pdb
 pdb.JAX=Path(subprocess.check_output(['nix-build','tools/blackjax-reference/default.nix','--no-out-link'],cwd=ROOT,text=True).strip().splitlines()[-1])/'bin/python3'
 if models is None:
  build('bridge-common',['nix','develop','.#stable','--command','make','-C',str(bridge),'src/bridgestan_threads_adhessian.o',f'STAN={CMDSTAN}/stan/',f'STANC={CMDSTAN}/bin/stanc','STAN_THREADS=true','BRIDGESTAN_AD_HESSIAN=true'])
  models=output/'models';build_models(models)
 corpus=schema.load(ROOT/'tools/bench/corpus/pdb-v2.json');cases={c['id']:c for c in corpus['cases']}
 for row in schema.load(models/'models.json')['records']:
  for kind,item in row['builds'].items():
   folder=Path(item['binary']).parent
   if file_hash(folder/'model.stan')!=cases[row['case']]['files']['model.stan'] or file_hash(folder/'build.log')!=item['log_sha256']:raise ValueError('model build input/log drift')
   if item['returncode']==0 and file_hash(Path(item['binary']))!=item['binary_sha256']:raise ValueError('model executable drift')
 expression='let f=builtins.getFlake '+json.dumps('git+file://'+str(ROOT))+'; p=import f.inputs.nixpkgs {system=builtins.currentSystem;}; in p.llvmPackages.libclang.lib'
 libclang=subprocess.check_output(['nix','build','--impure','--no-link','--print-out-paths','--expr',expression],text=True).strip().splitlines()[-1]
 env=dict(os.environ,LIBCLANG_PATH=libclang+'/lib')
 inputs={p.relative_to(ROOT).as_posix():file_hash(p) for p in source_paths(ROOT) if p.suffix in ('.rs','.toml','.lock','.nix','.c','.h','.cpp','.hpp')}
 build('worker-fetch',['nix','develop','.#stable','--command','cargo','fetch','--locked','--manifest-path','tools/bench/pdb-worker/Cargo.toml'],env)
 build('worker',['nix','develop','.#stable','--command','cargo','build','--release','--locked','--offline','--manifest-path','tools/bench/pdb-worker/Cargo.toml'],env)
 if any(file_hash(ROOT/name)!=sha for name,sha in inputs.items()):raise ValueError('worker inputs changed during compilation')
 result=prepare_metadata(models,output/'metadata');result['build_records']['worker']=records
 if workflow_version==3:
  result['third_ad_directory']=str(output/'third-ad')
  build_third_ad(result,output/'third-ad')
  result['schema_version']=3
  result['third_ad_source_sha256']=file_hash(ROOT/'tools/bench/third_ad.cpp')
  if any(file_hash(ROOT/name)!=sha for name,sha in inputs.items()):raise ValueError('compiled inputs changed during third-AD build')
 from .pdb_external import NATIVE_JAX_CASES
 from .runner import environment
 for case in corpus['cases']:
  if case['id'] not in NATIVE_JAX_CASES:continue
  model=result['models'][case['id']]
  if not model.get('metadata'):continue
  folder=output/('jax-validation-'+case['id']);folder.mkdir()
  task=dict(model['metadata'],method='blackjax_validate',case=case['id'],directory=str(folder),library=model['builds']['bridge']['binary'],data=str(ROOT/case['directory']/'data.json'))
  write_new(folder/'job.json',task)
  build('jax-validation-'+case['id'],[str(pdb.JAX),'-m','alea_bench.pdb_external',str(folder/'job.json')],environment(ROOT))
 result['build_records']['worker']=records
 result['compiled_source_inputs']=inputs
 result['build_directory']=str(output)
 write_new(output/'prepared.json',result)
 return output/'prepared.json'

def build_third_ad(prepared,output,maximum_dimension=10,cxx=None):
 """Build a concrete grad_hessian helper per admitted model, never during jobs."""
 output=output.resolve();output.mkdir(parents=True,exist_ok=False)
 stan=CMDSTAN/'stan';math_root=stan/'lib/stan_math'
 for case,model in prepared['models'].items():
  meta=model.get('metadata')
  if not meta or meta['dimension']>maximum_dimension:
   model['third_ad']={'returncode':None,'reason':'declared third-AD compilation budget dimension<=10; missing metadata remains visible'};continue
  folder=output/case;folder.mkdir();source=folder/'model.stan'
  shutil.copyfile(ROOT/'tools/bench/corpus/pdb-v2'/case/'model.stan',source)
  header=folder/'model.hpp';binary=folder/'third_ad.so'
  stanc=[str(CMDSTAN/'bin/stanc'),'--name=alea_ad_model',f'--o={header}',str(source)]
  compiler=([str(cxx)] if cxx else ['nix','develop','.#stable','--command','c++'])+['-O2','-std=c++17','-pthread','-D_REENTRANT','-DSTAN_THREADS','-fPIC','-fvisibility=hidden',
    '-I'+str(folder),'-I'+str(stan/'src'),'-I'+str(math_root),'-I'+str(stan/'lib/rapidjson_1.1.0'),
    *['-I'+str(math_root/'lib'/suffix) for suffix in ('eigen_3.4.0','boost_1.87.0','tbb_2020.3/include','sundials_6.1.1/include')],
    str(ROOT/'tools/bench/third_ad.cpp'),'-L'+str(math_root/'lib/tbb'),'-Wl,-rpath,'+str(math_root/'lib/tbb'),'-shared','-ltbb','-o',str(binary)]
  start=time.perf_counter_ns()
  with (folder/'build.log').open('x') as log:
   result=subprocess.run(stanc,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   if result.returncode==0:result=subprocess.run(compiler,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  row={'returncode':result.returncode,'commands':[stanc,compiler],'seconds':(time.perf_counter_ns()-start)/1e9,
    'log_sha256':file_hash(folder/'build.log'),'model_source_sha256':file_hash(source),
    'header_sha256':file_hash(header) if header.exists() else None,'helper_source_sha256':file_hash(ROOT/'tools/bench/third_ad.cpp'),
    'binary':str(binary),'binary_sha256':file_hash(binary) if result.returncode==0 else None}
  write_new(folder/'build.json',row);model['third_ad']=row
  print('third AD',case,result.returncode,flush=True)
 return prepared

def build_models(output):
 output=output.resolve();output.mkdir(parents=True,exist_ok=False)
 corpus=schema.load(ROOT/'tools/bench/corpus/pdb-v2.json');bridge=ROOT/'target/m16b-source/bridgestan'
 if subprocess.check_output(['git','-C',str(bridge),'rev-parse','HEAD'],text=True).strip()!=BRIDGE_REV:raise ValueError('BridgeStan revision drift')
 def one(case):
  directory=output/case['id'];directory.mkdir();result={'case':case['id'],'builds':{}}
  for kind in ('bridge','stan'):
   folder=directory/kind;folder.mkdir();source=folder/'model.stan';shutil.copyfile(ROOT/case['directory']/'model.stan',source)
   if kind=='bridge':
    binary=source.with_name('model_model.so');command=['nix','develop','.#stable','--command','make','-C',str(bridge),str(binary),f'STAN={CMDSTAN}/stan/',f'STANC={CMDSTAN}/bin/stanc','STAN_THREADS=true','BRIDGESTAN_AD_HESSIAN=true']
   else:binary=source.with_suffix('');command=['nix','develop','.#stable','--command','make','-C',str(CMDSTAN),str(binary),'PRECOMPILED_MODEL_HEADER=']
   start=time.perf_counter()
   with (folder/'build.log').open('x') as log:p=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   result['builds'][kind]={'command':command,'returncode':p.returncode,'seconds':time.perf_counter()-start,'log_sha256':file_hash(folder/'build.log'),'binary':str(binary),'binary_sha256':file_hash(binary) if p.returncode==0 else None}
  write_new(directory/'build.json',result);print(case['id'],{k:v['returncode'] for k,v in result['builds'].items()},flush=True);return result
 # The shared BridgeStan object was built by the pilot before these independent
 # models; two compiler slots never write the same model object.
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(one,corpus['cases']))
 write_new(output/'models.json',{'schema_version':2,'bridgestan_revision':BRIDGE_REV,'stan_version':'2.38.0','records':records})
 return records
if __name__=='__main__':
 import sys
 build_models(Path(sys.argv[1]))
