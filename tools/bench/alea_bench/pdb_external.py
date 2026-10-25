"""External posterior workers. Native JAX translations and shared-backend scope are explicit."""
import csv,ctypes as C,hashlib,json,math,re,subprocess,sys,time
from pathlib import Path
NATIVE_JAX_CASES={'Rate_1_data-Rate_1_model','Rate_2_data-Rate_2_model','eight_schools-eight_schools_centered','eight_schools-eight_schools_noncentered','wells_data-wells_dist100_model','kidiq-kidscore_momhs'}

class Bridge:
 def __init__(self,library,data):
  self.lib=C.CDLL(str(library));self.calls=0;P=C.c_void_p;D=C.POINTER(C.c_double);E=C.POINTER(C.c_char_p)
  signatures={'bs_model_construct':([C.c_char_p,C.c_uint,E],P),'bs_model_destruct':([P],None),'bs_free_error_msg':([C.c_char_p],None),'bs_param_unc_num':([P],C.c_int),'bs_param_num':([P,C.c_bool,C.c_bool],C.c_int),'bs_param_names':([P,C.c_bool,C.c_bool],C.c_char_p),'bs_log_density_gradient':([P,C.c_bool,C.c_bool,D,D,D,E],C.c_int),'bs_log_density':([P,C.c_bool,C.c_bool,D,D,E],C.c_int),'bs_param_constrain':([P,C.c_bool,C.c_bool,D,D,P,E],C.c_int),'bs_param_unconstrain':([P,D,D,E],C.c_int)}
  for name,(args,result) in signatures.items():f=getattr(self.lib,name);f.argtypes=args;f.restype=result
  error=C.c_char_p();self.model=self.lib.bs_model_construct(str(data).encode(),17001,C.byref(error))
  if not self.model:self.error(error)
  self.d=self.lib.bs_param_unc_num(self.model)
 def error(self,error):
  message=error.value.decode() if error.value else 'BridgeStan evaluation failed'
  if error.value:self.lib.bs_free_error_msg(error)
  raise ValueError(message)
 def gradient(self,q):
  self.calls+=1;q=(C.c_double*self.d)(*q);g=(C.c_double*self.d)();v=C.c_double();error=C.c_char_p()
  if self.lib.bs_log_density_gradient(self.model,True,True,q,C.byref(v),g,C.byref(error)):
   if error.value:self.lib.bs_free_error_msg(error)
   return -math.inf,[math.nan]*self.d
  return v.value,list(g)
 def value(self,q):
  v=C.c_double();error=C.c_char_p();q=(C.c_double*self.d)(*q)
  if self.lib.bs_log_density(self.model,True,True,q,C.byref(v),C.byref(error)):self.error(error)
  return v.value
 def constrain(self,q,tp):
  n=self.lib.bs_param_num(self.model,tp,False);out=(C.c_double*n)();q=(C.c_double*self.d)(*q);error=C.c_char_p()
  if self.lib.bs_param_constrain(self.model,tp,False,q,out,None,C.byref(error)):self.error(error)
  return list(out)
 def unconstrain(self,q):
  q=(C.c_double*len(q))(*q);out=(C.c_double*self.d)();error=C.c_char_p()
  if self.lib.bs_param_unconstrain(self.model,q,out,C.byref(error)):self.error(error)
  return list(out)
 def close(self):
  if self.model:self.lib.bs_model_destruct(self.model);self.model=None


def native_density(case,data,xp):
 """Small independent native-JAX subset; remaining models use labelled host callbacks."""
 name=case
 if name in ('Rate_1_data-Rate_1_model','Rate_2_data-Rate_2_model'):
  pairs=[(data['k'],data['n'])] if name.startswith('Rate_1') else [(data['k1'],data['n1']),(data['k2'],data['n2'])]
  return lambda q:sum((k+1)*(-xp.logaddexp(0.,-q[i]))+(n-k+1)*(-xp.logaddexp(0.,q[i])) for i,(k,n) in enumerate(pairs))
 if name=='wells_data-wells_dist100_model':
  x=xp.asarray(data['dist'])/100.;y=xp.asarray(data['switched'])
  return lambda q:xp.sum(y*(q[0]+q[1]*x)-xp.logaddexp(0.,q[0]+q[1]*x))
 if name in ('eight_schools-eight_schools_centered','eight_schools-eight_schools_noncentered'):
  n=data['J'];y=xp.asarray(data['y']);sigma=xp.asarray(data['sigma'])
  def density(q):
   mu=q[n];lt=q[n+1];tau=xp.exp(lt)
   lp=-.5*(mu/5.)**2+lt-xp.logaddexp(0.,2*(lt-math.log(5.)))
   if name.endswith('noncentered'):theta=q[:n]*tau+mu;lp-=.5*xp.sum(q[:n]**2)
   else:theta=q[:n];lp-=n*lt+.5*xp.sum(((theta-mu)/tau)**2)
   return lp-.5*xp.sum(((y-theta)/sigma)**2)
  return density
 if name=='kidiq-kidscore_momhs':
  x=xp.asarray(data['mom_hs']);y=xp.asarray(data['kid_score']);n=data['N']
  return lambda q:q[2]-xp.logaddexp(0.,2*(q[2]-math.log(2.5)))-n*q[2]-.5*xp.sum(((y-q[0]-q[1]*x)*xp.exp(-q[2]))**2)
 return None


def json_init(names,values):
 result={}
 for name,value in zip(names.split(','),values,strict=True):
  pieces=name.split('.');key=pieces[0]
  if len(pieces)==1:result[key]=value;continue
  indexes=[int(x)-1 for x in pieces[1:]];node=result.setdefault(key,[])
  for level,index in enumerate(indexes):
   while len(node)<=index:node.append([] if level<len(indexes)-1 else None)
   if level==len(indexes)-1:node[index]=value
   else:node=node[index]
 return result


def open_outputs(job):
 d=Path(job['directory']);raw=(d/'draws.csv').open('x');constrained=(d/'constrained.csv').open('x');r=csv.writer(raw);p=csv.writer(constrained)
 r.writerow(['chain','draw','divergent','saturated','steps','acceptance','energy','energy_error']+[f'q{i}' for i in range(job['dimension'])]);p.writerow(['chain','draw']+job['parameter_names'].split(','));return raw,constrained,r,p


def stan(job,bridge):
 directory=Path(job['directory']);phases=[];raw,constrained,rows,parameters=open_outputs(job)
 try:
  for chain,seed in enumerate(job['seeds']):
   init=directory/f'init-{chain}.json';init.write_text(json.dumps(json_init(job['initial_parameter_names'],bridge.constrain(job['initials'][chain],False))))
   initialization=0.
   if job['method']=='stan_pathfinder_nuts':
    pf=directory/f'pathfinder-{chain}.csv';start=time.perf_counter_ns();initialization_seed=int.from_bytes(hashlib.sha256(f'pathfinder:{seed}'.encode()).digest()[:4],'big')%(2**31-1) or 1
    if initialization_seed==seed:raise ValueError('phase seed collision')
    command=[job['stan_binary'],'pathfinder','num_paths=4','num_psis_draws=1','num_draws=16','num_elbo_draws=8','max_lbfgs_iters=40','data',f'file={job["data"]}','random',f'seed={initialization_seed}',f'init={init}','output',f'file={pf}','sig_figs=17','refresh=0'];subprocess.run(command,check=True)
    initialization=(time.perf_counter_ns()-start)/1e9
    with pf.open() as f:points=list(csv.DictReader(line for line in f if not line.startswith('#')))
    if not points:raise ValueError('Pathfinder returned no initial position')
    names=job['initial_parameter_names'];values=[float(points[0][n]) for n in names.split(',')];init=directory/f'pathfinder-init-{chain}.json';init.write_text(json.dumps(json_init(names,values)))
   output=directory/f'stan-{chain}.csv';metric='dense_e' if job['method']=='stan_nuts_dense' else 'diag_e'
   command=[job['stan_binary'],'sample',f'num_warmup={job["warmup"]}',f'num_samples={job["draws"]}','save_warmup=1','adapt','engaged=1','delta=0.8','algorithm=hmc','engine=nuts',f'max_depth={job["depth"]}',f'metric={metric}','stepsize_jitter=0','data',f'file={job["data"]}','random',f'seed={seed}',f'init={init}','output',f'file={output}','sig_figs=17','refresh=0'];subprocess.run(command,check=True)
   lines=output.read_text().splitlines();samples=list(csv.DictReader(line for line in lines if not line.startswith('#')))
   if len(samples)!=job['warmup']+job['draws']:raise ValueError('Stan lost draws')
   text='\n'.join(lines);warm=float(re.search(r'([\d.eE+-]+) seconds \(Warm-up\)',text)[1]);retained=float(re.search(r'([\d.eE+-]+) seconds \(Sampling\)',text)[1]);serialize=time.perf_counter_ns()
   for i,row in enumerate(samples[job['warmup']:]):
    original=[float(row[n]) for n in job['initial_parameter_names'].split(',')];q=bridge.unconstrain(original)
    rows.writerow([chain,i,row['divergent__'],int(int(row['treedepth__'])>=job['depth']),row['n_leapfrog__'],row['accept_stat__'],row['energy__'],'',*q]);parameters.writerow([chain,i,*[row[n] for n in job['parameter_names'].split(',')]])
   raw.flush();constrained.flush();phases.append({'chain':chain,'initialization_seconds':initialization,'warmup_seconds':warm,'retained_seconds':retained,'serialization_seconds':(time.perf_counter_ns()-serialize)/1e9,'native_retained_includes_csv_and_generated_quantities':True})
   (directory/'progress-worker.json').write_text(json.dumps({'phases':phases}))
 finally:raw.close();constrained.close()
 return {'schema_version':2,'status':'completed','phases':phases,'work':{'gradient_api_calls':None,'reason':'CmdStan does not expose total target callbacks'},'backend':'native CmdStan 2.38.0 reverse AD'}


def blackjax(job,bridge):
 start=time.perf_counter_ns()
 import importlib.metadata
 import numpy as np
 import jax
 import jax.numpy as jnp
 import blackjax
 versions={n:importlib.metadata.version(n) for n in ('jax','jaxlib','blackjax')}
 if versions!={'jax':'0.9.2','jaxlib':'0.9.2','blackjax':'1.5'}:raise ValueError('unmatched engine versions')
 jax.config.update('jax_enable_compilation_cache',False)
 if not jax.config.x64_enabled or jax.default_backend()!='cpu':raise ValueError('requires float64 CPU')
 shape=(jax.ShapeDtypeStruct((),jnp.float64),jax.ShapeDtypeStruct((bridge.d,),jnp.float64))
 def callback(q):
  value,gradient=bridge.gradient(q);return np.asarray(value,dtype=np.float64),np.asarray(gradient,dtype=np.float64)
 @jax.custom_vjp
 def shared(q):return jax.pure_callback(callback,shape,q)[0]
 def forward(q):return jax.pure_callback(callback,shape,q)
 def backward(g,weight):return (g*weight,)
 shared.defvjp(forward,backward)
 if job.get('require_native_jax'):
  from .native_jax import density as native_jax_density
  density=native_jax_density(job['case'],json.loads(Path(job['data']).read_text()))
 else:density=native_density(job['case'],json.loads(Path(job['data']).read_text()),jnp)
 backend='native JAX translation' if density is not None else 'BridgeStan host callback through custom_vjp; includes callback/host transfer overhead'
 if density is None:density=shared
 # Independently check native scores and density differences at three points.
 if job['method']=='blackjax_validate':
  if backend!='native JAX translation':raise ValueError('validation requires native translation')
  base=job['initials'][0];offset=None
  for j in range(3):
   q=np.asarray(base)+(.03*(j-1))*np.arange(1,bridge.d+1);value,g=jax.value_and_grad(density)(jnp.asarray(q));actual,score=bridge.gradient(q)
   if not np.allclose(np.asarray(g),score,rtol=1e-6,atol=1e-6):raise ValueError('native JAX gradient differs from Stan')
   difference=float(value)-actual
   if offset is None:offset=difference
   if abs(difference-offset)>1e-6*(1+abs(actual)):raise ValueError('native JAX density measure mismatch')
  return {'schema_version':2,'status':'completed','backend':backend,'versions':versions,'validation':'three independent score and density-difference checks against Stan; before measurements'}
 cold=(time.perf_counter_ns()-start)/1e9
 adapter=blackjax.window_adaptation(blackjax.nuts,density,is_mass_matrix_diagonal=True,target_acceptance_rate=.8,max_num_doublings=job['depth'])
 q=jnp.asarray(job['initials'][0]);key=jax.random.key(job['seeds'][0]);compile_start=time.perf_counter_ns()
 warm=jax.jit(lambda key,q:adapter.run(key,q,num_steps=job['warmup'])).lower(key,q).compile();jit=(time.perf_counter_ns()-compile_start)/1e9
 def sample(state,key,step,mass):
  kernel=blackjax.nuts.build_kernel();keys=jax.random.split(key,job['draws'])
  def advance(state,key):
   state,info=kernel(key,state,density,step,mass,max_num_doublings=job['depth']);row=jnp.concatenate((jnp.array([info.is_divergent,info.num_integration_steps,info.acceptance_rate,info.num_trajectory_expansions]),state.position));return state,row
  return jax.lax.scan(advance,state,keys)
 raw,constrained,rows,parameters=open_outputs(job);phases=[];sampling=None
 try:
  for chain,seed in enumerate(job['seeds']):
   q=jnp.asarray(job['initials'][chain]);key,retained_key=jax.random.split(jax.random.key(seed));before=bridge.calls;warm_start=time.perf_counter_ns();(state,params),_=jax.block_until_ready(warm(key,q));warm_seconds=(time.perf_counter_ns()-warm_start)/1e9
   if sampling is None:
    c=time.perf_counter_ns();sampling=jax.jit(sample).lower(state,key,params['step_size'],params['inverse_mass_matrix']).compile();jit+=(time.perf_counter_ns()-c)/1e9
   retained=time.perf_counter_ns();_,draws=jax.block_until_ready(sampling(state,retained_key,params['step_size'],params['inverse_mass_matrix']));seconds=(time.perf_counter_ns()-retained)/1e9;serialize=time.perf_counter_ns()
   for i,row in enumerate(np.asarray(draws)):
    q=row[4:].tolist();rows.writerow([chain,i,int(row[0]),int(row[3]>=job['depth']),int(row[1]),row[2],'','',*q]);parameters.writerow([chain,i,*bridge.constrain(q,job['include_tp'])])
   raw.flush();constrained.flush();phases.append({'chain':chain,'warmup_seconds':warm_seconds,'retained_seconds':seconds,'serialization_seconds':(time.perf_counter_ns()-serialize)/1e9,'step':float(params['step_size']),'gradient_api_calls':bridge.calls-before if backend!='native JAX translation' else None});(Path(job['directory'])/'progress-worker.json').write_text(json.dumps({'phases':phases,'backend':backend}))
 finally:raw.close();constrained.close()
 return {'schema_version':2,'status':'completed','phases':phases,'cold_seconds':cold,'jit_seconds':jit,'backend':backend,'versions':versions,'work':{'gradient_api_calls':bridge.calls if backend!='native JAX translation' else None,'scope':'Actual host callback count; native JAX gradient count unavailable'}}


def main():
 start=time.perf_counter_ns();job=json.loads(Path(sys.argv[1]).read_text());directory=Path(job['directory']);bridge=Bridge(job['library'],job['data']);target_load=(time.perf_counter_ns()-start)/1e9
 try:
  record=blackjax(job,bridge) if job['method'].startswith('blackjax_') else stan(job,bridge)
 except Exception as error:
  record={'schema_version':2,'status':'failed','failure':{'kind':type(error).__name__,'message':str(error)}}
  if (directory/'progress-worker.json').exists():record['partial']=json.loads((directory/'progress-worker.json').read_text())
  raise
 finally:
  if 'record' in locals():
   record['cold_seconds']=record.get('cold_seconds',0.)+target_load;record['cold_scope']='driver setup and target load plus JAX imports/backend initialization where applicable; interpreter and native subprocess startup remain in process wall time'
   record['worker_seconds']=(time.perf_counter_ns()-start)/1e9;(directory/'worker.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
  bridge.close()
if __name__=='__main__':main()
