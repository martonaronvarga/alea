"""Expose recorded disjoint derivative intervals and stage partitions without inference."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
KEYS=['gradient_api_calls','hessian_api_calls','hvp_api_calls','third_ad_api_calls']
SHA=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def export(run,out):
 index=json.loads((run/'index.json').read_text());rows=[];stages=[];verified_jobs=0;verified_partitions=0;inputs={}
 for job,sha in index['records'].items():
  path=run/'jobs'/job/'result.json';assert SHA(path)==sha;inputs[job]=sha
  r=json.loads(path.read_text());w=r.get('worker') or {};w=dict(w,**w.get('partial',{}));phases=w.get('phases',[]);identity=dict(job=job,case=r['job']['case'],method=r['job']['method'],status=r['status']);sums={k:0 for k in KEYS};full=True;timed=0
  for p in phases:
   for phase in ['initialization','warmup','retained','serialization']:
    if phase+'_seconds' not in p:continue
    start=p.get('initialization_work_before') if phase=='initialization' else p.get('work_before') if phase=='warmup' else p.get('work_before_retained') if phase=='retained' else None
    end=p.get('initialization_work_after') if phase=='initialization' else p.get('work_before_retained') or p.get('work_after_warmup') if phase=='warmup' else p.get('work_after') if phase=='retained' else None
    values={k:end[k]-start[k] if start is not None and end is not None and type(start.get(k)) is int and type(end.get(k)) is int else None for k in KEYS}
    for k,v in values.items():
     if v is not None:assert v>=0
     if phase!='serialization':
      if v is None:full=False
      else:sums[k]+=v
    if phase!='serialization':timed+=1
    rows.append(dict(identity,chain=p.get('chain','cohort'),phase=phase,seconds=p[phase+'_seconds'],**values,counter_interval='recorded' if all(v is not None for v in values.values()) else 'unavailable; no calls inferred'))
   f=p.get('frozen') or {}
   if 'first_stage_integration_steps' in f:
    a,b,c=f['first_stage_integration_steps'],f['delayed_integration_steps'],f['ghost_integration_steps'];assert a+b+c==f['kernel_warmup_integration_steps']
    stages.append(dict(identity,chain=p.get('chain','cohort'),phase='kernel_warmup_only',first_stage_steps=a,delayed_steps=b,forward_steps=None,reverse_steps=None,ghost_steps=c,hvp_calls=None))
   if 'retained_first_stage_steps' in p:stages.append(dict(identity,chain=p['chain'],phase='retained',first_stage_steps=p['retained_first_stage_steps'],delayed_steps=p['retained_delayed_steps'],forward_steps=None,reverse_steps=None,ghost_steps=p['retained_ghost_steps'],hvp_calls=None))
   if 'atlas_warmup_forward_steps' in p:
    a,b,c=p['atlas_warmup_forward_steps'],p['atlas_warmup_reverse_steps'],p['atlas_warmup_ghost_steps'];assert a+b+c==p['atlas_warmup_integration_steps']
    for phase,prefix in [('kernel_warmup_only','atlas_warmup'),('retained','retained')]:stages.append(dict(identity,chain=p['chain'],phase=phase,first_stage_steps=None,delayed_steps=None,forward_steps=p[prefix+'_forward_steps'],reverse_steps=p[prefix+'_reverse_steps'],ghost_steps=p[prefix+'_ghost_steps'],hvp_calls=p.get(prefix+'_hvp_calls')))
  if r['status']=='completed' and identity['method'] in ['drghmc','atlas_full']:
   retained=[x for x in stages if x['job']==job and x['phase']=='retained'];assert len(retained)==len(r['job']['seeds'])
   assert sum(sum(x.get(k) or 0 for k in ['first_stage_steps','delayed_steps','forward_steps','reverse_steps','ghost_steps']) for x in retained)==r['metrics']['unconstrained']['integration_steps'],job
   if identity['method']=='atlas_full':assert sum(x['hvp_calls'] for x in stages if x['job']==job)==w['work']['hvp_api_calls'],job
   verified_partitions+=1
  if full and timed and r['status']=='completed' and all(type((w.get('work') or {}).get(k)) is int for k in KEYS):
   assert sums=={k:w['work'][k] for k in KEYS},(job,sums,w['work']);verified_jobs+=1
  if not any(x['job']==job for x in rows):rows.append(dict(identity,chain='unavailable',phase='unavailable',seconds=None,**{k:None for k in KEYS},counter_interval='unsupported or no completed interval; see raw worker total counts'))
 for file,values,fields in [('phase-work.csv',rows,list(rows[0])),('stage-work.csv',stages,['job','case','method','status','chain','phase','first_stage_steps','delayed_steps','forward_steps','reverse_steps','ghost_steps','hvp_calls'])]:
  with (out/file).open('x',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(values)
 shutil.copyfile(__file__,out/'export_phase_work.py')
 receipt=dict(manifest_sha256=index['manifest_sha256'],planned_jobs=len(inputs),result_sha256=inputs,disjoint_counter_sum_verified_completed_jobs=verified_jobs,phase_rows=len(rows),stage_partition_rows=len(stages),verified_retained_macrostep_partition_jobs=verified_partitions,scope='Actual recorded intervals only. Pathfinder initialization is disjoint from warmup; ChEES cohort warmup uses recorded work_after_warmup. Serialization and missing upstream counters are unavailable, never inferred from leapfrogs. DRGHMC/Atlas kernel warmup excludes native pilot.',files={n:SHA(out/n) for n in ['phase-work.csv','stage-work.csv','export_phase_work.py']})
 (out/'phase-work-verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k not in ['files','result_sha256']}))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args();export(a.run,a.output)
