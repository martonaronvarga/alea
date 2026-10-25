"""Verify the archived raw paired outputs and derive descriptive timing ratios."""
import csv,hashlib,json,statistics,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]);out=Path(sys.argv[2]);sha=lambda x:hashlib.sha256(x).hexdigest()
with tarfile.open(root/'remote-evidence.tar.gz') as tar:
 members=tar.getmembers()
 selected={}
 for member in members:
  name=member.name.removeprefix('./')
  if member.isfile() and (name in ['fair-baseline-profile','optimized-profile','baseline-worker','optimized-worker','confirmation/results.json','worker-pairs/results.json'] or (name.startswith('confirmation/') and name.endswith('.jsonl') and 'progress' not in name) or (name.startswith('worker-pairs/') and name.endswith(('/draws.csv','/constrained.csv','/worker.json'))) or (name.startswith('final-counters/') and name.endswith('.stat.csv'))):selected[name]=tar.extractfile(member).read()
 def read(name):return selected[name]
 def load(name):return json.loads(read(name))
 native=load('confirmation/results.json');worker=load('worker-pairs/results.json')
 assert len(native['records'])==70 and len(worker['records'])==60
 assert native['spec']['rounds']==7 and worker['spec']['rounds']==5
 for name,h in native['spec']['binaries'].items():assert sha(read(name))==h,name
 for name,h in worker['spec']['binaries'].items():assert sha(read(name))==h,name
 identities={};checks=0
 for row in native['records']:
  path=f"confirmation/{row['round']}-{row['workload']}-{row['binary']}.jsonl";raw=read(path);assert sha(raw)==row['raw_sha256'],path
  rows=[json.loads(x) for x in raw.decode().splitlines()];identity=[{k:v for k,v in x.items() if not k.endswith('seconds')} for x in rows];key=(row['round'],row['workload'])
  if key in identities:assert identities[key]==identity,key
  identities[key]=identity;checks+=1
 def strip_time(x):
  if isinstance(x,dict):return {k:strip_time(v) for k,v in x.items() if not k.endswith('seconds')}
  if isinstance(x,list):return [strip_time(v) for v in x]
  return x
 identities={}
 for row in worker['records']:
  prefix=f"worker-pairs/{row['round']}-{row['case']}-{row['method']}-{row['binary']}/";raw=read(prefix+'draws.csv');constrained=read(prefix+'constrained.csv');data=load(prefix+'worker.json');assert sha(raw)==row['draw_sha256'];assert sha(constrained)==row['constrained_sha256'];identity=(sha(raw),sha(constrained),sha(json.dumps(strip_time(data),sort_keys=True,allow_nan=False).encode()));key=(row['round'],row['case'],row['method'])
  if key in identities:assert identities[key]==identity,key
  identities[key]=identity;checks+=1
 def ratios(rows,groups,names,keys):
  result=[]
  for group in sorted({tuple(row[k] for k in groups) for row in rows}):
   selected=[row for row in rows if tuple(row[k] for k in groups)==group];item=dict(zip(groups,group))
   for key in keys:
    before=[x[key] for x in selected if x['binary']==names[0]];after=[x[key] for x in selected if x['binary']==names[1]];paired=[]
    for rnd in sorted({x['round'] for x in selected}):
     pair=[next(x for x in selected if x['round']==rnd and x['binary']==name) for name in names];paired.append(pair[1][key]/pair[0][key])
    item[key]=dict(before_median=statistics.median(before),after_median=statistics.median(after),ratio_of_medians=statistics.median(after)/statistics.median(before),median_paired_ratio=statistics.median(paired),minimum_paired_ratio=min(paired),maximum_paired_ratio=max(paired))
   result.append(item)
  return result
 counters={}
 for member in tar.getmembers():
  if member.name.startswith('./final-counters/') and member.name.endswith('.stat.csv'):
   events={}
   for row in csv.reader(read(member.name[2:]).decode().splitlines()):
    if len(row)>2 and row[2] in ['instructions:u','cycles:u','branches:u','branch-misses:u']:events[row[2]]=int(row[0])
   assert len(events)==4,member.name;counters[Path(member.name).name]=events
 result=dict(native=ratios(native['records'],['workload'],['fair-baseline-profile','optimized-profile'],['process_seconds','cpu_seconds']),posteriordb=ratios(worker['records'],['case','method'],['baseline-worker','optimized-worker'],['process_seconds','cpu_seconds','warmup_seconds','retained_seconds']),counters=counters,verified_native_processes=70,verified_real_model_processes=60,identities_checked=checks,all_native_fingerprints_and_real_draw_trace_work_identities_match=True)
out.write_text(json.dumps(result,indent=2)+'\n');print('Verified all130 primary paired processes and hardware counter records.')
