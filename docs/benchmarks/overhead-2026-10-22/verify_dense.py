"""Verify the separate balanced dense-workflow counter experiment."""
import csv,hashlib,json,statistics,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]);output=Path(sys.argv[2]);sha=lambda x:hashlib.sha256(x).hexdigest()
def strip(x):
 if isinstance(x,dict):return {k:strip(v) for k,v in x.items() if not k.endswith('seconds')}
 if isinstance(x,list):return list(map(strip,x))
 return x
with tarfile.open(root/'dense-confirmation.tar.gz') as tar:files={m.name:tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
d=json.loads(files['dense-confirmation/results.json']);rows=d['records'];assert len(rows)==12;identities={};counters=[]
for row in rows:
 prefix=f"dense-confirmation/{row['round']}-{row['case']}-{row['method']}-{row['binary']}/";worker=json.loads(files[prefix+'worker.json']);raw=sha(files[prefix+'draws.csv']);constrained=sha(files[prefix+'constrained.csv']);assert raw==row['draw_sha256'] and constrained==row['constrained_sha256'];identity=(raw,constrained,sha(json.dumps(strip(worker),sort_keys=True,allow_nan=False).encode()))
 if row['round'] in identities:assert identities[row['round']]==identity
 identities[row['round']]=identity
 events={r[2]:int(r[0]) for r in csv.reader(files[prefix+'counters.csv'].decode().splitlines()) if len(r)>2 and r[2] in ['instructions:u','cycles:u']};assert len(events)==2;counters.append(dict(round=row['round'],binary=row['binary'],**events))
ratios={}
for key in ['process_seconds','cpu_seconds','warmup_seconds','retained_seconds']:
 values=[]
 for rnd in range(6):
  pair=[next(x for x in rows if x['round']==rnd and x['binary']==name) for name in ['baseline-worker','optimized-worker']];values.append(pair[1][key]/pair[0][key])
 ratios[key]=dict(median_paired_ratio=statistics.median(values),minimum_paired_ratio=min(values),maximum_paired_ratio=max(values))
medians={name:statistics.median(r['instructions:u'] for r in counters if r['binary']==name) for name in ['baseline-worker','optimized-worker']}
result=dict(processes=12,affinity=d['spec']['affinity'],instrumented=True,all_draw_trace_work_identities_match=True,ratios=ratios,instruction_medians=medians,instruction_ratio=medians['optimized-worker']/medians['baseline-worker'],counters=counters)
output.write_text(json.dumps(result,indent=2)+'\n');print('Verified all12 dense confirmation outcomes, traces, draws and counters.')
