"""Summarize independent-seed moments without treating finite-step error as exact.

Uncertainty uses independent seed means at equal physical time, not iid draws.
Failed/partial trials remain in trials.csv and are excluded from complete-seed
moment estimates explicitly, with both denominators reported.
"""
import argparse,csv,gzip,json,math,statistics
from pathlib import Path

def analyze(directory):
 directory=Path(directory)
 with (directory/'trials.csv').open() as f:trials=list(csv.DictReader(f))
 groups={};sums={}
 path=directory/'draws.csv';opener=open
 if not path.exists():path=directory/'draws.csv.gz';opener=gzip.open
 with opener(path,'rt') as f:
  for r in csv.DictReader(f):
   key=(r['target'],r['method'],r['step'],r['seed']);x=float(r['x']);y=float(r['y'])
   a=sums.setdefault(key,[0,0.,0.,0.,0.]);a[0]+=1
   for i,v in enumerate((x,y,x*x,y*y),1):a[i]+=v
 for trial in trials:
  key=tuple(trial[k] for k in ('target','method','step','seed'))
  assert sums.get(key,[0])[0]==int(trial['completed_draws'])
  groups.setdefault(key[:3],[]).append((trial,sums.get(key)))
 rows=[]
 for (target,method,h),cases in sorted(groups.items()):
  complete=[a for t,a in cases if not t['failure'] and int(t['completed_draws'])==int(t['planned_draws'])]
  for i,(moment,truth) in enumerate(zip(['mean_x','mean_y','second_x','second_y'],[0.,0.,1.,1. if target=='gaussian' else 1.5]),1):
   values=[a[i]/a[0] for a in complete];mean=statistics.mean(values) if values else None
   se=statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else None
   rows.append(dict(target=target,method=method,step=float(h),moment=moment,truth=truth,
      estimate=mean,bias=None if mean is None else mean-truth,independent_seed_se=se,
      planned_seeds=len(cases),complete_seeds=len(complete),failed_or_partial_seeds=len(cases)-len(complete),
      total_gradient_calls=sum(int(t['gradient_calls']) for t,a in cases)))
 return dict(uncertainty='Standard error across independent equal-physical-time seed means; no iid draw assumption or pass gate.',
   limitations='Two analytic two-dimensional targets, eight seeds, fixed refresh sqrt(2), eight macrosteps; no universal bias law or monotonicity claim.',trials=trials,moments=rows)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
 result=analyze(a.directory)
 with (a.directory/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 with (a.directory/'moments.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,result['moments'][0]);w.writeheader();w.writerows(result['moments'])
