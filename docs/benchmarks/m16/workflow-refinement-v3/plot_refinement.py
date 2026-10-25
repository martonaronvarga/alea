import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path.cwd();out=root/'docs/benchmarks/m16/workflow-refinement-v3';r=json.loads((out/'report.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(10,6.3),sharex=True,layout='constrained')
for row,target in enumerate(['gaussian','banana']):
 for col,(moment,label) in enumerate([('second_x','x²'),('second_y','y²')]):
  ax=axes[row,col]
  for method,color in [('mams','#247596'),('mclmc','#b86528')]:
   points=sorted([v for v in r['moments'] if v['target']==target and v['moment']==moment and v['method']==method],key=lambda v:v['step'])
   ax.errorbar([v['step'] for v in points],[v['bias'] for v in points],yerr=[2*v['independent_seed_se'] for v in points],marker='o',capsize=3,color=color,label=method.upper())
  ax.axhline(0,color='#555',linewidth=.7);ax.set_xscale('log',base=2);ax.set_xticks([.125,.25,.5,1.],labels=['0.125','0.25','0.5','1'])
  ax.set_title(target.capitalize()+' · '+label);ax.set_ylabel('Moment estimate − analytic value');ax.grid(alpha=.18)
  if row==1:ax.set_xlabel('Frozen integration step')
axes[0,0].legend(frameon=False)
fig.suptitle('Fixed physical time refinement · 8 independent seeds\nError bars: ±2 SE across seed means; no iid draw assumption',fontsize=12)
fig.savefig(out/'refinement.svg');fig.savefig(out/'refinement.png',dpi=170)
