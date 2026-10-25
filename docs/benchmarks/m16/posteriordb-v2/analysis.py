import csv,json,statistics,sys,collections
from pathlib import Path
base=Path(sys.argv[1]);out=Path(sys.argv[2]);report=json.loads((base/'report.json').read_text());groups=report['groups'];rows=report['rows'];fmt=lambda x:'—' if x is None else f'{x:.4g}'
lines=['# Descriptive PosteriorDB findings','',f"{report['summary']['planned_jobs']} planned jobs: {report['summary']['status_counts']}. Twenty posteriors, 26 configurations, three repetitions, four chains per job. One {report['host']['cpu']} host; one pinned CPU per job.",'','Completion is a technical outcome, not an adequacy gate. Reference discrepancies, R-hat, ESS, divergences and failures remain separate observations. All medians below condition on completion, with denominators shown. There is no aggregate promotion score.','', '## Configuration coverage and observations','', '| Configuration | Completed / 60 | Failed | Unsupported | Worst R-hat | Smallest bulk ESS | Largest median max mean error / ref SD |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for method in report['groups'][:26]:
 name=method['method'];g=[x for x in groups if x['method']==name];defined=lambda key:[x[key] for x in g if x[key] is not None]
 lines.append(f"| {name} | {sum(x['completed'] for x in g)}/60 | {sum(x['failed'] for x in g)} | {sum(x['unsupported'] for x in g)} | {fmt(max(defined('worst_rhat'),default=None))} | {fmt(min(defined('smallest_bulk_ess'),default=None))} | {fmt(max(defined('median_maximum_abs_mean_error_reference_sd'),default=None))} |")
lines+=['','Worst diagnostics and discrepancies span very different posteriors and budgets; they are not a ranking. RMHMC has only 64 retained draws per chain and remains exactness-pending. MCLMC is approximate. ATLAS and Pathfinder identify composed workflows. See the full report for each posterior and repetition.','', '## Paired baseline costs','', 'Alea below means covariance-diagonal warmup plus NUTS. Ratios use per-posterior medians where both configurations complete all three repetitions. Counts state the available denominator; failed comparisons are not hidden or treated as successful. Process ratios include driver/startup costs. Retained scopes differ across engines, and adaptation policies are not identical.','', '| Comparison | Paired posteriors | Median Alea / other process time | Median Alea / other retained time | Median Alea / other bulk ESS per process second |','| --- | ---: | ---: | ---: | ---: |']
by={(g['case'],g['method']):g for g in groups};native={'Rate_1_data-Rate_1_model','Rate_2_data-Rate_2_model','eight_schools-eight_schools_centered','eight_schools-eight_schools_noncentered','wells_data-wells_dist100_model','kidiq-kidscore_momhs'};comparisons=[]
for other,label,subset in [('stan_nuts_diag','CmdStan diagonal NUTS',None),('stan_nuts_dense','CmdStan dense NUTS',None),('nuts_rs','nuts-rs diagonal NUTS',None),('blackjax_nuts','BlackJAX native JAX targets',native),('blackjax_nuts','BlackJAX BridgeStan callbacks',set(g['case'] for g in groups)-native)]:
 pairs=[]
 for (case,method),g in by.items():
  if method!='nuts_cov_diag' or subset is not None and case not in subset:continue
  if (case,other) not in by:continue
  b=by[(case,other)]
  if g['completed']==g['repetitions'] and b['completed']==b['repetitions']:pairs.append((g,b))
 ratio=lambda key:statistics.median(a[key]/b[key] for a,b in pairs if a[key] is not None and b[key] is not None and b[key]>0) if any(a[key] is not None and b[key] is not None and b[key]>0 for a,b in pairs) else None
 values=[ratio(k) for k in ('median_process_seconds','median_retained_seconds','median_minimum_bulk_ess_per_process_second')];comparisons.append({'comparison':label,'posteriors':len(pairs),'ratios':values});lines.append(f'| {label} | {len(pairs)} | '+ ' | '.join(map(fmt,values))+' |')
lines+=['','A process/retained time ratio below one means less recorded time for the Alea workflow. An ESS/s ratio above one means greater recorded bulk ESS per second on those completed cases; inspect the corresponding R-hat and reference discrepancies before interpreting it. There is no quality eligibility filter. Callback timings include Python/host transfers and do not establish native JAX performance.','', '## Failures and unsupported combinations','', 'RMHMC rows reporting inconsistent SoftAbs third derivatives failed the finite-difference backend contract before sampling; they do not constitute RMHMC inference timings. ATLAS interval rejections and fixed transport-learning domain errors describe the declared warmup workflow. Native engine logs distinguish optimization failures from adapter/runtime errors.','']
counts=collections.Counter(r['method'] for r in rows if r['failure'])
for method,count in counts.most_common():
 examples=list(dict.fromkeys(r['failure']['message'] for r in rows if r['method']==method and r['failure']))
 example='; '.join(m.replace('|','/').replace('\\n',' ')[:170] for m in examples[:3]);lines.append(f'- {method}: {count} failed jobs. Examples: `{example}`. Full per-job reasons are in failures.csv.')
for reason,count in collections.Counter(r['unsupported_reason'] for r in rows if r['unsupported_reason']).items():lines.append(f'- Unsupported, {count} job(s): {reason}.')
lines+=['','Full failure commands and partial outputs are in each job artifact. Timeouts preserve the declared budgets. Undefined diagnostics stay in the report; completion alone is not convergence.','', '## Review and reproduction','', 'Open [the interactive report](report.html), inspect [the per-configuration CSV](summary.csv) or [coordinate reference comparisons](reference-comparisons.csv), and read [the method and timing scopes](../../../m16-posteriordb.md). The complete raw bundle includes every planned outcome and its source/preparation archives.']
(out/'findings.md').write_text('\n'.join(lines)+'\n');(out/'paired-baseline-costs.json').write_text(json.dumps(comparisons,indent=2)+'\n')
with (out/'failures.csv').open('w',newline='') as f:
 writer=csv.writer(f);writer.writerow(['job','case','method','status','reason'])
 for r in rows:
  if r['status']!='completed':writer.writerow([r['job'],r['case'],r['method'],r['status'],r['unsupported_reason'] or r['failure']['message']])
print(json.dumps({'summary':report['summary'],'paired':comparisons},indent=2))

"""Reproduce phase costs from sealed results; no statistical eligibility decisions."""
import csv,json,sys
from pathlib import Path

def analyze(run,output):
 run=Path(run);output=Path(output);rows=[]
 def add(job,status,chain,phase,seconds,work=None,extra=None):
  row={k:job[k] for k in ('id','case','method','repetition')};row.update(status=status,chain=chain,phase=phase,seconds=seconds,gradient_api_calls=None,hessian_api_calls=None,hvp_api_calls=None,retained_factorizations=None,retained_triangular_solves=None,scope='')
  if work:row.update(work)
  if extra:row.update(extra)
  rows.append(row)
 def difference(end,start):
  if end is None or start is None:return None
  return {k:end[k]-start[k] for k in ('gradient_api_calls','hessian_api_calls','hvp_api_calls') if isinstance(end.get(k),int) and isinstance(start.get(k),int)}
 plan=json.loads((run/'plan.json').read_text())
 for job in plan['jobs']:
  record=json.loads((run/'jobs'/job['id']/'result.json').read_text());status=record['status'];worker=record['worker'] or {};worker=dict(worker,**worker.get('partial',{}));phases=worker.get('phases',[])
  if record['process']:add(job,status,'worker','process',record['process']['wall_seconds'],extra={'scope':'fresh-process wall time including driver and output; Linux pidfd completion'})
  for key in ('cold_seconds','jit_seconds'):
   if key in worker:add(job,status,'worker',key[:-8],worker[key],extra={'scope':worker.get('cold_scope','worker phase') if key=='cold_seconds' else 'synchronized JAX compilation'})
  for p in phases:
   chain=p.get('chain','cohort');before=p.get('work_before');middle=p.get('work_before_retained');after=p.get('work_after')
   for key in ('warmup_seconds','retained_seconds','serialization_seconds','initialization_seconds'):
    if key not in p:continue
    phase=key[:-8];work=difference(middle,before) if phase=='warmup' else difference(after,middle) if phase=='retained' else None
    scope=''
    if phase=='warmup' and job['method']=='nuts_pathfinder':scope='includes Pathfinder initializer; separate nested initialization observation also emitted'
    if phase=='retained':scope='native Stan CSV/generated quantities included' if p.get('native_retained_includes_csv_and_generated_quantities') else 'kernel conversion and owned positions included; serialization excluded'
    extra={'scope':scope}
    if phase=='retained':extra.update({k:p.get(k) for k in ('retained_factorizations','retained_triangular_solves')})
    add(job,status,chain,phase,p[key],work,extra)
   initialization=p.get('frozen',{}).get('pathfinder_seconds')
   if initialization is not None:add(job,status,chain,'initialization',initialization,extra={'scope':'Alea raw-importance Pathfinder; already included in warmup_seconds, do not add twice'})
   if isinstance(p.get('gradient_api_calls'),int):add(job,status,chain,'combined_target_work',None,{'gradient_api_calls':p['gradient_api_calls']},{'scope':'combined warmup and retained actual host callback count; no phase split exposed'})
 with (output/'phase-costs.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)
 return rows
if __name__=='__main__':analyze(sys.argv[1],sys.argv[2])

# Observed repetition spread, not a confidence interval or eligibility gate.
with (out/'timing-dispersion.csv').open('w',newline='') as stream:
 fields=['case','method','status','planned_repetitions','jobs_with_status','phase','observations','minimum_seconds','median_seconds','maximum_seconds','median_absolute_deviation_seconds'];writer=csv.DictWriter(stream,fields);writer.writeheader()
 for g in groups:
  selected=[r for r in rows if r['case']==g['case'] and r['method']==g['method']]
  for status in ('completed','failed','unsupported'):
   subset=[r for r in selected if r['status']==status]
   if not subset:continue
   for phase in ('process','cold','jit','initialization','warmup','retained','serialization'):
    values=[r[phase+'_seconds'] for r in subset if r[phase+'_seconds'] is not None];median=statistics.median(values) if values else None
    writer.writerow(dict(case=g['case'],method=g['method'],status=status,planned_repetitions=g['repetitions'],jobs_with_status=len(subset),phase=phase,observations=len(values),minimum_seconds=min(values,default=None),median_seconds=median,maximum_seconds=max(values,default=None),median_absolute_deviation_seconds=statistics.median(abs(v-median) for v in values) if values else None))
