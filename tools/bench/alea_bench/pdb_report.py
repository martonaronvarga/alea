"""Descriptive posterior accuracy, diagnostics and costs; no verdict or score gate."""
import csv,html,json,math,re,statistics
from collections import Counter,defaultdict
from pathlib import Path
from . import schema
from .manifest import write_new
from .pdb import load,selected_cases


def reference_name(name):
 parts=name.split('.')
 return parts[0]+'['+','.join(parts[1:])+']' if len(parts)>1 else name


def coordinate_summary(metrics):
 rows=[x['summary'] for x in metrics['coordinates']];valid=all(x['error'] is None for x in rows)
 return {'maximum_rhat':max((x['rhat'] for x in rows if x['rhat'] is not None),default=None),'minimum_bulk_ess':min((x['bulk_ess'] for x in rows),default=None) if valid else None,'minimum_tail_ess':min((x['tail_ess'] for x in rows),default=None) if valid else None,'undefined_diagnostics':sum(x['error'] is not None for x in rows)}


def reference_comparison(metrics,reference):
 comparisons=[]
 for coordinate in metrics['coordinates']:
  name=reference_name(coordinate['name']);ref=reference.get(name);s=coordinate['summary']
  if ref is None or s['mean'] is None or ref['sd']<=0:continue
  comparisons.append({'parameter':name,'mean':s['mean'],'reference_mean':ref['mean'],'reference_sd':ref['sd'],'mean_error_reference_sd':(s['mean']-ref['mean'])/ref['sd'],'posterior_sd_ratio':s.get('sd')/ref['sd'] if s.get('sd') is not None else None,'mcse':s['mcse']})
 return comparisons


def work_metrics(worker, bulk, completed):
 """Per-capability API counts, never a fabricated exchange rate between derivatives."""
 keys=('gradient_api_calls','hessian_api_calls','hvp_api_calls','third_ad_api_calls')
 total=worker.get('work') or {};phases=worker.get('phases',[]);out={}
 for key in keys:
  count=total.get(key)
  if type(count) is not int:
   measured=[p[key] for p in phases if type(p.get(key)) is int]
   count=sum(measured) if measured else None
  retained=0;retained_available=False
  for phase in phases:
   before=phase.get('work_before_retained') or {};after=phase.get('work_after') or {}
   if type(before.get(key)) is int and type(after.get(key)) is int:
    retained+=after[key]-before[key];retained_available=True
  label=key.removesuffix('_api_calls')
  out['total_'+key]=count
  out['retained_'+key]=retained if retained_available else None
  valid=completed and bulk is not None
  out['minimum_bulk_ess_per_'+label+'_call']=bulk/count if valid and count else None
  out['minimum_bulk_ess_per_retained_'+label+'_call']=bulk/retained if valid and retained_available and retained else None
  out[label+'_call_ratio_unavailable_reason']=None if valid and count else ('job did not complete' if not completed else 'undefined bulk ESS' if bulk is None else 'API count unavailable' if count is None else 'no calls of this capability')
 return out


def summarize(m,records,references):
 rows=[];coordinates=[]
 for record in records:
  j=record['job'];process=record['process'];worker=record['worker'] or {};metrics=record['metrics'];phases=worker.get('phases',[])
  aggregate=lambda name:sum(p.get(name,0.) for p in phases) if phases and any(name in p for p in phases) else None
  diagnostic=coordinate_summary(metrics['unconstrained']) if metrics else None;constrained=coordinate_summary(metrics['constrained']) if metrics else None
  comparison=reference_comparison(metrics['constrained'],references[j['case']]) if metrics else []
  for c in comparison:coordinates.append(dict(job=j['id'],case=j['case'],method=j['method'],repetition=j['repetition'],**c))
  spread=[abs(c['posterior_sd_ratio']-1) for c in comparison if c['posterior_sd_ratio'] is not None]
  retained=aggregate('retained_seconds');bulk=diagnostic['minimum_bulk_ess'] if diagnostic else None
  row={'job':j['id'],'case':j['case'],'family':j['family'],'method':j['method'],**j['classification'],'repetition':j['repetition'],'status':record['status'],'failure':record['failure'],'unsupported_reason':j['unsupported_reason'],'warmup_budget':j['warmup'],'draws_per_chain':j['draws'],'chains':len(j['seeds']),'dimension':m['preparation']['models'][j['case']].get('metadata',{}).get('dimension'),'process_seconds':process['wall_seconds'] if process else None,'process_nanoseconds':process['wall_nanoseconds'] if process else None,'peak_rss_kib':process['peak_rss_kib'] if process else None,'warmup_seconds':aggregate('warmup_seconds'),'initialization_seconds':aggregate('initialization_seconds'),'retained_seconds':retained,'serialization_seconds':aggregate('serialization_seconds'),'cold_seconds':worker.get('cold_seconds'),'jit_seconds':worker.get('jit_seconds'),'backend':worker.get('backend','BridgeStan Stan reverse AD; AD Hessians for derivative methods'),'work':worker.get('work'),'diagnostics':diagnostic,'constrained_diagnostics':constrained,'divergences':metrics['unconstrained']['divergences'] if metrics else None,'depth_saturations':metrics['unconstrained']['depth_saturations'] if metrics else None,'integration_steps':metrics['unconstrained']['integration_steps'] if metrics else None,'reference_parameters_compared':len(comparison),'maximum_abs_mean_error_reference_sd':max((abs(c['mean_error_reference_sd']) for c in comparison),default=None),'rms_mean_error_reference_sd':math.sqrt(statistics.mean(c['mean_error_reference_sd']**2 for c in comparison)) if comparison else None,'maximum_abs_sd_ratio_error':max(spread,default=None),'minimum_bulk_ess_per_retained_second':bulk/retained if bulk is not None and retained else None,'minimum_bulk_ess_per_process_second':bulk/process['wall_seconds'] if bulk is not None and process and process['wall_seconds'] else None}
  if m['protocol'].get('report_work_metrics',False):row.update(work_metrics(worker,bulk,record['status']=='completed'))
  if m['preparation'].get('native'):
   from .native_report import declared_backend
   row['planned_backend']=declared_backend(m['preparation']['native'],j['classification']['engine'])
   row['observed_backend']=worker.get('backend')
   row['backend']=row['observed_backend'] or row['planned_backend']+' (not reported by worker)'
  rows.append(row)
 groups=[]
 for case in selected_cases(m):
  for method in m['protocol']['methods']:
   selected=[r for r in rows if r['case']==case['id'] and r['method']==method];complete=[r for r in selected if r['status']=='completed'];median=lambda key:statistics.median([r[key] for r in complete if r[key] is not None]) if any(r[key] is not None for r in complete) else None
   defined=lambda key:[r['diagnostics'][key] for r in complete if r['diagnostics'][key] is not None]
   groups.append({'case':case['id'],'family':case['family'],'method':method,**{k:selected[0][k] for k in ('engine','guarantee','component')},'dimension':selected[0]['dimension'],'repetitions':len(selected),'completed':len(complete),'failed':sum(r['status']=='failed' for r in selected),'unsupported':sum(r['status']=='unsupported' for r in selected),'failures':[r['failure'] for r in selected if r['failure']],'unsupported_reasons':sorted({r['unsupported_reason'] for r in selected if r['unsupported_reason']}),'median_process_seconds':median('process_seconds'),'median_warmup_seconds':median('warmup_seconds'),'median_retained_seconds':median('retained_seconds'),'median_initialization_seconds':median('initialization_seconds'),'median_serialization_seconds':median('serialization_seconds'),'median_jit_seconds':median('jit_seconds'),'maximum_peak_rss_kib':max((r['peak_rss_kib'] for r in complete if r['peak_rss_kib'] is not None),default=None),'worst_rhat':max(defined('maximum_rhat'),default=None),'smallest_bulk_ess':min(defined('minimum_bulk_ess'),default=None),'smallest_tail_ess':min(defined('minimum_tail_ess'),default=None),'undefined_diagnostics':sum(r['diagnostics']['undefined_diagnostics'] for r in complete),'divergences':sum(r['divergences'] for r in complete),'depth_saturations':sum(r['depth_saturations'] for r in complete),'median_maximum_abs_mean_error_reference_sd':median('maximum_abs_mean_error_reference_sd'),'median_maximum_abs_sd_ratio_error':median('maximum_abs_sd_ratio_error'),'median_minimum_bulk_ess_per_retained_second':median('minimum_bulk_ess_per_retained_second'),'median_minimum_bulk_ess_per_process_second':median('minimum_bulk_ess_per_process_second'),'backends':sorted({r['backend'] for r in complete})})
 if m['protocol'].get('report_work_metrics',False):
  for group in groups:
   selected=[r for r in rows if r['case']==group['case'] and r['method']==group['method'] and r['status']=='completed']
   for label in ('gradient','hessian','hvp','third_ad'):
    for phase in ('','retained_'):
     key='minimum_bulk_ess_per_'+phase+label+'_call';values=[r[key] for r in selected if r[key] is not None]
     group['median_'+key]=statistics.median(values) if values else None
 return {'schema_version':m['schema_version'],'purpose':'descriptive','manifest_sha256':m['manifest_sha256'],'summary':{'planned_jobs':len(rows),'status_counts':dict(Counter(r['status'] for r in rows)),'posterior_cases':len(selected_cases(m)),'methods':len(m['protocol']['methods']),'physical_hosts':1},'notes':m['protocol']['notes'],'groups':groups,'rows':rows},coordinates


def report_details(directory):
 m,records=load(directory);refs={}
 # Reference summaries are read from their sealed source archive, never from a
 # potentially advanced worktree. No engine is needed to regenerate this report.
 import tarfile
 with tarfile.open(Path(directory)/'sources.tar.gz','r:gz') as source:
  for case in selected_cases(m):
   name=case['directory']+'/reference-summary.json'
   refs[case['id']]=json.load(source.extractfile(name)) if name in source.getnames() else {}
 report,coordinates=summarize(m,records,refs);report['host']=schema.load(Path(directory)/'host.json');return report,coordinates

def build_report(directory):return report_details(directory)[0]


def markdown(report):
 s=report['summary'];lines=['# Descriptive PosteriorDB benchmark','',f'{s["posterior_cases"]} posteriors, {s["methods"]} method configurations, {s["planned_jobs"]} planned jobs. Status counts: `{s["status_counts"]}`.','', 'Every failure and unsupported combination is retained. Diagnostics and accuracy are observations; this report makes no promotion, pass/fail or default-selection decision.','', 'Cold wall time includes process startup, adaptation, sampling and serialization. Retained timings have engine-specific scopes; JAX host callbacks and native-JAX translations are labelled separately. RMHMC is exactness-pending; MCLMC is approximate; ATLAS/Pathfinder rows are labelled composed workflows.','', '| Posterior | Method | Completed/planned | Process s | Warmup s | Retained s | Worst R-hat | Min bulk ESS | Max mean error / ref SD |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
 fmt=lambda v:'—' if v is None else f'{v:.4g}'
 for g in report['groups']:lines.append(f'| {g["case"]} | {g["method"]} | {g["completed"]}/{g["repetitions"]} | '+ ' | '.join(fmt(g[k]) for k in ('median_process_seconds','median_warmup_seconds','median_retained_seconds','worst_rhat','smallest_bulk_ess','median_maximum_abs_mean_error_reference_sd'))+' |')
 lines+=['','## Accounting and scope','']+['- '+note for note in report['notes']]
 return '\n'.join(lines)+'\n'


def interactive(report):
 data=json.dumps(report['groups'],allow_nan=False).replace('<','\\u003c')
 return '''<!doctype html><meta charset="utf-8"><title>Alea PosteriorDB results</title><style>body{font:15px system-ui;margin:28px;color:#18232c}select,input{font:inherit;padding:6px;margin:6px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:7px;border-bottom:1px solid #ddd}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}th{position:sticky;top:0;background:#eff3f6;cursor:pointer}.note{max-width:1000px;line-height:1.5}svg{border:1px solid #ddd;background:#fafcfd}.pending{color:#874f00}</style><h1>Descriptive PosteriorDB benchmark</h1><p class="note">Compare statistical fidelity and cost together. All failures and unsupported combinations remain visible. No promotion score or eligibility gate is computed. Research RMHMC remains exactness-pending; MCLMC is approximate. BlackJAX native JAX and BridgeStan host-callback backends are identified in row details. Click a column to sort; click a row for its full record.</p><label>Posterior <select id="case"></select></label><label>Engine <select id="engine"></select></label><label>Search <input id="search"></label><p id="count"></p><svg id="plot" width="950" height="290" role="img" aria-label="Retained time versus minimum bulk ESS, logarithmic axes"></svg><p>Plot: log retained seconds (horizontal) versus log minimum bulk ESS (vertical). Each point is a method configuration; hover for details. Diagnostics and budgets differ, so inspect the table alongside the plot.</p><table><thead><tr id="head"></tr></thead><tbody id="body"></tbody></table><pre id="detail"></pre><script>const rows='''+data+''';const fields=[['case','Posterior'],['method','Method'],['guarantee','Target intent'],['completed','Completed'],['failed','Failed'],['unsupported','Unsupported'],['median_process_seconds','Process s'],['median_warmup_seconds','Warmup s'],['median_retained_seconds','Retained s'],['worst_rhat','Worst R-hat'],['smallest_bulk_ess','Min bulk ESS'],['smallest_tail_ess','Min tail ESS'],['median_maximum_abs_mean_error_reference_sd','Mean error/ref SD'],['divergences','Divergences'],['depth_saturations','Depth limits']];if(rows.some(r=>'median_minimum_bulk_ess_per_gradient_call' in r)){fields.push(['median_minimum_bulk_ess_per_gradient_call','Bulk ESS/gradient call'],['median_minimum_bulk_ess_per_retained_gradient_call','Bulk ESS/retained gradient call'],['median_minimum_bulk_ess_per_hvp_call','Bulk ESS/HVP call'],['median_minimum_bulk_ess_per_third_ad_call','Bulk ESS/third-AD call'])}let sort='case',reverse=false;for(const key of ['case','engine']){const select=document.getElementById(key);for(const value of ['All',...new Set(rows.map(r=>r[key]))]){const option=document.createElement('option');option.value=value;option.textContent=value;select.append(option)}select.onchange=render}document.getElementById('search').oninput=render;const head=document.getElementById('head');for(const [key,label] of fields){const th=document.createElement('th');th.textContent=label;th.onclick=()=>{reverse=sort===key?!reverse:false;sort=key;render()};head.append(th)}function render(){let selected=rows.filter(r=>(document.getElementById('case').value==='All'||r.case===document.getElementById('case').value)&&(document.getElementById('engine').value==='All'||r.engine===document.getElementById('engine').value)&&JSON.stringify(r).toLowerCase().includes(document.getElementById('search').value.toLowerCase()));selected.sort((a,b)=>((a[sort]??Infinity)>(b[sort]??Infinity)?1:-1)*(reverse?-1:1));document.getElementById('count').textContent=selected.length+' method/posterior configurations';const body=document.getElementById('body');body.replaceChildren();for(const row of selected){const tr=document.createElement('tr');for(const [key] of fields){const td=document.createElement('td');const value=row[key];td.textContent=value==null?'—':typeof value==='number'?(Number.isInteger(value)?value:value.toPrecision(4)):value;tr.append(td)}tr.onclick=()=>document.getElementById('detail').textContent=JSON.stringify(row,null,2);body.append(tr)}const svg=document.getElementById('plot');svg.replaceChildren();const points=selected.filter(r=>r.median_retained_seconds>0&&r.smallest_bulk_ess>0),ns='http://www.w3.org/2000/svg';if(points.length){const xs=points.map(r=>Math.log10(r.median_retained_seconds)),ys=points.map(r=>Math.log10(r.smallest_bulk_ess)),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);points.forEach((r,i)=>{const c=document.createElementNS(ns,'circle');c.setAttribute('cx',30+890*(xs[i]-xmin)/(xmax-xmin||1));c.setAttribute('cy',260-230*(ys[i]-ymin)/(ymax-ymin||1));c.setAttribute('r',5);c.setAttribute('fill',r.guarantee==='approximate'?'#b46d25':r.guarantee==='exactness_pending'?'#934b91':'#176e95');const title=document.createElementNS(ns,'title');title.textContent=r.case+' / '+r.method+' / R-hat '+r.worst_rhat+' / retained '+r.median_retained_seconds+' s';c.append(title);c.onclick=()=>document.getElementById('detail').textContent=JSON.stringify(r,null,2);svg.append(c)})}}render();</script>'''


def write_report(directory):
 directory=Path(directory);report,coordinates=report_details(directory);write_new(directory/'report.json',report)
 with (directory/'report.md').open('x') as out:out.write(markdown(report))
 with (directory/'report.html').open('x') as out:out.write(interactive(report))
 with (directory/'summary.csv').open('x',newline='') as out:
  keys=[k for k in report['groups'][0] if not isinstance(report['groups'][0][k],(dict,list))];writer=csv.DictWriter(out,keys,extrasaction='ignore');writer.writeheader();writer.writerows(report['groups'])
 with (directory/'reference-comparisons.csv').open('x',newline='') as out:
  if coordinates:
   writer=csv.DictWriter(out,list(coordinates[0]));writer.writeheader();writer.writerows(coordinates)
 return report
