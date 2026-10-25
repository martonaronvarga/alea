"""Falsifiable scientific gates, promotion decisions and host-separated tracking."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import tarfile
from . import models, schema
from .manifest import file_hash, write_new
from .runner import jobs


def verify_metrics(metrics,case,policy):
    schema.keys(metrics,{'schema_version','draws','divergences','depth_saturations','integration_steps','coordinates','second_moments','ebfmi','acceptance_observations','mean_acceptance','maximum_reported_energy_error','nonfinite_energy_errors'},'metrics')
    count=policy['draws']*policy['chains']
    if metrics['schema_version']!=1 or metrics['draws']!=count or len(metrics['coordinates'])!=case['dimension'] or len(metrics['second_moments'])!=case['dimension'] or len(metrics['ebfmi'])!=policy['chains']:raise schema.Invalid('diagnostic schema/count drift')
    for name in ('divergences','depth_saturations'):schema.integer(metrics[name],0,count,name)
    schema.integer(metrics['integration_steps'],0,2**63-1,'trajectory work')
    for row in metrics['coordinates']+metrics['second_moments']:
        schema.keys(row,{'mean','rhat','bulk_ess','tail_ess','mcse','error'},'coordinate diagnostic')
        if row['error'] is None:
            for name in ('mean','rhat','bulk_ess','tail_ess','mcse'):
                if type(row[name]) not in (int,float) or not math.isfinite(row[name]):raise schema.Invalid('nonfinite/missing successful diagnostic')
            if min(row['rhat'],row['bulk_ess'],row['tail_ess'],row['mcse'])<0:raise schema.Invalid('negative diagnostic')
        elif not isinstance(row['error'],str) or not row['error'] or any(row[n] is not None for n in ('mean','rhat','bulk_ess','tail_ess','mcse')):raise schema.Invalid('undefined diagnostics were hidden')
    schema.integer(metrics['acceptance_observations'],0,count,'acceptance observations')
    schema.integer(metrics['nonfinite_energy_errors'],0,count,'nonfinite energy errors')
    if metrics['mean_acceptance'] is not None:schema.real(metrics['mean_acceptance'],0,1,'mean acceptance')
    if metrics['maximum_reported_energy_error'] is not None:schema.real(metrics['maximum_reported_energy_error'],0,1e300,'energy error')
    for e in metrics['ebfmi']:
        if e is not None:schema.real(e,0,1e100,'E-BFMI')


def quality(record,thresholds):
    if record['status']!='completed':return {'passed':False,'reasons':['worker_failure']}
    m=record['metrics'];reasons=[]
    for row in m['coordinates']:
        if row['error'] is not None:reasons.append('undefined_diagnostic');continue
        if row['rhat']>thresholds['maximum_rhat']:reasons.append('rhat')
        if min(row['bulk_ess'],row['tail_ess'])<thresholds['minimum_ess']:reasons.append('ess')
    if m['nonfinite_energy_errors']:reasons.append('nonfinite_energy_error')
    if m['divergences']/m['draws']>thresholds['maximum_divergence_rate']:reasons.append('divergences')
    if m['depth_saturations']/m['draws']>thresholds['maximum_depth_rate']:reasons.append('depth')
    reference=models.expected(record['job']['case'])
    if reference is not None:
        for i,(mean,variance) in enumerate(reference):
            for row,expected,label in [(m['coordinates'][i],mean,'analytic_mean'),(m['second_moments'][i],variance+mean*mean,'analytic_second_moment')]:
                if row['error'] is not None or abs(row['mean']-expected)>thresholds['mean_mcse_multiplier']*row['mcse']+thresholds['mean_absolute_tolerance']:reasons.append(label)
    return {'passed':not reasons,'reasons':sorted(set(reasons))}


def load_bundle(directory):
    manifest=schema.manifest(schema.load(directory/'manifest.json'));index=schema.load(directory/'index.json');plan=schema.load(directory/'plan.json')
    schema.keys(index,{'schema_version','manifest_sha256','records','source_archive_sha256','preparation_archive_sha256'},'index')
    schema.keys(plan,{'schema_version','manifest_sha256','jobs','source_archive_sha256','preparation_archive_sha256'},'plan')
    if index['schema_version']!=1 or plan['schema_version']!=1 or index['manifest_sha256']!=manifest['manifest_sha256'] or plan['manifest_sha256']!=manifest['manifest_sha256']:raise schema.Invalid('mixed experiment identities')
    expected={j['id']:j for j in jobs(manifest)}
    if set(index['records'])!=set(expected) or plan['jobs']!=list(expected):raise schema.Invalid('missing/duplicate/unplanned job; failures cannot be dropped')
    archive=directory/'sources.tar.gz'
    if file_hash(archive)!=index['source_archive_sha256'] or plan['source_archive_sha256']!=index['source_archive_sha256']:raise schema.Invalid('source archive digest mismatch')
    if file_hash(directory/'preparation.tar.gz')!=index['preparation_archive_sha256'] or plan['preparation_archive_sha256']!=index['preparation_archive_sha256']:raise schema.Invalid('preparation record/log digest mismatch')
    seen=set();total=0
    with tarfile.open(archive,'r:gz') as stream:
        for item in stream:
            if not item.isfile() or item.name not in manifest['sources'] or item.name in seen or item.size>16_777_216:raise schema.Invalid('invalid source archive member')
            total+=item.size
            if total>268_435_456:raise schema.Invalid('source archive size limit')
            seen.add(item.name)
            if hashlib.sha256(stream.extractfile(item).read()).hexdigest()!=manifest['sources'][item.name]:raise schema.Invalid('archived source changed')
    if seen!=set(manifest['sources']):raise schema.Invalid('incomplete source archive')
    records=[]
    for name,expected_job in expected.items():
        job_dir=directory/'jobs'/name;path=job_dir/'result.json'
        if not path.exists() or file_hash(path)!=index['records'][name]:raise schema.Invalid(f'changed/missing result {name}')
        r=schema.result(schema.load(path))
        if r['manifest_sha256']!=manifest['manifest_sha256'] or r['job']!=expected_job:raise schema.Invalid('job/protocol/data/truth drift')
        for artifact,info in r['artifacts'].items():
            f=job_dir/artifact
            if not f.is_file() or f.stat().st_size!=info['bytes'] or file_hash(f)!=info['sha256']:raise schema.Invalid(f'changed/missing raw artifact {name}/{artifact}')
        if r['status']=='completed':
            verify_metrics(r['metrics'],r['job']['case'],manifest['protocol']['policy'])
            if r['phases']['retained_steps']!=r['metrics']['integration_steps']:raise schema.Invalid('actual retained work counter disagreement')
            if r['process']['wall_seconds']<=0 or r['process']['peak_rss_kib'] is None:raise schema.Invalid('lost process timing/RSS')
            if not r['preflight'] or not r['preflight']['passed']:raise schema.Invalid('successful job has no passing independent gradient check')
        records.append(r)
    return manifest,records


def raw_draws(directory,record):
    with gzip.open(directory/'jobs'/record['job']['id']/'draws.csv.gz','rt',newline='') as f:
        for row in csv.DictReader(f):yield [float(row[f'q{i}']) for i in range(record['job']['case']['dimension'])]


def calibration(manifest,records,directory):
    policy=manifest['protocol']['calibration'];groups=[]
    for name in policy['cases']:
        selected=[r for r in records if r['job']['dataset'] is not None and r['job']['case']['id']==name]
        dimension=selected[0]['job']['case']['dimension'];bins=[[0]*policy['bins'] for _ in range(dimension)];coverage=[0]*dimension;predictive_bins=[0]*policy['bins'];predictive_coverage=0;details=[];failures=0
        diagnostic_failures=[r['job']['id'] for r in selected if not quality(r,manifest['protocol']['thresholds'])['passed']]
        for r in selected:
            if r['status']!='completed':failures+=1;details.append({'job':r['job']['id'],'failure':r['failure']});continue
            values=list(raw_draws(directory,r));n=len(values);truth=r['job']['truth'];ranks=[];covered=[]
            for i in range(dimension):
                column=sorted(q[i] for q in values);rank=sum(v<truth[i] for v in column)
                # Continuous ties are randomized, including repeated retained states.
                ties=sum(v==truth[i] for v in column);rng=random.Random(policy['seed']+r['job']['dataset']*128+i)
                rank+=rng.randrange(ties+1);bins[i][min(policy['bins']-1,rank*policy['bins']//(n+1))]+=1
                cover=column[int(.1*(n-1))]<=truth[i]<=column[int(.9*(n-1))];coverage[i]+=int(cover);ranks.append(rank);covered.append(cover)
            seed=policy['seed']+r['job']['dataset']*7919; rng=random.Random(seed);kind=r['job']['case']['kind']
            if kind==7:observed=truth[0]+rng.gauss(0,.5)
            elif kind==5:observed=truth[0]+math.exp(truth[1])*truth[2]+rng.gauss(0,.7)
            else:observed=float(rng.random()<1/(1+math.exp(-1.2*(truth[0]+math.exp(truth[1])*truth[2]))))
            subset=values[::max(1,len(values)//256)];replicated=models.predictive(r['job']['case'],subset,seed+1)
            if kind==9:
                probability=sum(1/(1+math.exp(-1.2*(q[0]+math.exp(q[1])*q[2]))) for q in subset)/len(subset)
                pit=rng.random()*(1-probability) if observed==0 else 1-probability+rng.random()*probability
            else:pit=(sum(v<observed for v in replicated)+rng.random()*(sum(v==observed for v in replicated)+1))/(len(replicated)+1)
            predictive_bins[min(policy['bins']-1,int(pit*policy['bins']))]+=1;predictive_coverage+=int(.1<=pit<=.9)
            details.append({'job':r['job']['id'],'truth':truth,'ranks':ranks,'covered':covered,'posterior_predictive_pit':pit,'heldout_observation':observed})
        expected=policy['datasets']/policy['bins'];chi=lambda b:sum((v-expected)**2/expected for v in b)
        histograms=[{'coordinate':i,'bins':b,'chi_square':chi(b),'coverage':coverage[i]/policy['datasets'],'passed':sum(b)==policy['datasets'] and chi(b)<=policy['maximum_rank_chi_square'] and policy['minimum_coverage']<=coverage[i]/policy['datasets']<=policy['maximum_coverage']} for i,b in enumerate(bins)]
        ppc={'bins':predictive_bins,'chi_square':chi(predictive_bins),'coverage':predictive_coverage/policy['datasets']}
        ppc['passed']=sum(predictive_bins)==policy['datasets'] and ppc['chi_square']<=policy['maximum_rank_chi_square'] and policy['minimum_coverage']<=ppc['coverage']<=policy['maximum_coverage']
        groups.append({'case':name,'datasets':policy['datasets'],'failed_jobs':failures,'diagnostic_failures':diagnostic_failures,'ranks':histograms,'posterior_predictive':ppc,'details':details,'passed':failures==0 and not diagnostic_failures and all(h['passed'] for h in histograms) and ppc['passed']})
    return groups


def interval(values,confidence,resamples,seed=16000):
    if not values:return None
    rng=random.Random(seed);samples=sorted(statistics.median(rng.choices(values,k=len(values))) for _ in range(resamples));tail=(1-confidence)/2
    return {'median':statistics.median(values),'lower':samples[int(tail*(len(samples)-1))],'upper':samples[int((1-tail)*(len(samples)-1))],'replicates':len(values),'scope':'paired process-wall ESS efficiency; bootstrap timing repetitions, fixed numerical seeds'}


def build_report(directory):
    manifest,records=load_bundle(directory);p=manifest['protocol'];thresholds=p['thresholds'];quality_rows={r['job']['id']:quality(r,thresholds) for r in records}
    rows=[]
    for r in records:
        metric=r['metrics'];ess=min((min(c['bulk_ess'],c['tail_ess']) for c in metric['coordinates'] if c['error'] is None),default=None) if metric else None
        rows.append({'job':r['job']['id'],'case':r['job']['case']['id'],'partition':r['job']['case']['partition'],'engine':r['engine'],'dataset':r['job']['dataset'],'repetition':r['job']['repetition'],'host':r['host']['fingerprint'],'status':r['status'],'failure':r['failure'],'quality':quality_rows[r['job']['id']],'process_seconds':r['process']['wall_seconds'],'peak_rss_kib':r['process']['peak_rss_kib'],'ess_per_process_second':ess/r['process']['wall_seconds'] if ess is not None and r['process']['wall_seconds'] else None})
    comparisons=[];by_key={(r['case'],r['engine'],r['repetition'],r['host']):r for r in rows if r['dataset'] is None}
    for case in manifest['corpus']['cases']:
        for reference in p['engines']:
            if reference=='alea':continue
            paired=[];complete=True;agreement=[]
            for rep in range(p['policy']['repetitions']):
                a=next((r for r in rows if r['case']==case['id'] and r['engine']=='alea' and r['dataset'] is None and r['repetition']==rep),None)
                b=by_key.get((case['id'],reference,rep,a['host'])) if a else None
                if a is None or b is None or not a['quality']['passed'] or not b['quality']['passed'] or not a['ess_per_process_second'] or not b['ess_per_process_second']:complete=False
                else:paired.append(a['ess_per_process_second']/b['ess_per_process_second'])
                ar=next((r for r in records if r['job']['id']==f'{case["id"]}-alea-r{rep}'),None);br=next((r for r in records if r['job']['id']==f'{case["id"]}-{reference}-r{rep}'),None)
                if not ar or not br or ar['status']!='completed' or br['status']!='completed':continue
                for i,(x,y) in enumerate(zip(ar['metrics']['coordinates'],br['metrics']['coordinates'],strict=True)):
                    valid=x['error'] is None and y['error'] is None;error=abs(x['mean']-y['mean']) if valid else None;limit=thresholds['mean_mcse_multiplier']*math.hypot(x['mcse'],y['mcse'])+thresholds['mean_absolute_tolerance'] if valid else None
                    agreement.append({'repetition':rep,'coordinate':i,'error':error,'limit':limit,'passed':valid and error<=limit})
            comparisons.append({'case':case['id'],'partition':case['partition'],'reference':reference,'matched_geometry':reference!='nuts-rs','complete_quality_pairs':complete,'efficiency_ratio':interval(paired,thresholds['confidence'],thresholds['bootstrap_resamples']) if complete else None,'agreement':agreement,'agreement_passed':len(agreement)==p['policy']['repetitions']*case['dimension'] and all(v['passed'] for v in agreement)})
    sbc=calibration(manifest,records,directory)
    heldout=[r for r in rows if r['partition']=='heldout' and r['dataset'] is None];correct=all(r['quality']['passed'] for r in heldout) and all(c['agreement_passed'] for c in comparisons if c['partition']=='heldout') and all(c['passed'] for c in sbc)
    qualified=[c for c in comparisons if c['partition']=='heldout' and c['matched_geometry']]
    speed=bool(qualified) and all(c['efficiency_ratio'] and c['efficiency_ratio']['lower']>=thresholds['minimum_speedup'] for c in qualified)
    failure_rates={engine:sum(not r['quality']['passed'] for r in heldout if r['engine']==engine)/max(1,sum(r['engine']==engine for r in heldout)) for engine in p['engines']}
    failure_ok=all(failure_rates['alea']<=value+thresholds['maximum_failure_rate_increase'] for engine,value in failure_rates.items() if engine!='alea')
    laws=manifest['ledgers']['exactness']['content']['transitions'];open_laws=[{'path':v['path'],'law':name,'claim':law['claim']} for v in laws for name,law in v['laws'].items() if law['status']=='open']
    return {'schema_version':1,'manifest_sha256':manifest['manifest_sha256'],'protocol_sha256':manifest['protocol_sha256'],'corpus_sha256':manifest['corpus_sha256'],'thresholds':thresholds,'engine_differences':p['engine_differences'],'rows':rows,'comparisons':comparisons,'calibration':sbc,'failure_rates':failure_rates,'promotion':{'decision':'eligible_for_review' if correct and speed and failure_ok and p['purpose']=='validation' else 'not_eligible','correctness':correct,'performance':speed,'failure_rate':failure_ok,'ledger_changed':False,'open_map_laws':open_laws,'maturity_scope':'S2 experiment utility only; no automatic sampler or release promotion'},'excluded_claims':p['excluded_claims']}


def markdown(report):
    lines=['# Manifest-driven validation report','',f'Manifest: `{report["manifest_sha256"]}`','',f'Promotion decision: **{report["promotion"]["decision"]}**. Numerical defaults and maturity ledgers were not changed.','', '| Case | Engine | Completed jobs | Quality passes | Median process s | Peak RSS KiB |','| --- | --- | ---: | ---: | ---: | ---: |']
    groups=sorted({(r['case'],r['engine']) for r in report['rows'] if r['dataset'] is None})
    for case,engine in groups:
        rows=[r for r in report['rows'] if r['case']==case and r['engine']==engine and r['dataset'] is None];times=[r['process_seconds'] for r in rows if r['process_seconds'] is not None];rss=[r['peak_rss_kib'] for r in rows if r['peak_rss_kib'] is not None]
        lines.append(f'| {case} | {engine} | {sum(r["status"]=="completed" for r in rows)}/{len(rows)} | {sum(r["quality"]["passed"] for r in rows)}/{len(rows)} | {statistics.median(times):.5f}' + f' | {max(rss) if rss else "unavailable"} |' if times else f'| {case} | {engine} | 0/{len(rows)} | 0/{len(rows)} | unavailable | unavailable |')
    lines += ['', 'All failures, exclusions, phase/counter availability, uncertainty and calibration details are retained in `report.json`. Failed quality gates never enter a favorable speed ratio.', '', '## Excluded claims','']+[f'- {s}' for s in report['excluded_claims']]
    return '\n'.join(lines)+'\n'


def write_report(directory):
    report=build_report(directory);write_new(directory/'report.json',report)
    with (directory/'report.md').open('x') as out:out.write(markdown(report))
    return report


def tracking_summary(observations,confidence=.95,resamples=2000,maximum_time_regression=1.2):
    groups={}
    for observation in observations:
        key=tuple(observation[n] for n in ('host','corpus_sha256','policy_sha256','case','engine','build_sha256'))
        groups.setdefault(key,[]).append(observation)
    strata=[];families={}
    for key,values in groups.items():
        valid=[v['process_seconds'] for v in values if v['quality_passed'] and v['status']=='completed']
        qualified=len(valid)==len(values) and len(valid)>=3
        band=interval(valid,confidence,resamples) if qualified else None
        if band:band['scope']='process-wall seconds; bootstrap timing repetitions, fixed numerical seeds'
        row={'host':key[0],'corpus_sha256':key[1],'policy_sha256':key[2],'case':key[3],'engine':key[4],'build_sha256':key[5],'planned_observations':len(values),'quality_failures':len(values)-len(valid),'time_band':band}
        strata.append(row);families.setdefault(key[:5],[]).append((row,valid if qualified else []))
    regressions=[]
    for family,builds in families.items():
        baseline,base_values=builds[0]
        for candidate,values in builds[1:]:
            band=None
            if base_values and values:
                rng=random.Random(16016);ratios=[]
                for _ in range(resamples):ratios.append(statistics.median(rng.choices(values,k=len(values)))/statistics.median(rng.choices(base_values,k=len(base_values))))
                ratios.sort();tail=(1-confidence)/2
                band={'median':statistics.median(values)/statistics.median(base_values),'lower':ratios[int(tail*(resamples-1))],'upper':ratios[int((1-tail)*(resamples-1))]}
            regressions.append({'host':family[0],'case':family[3],'engine':family[4],'baseline_build':baseline['build_sha256'],'candidate_build':candidate['build_sha256'],'time_ratio':band,'threshold':maximum_time_regression,'informational_alert':bool(band and band['lower']>maximum_time_regression)})
    return {'schema_version':1,'hosts':sorted({k[0] for k in groups}),'strata':strata,'regressions':regressions,'correctness_ci_is_timing_independent':True,'scope':'Only matching physical-host/model/policy strata compare builds; every quality failure disables timing eligibility. Baseline is first input bundle for that stratum.'}


def track(directories,output):
    observations=[];limits=None
    for directory in directories:
        report=build_report(directory);manifest,records=load_bundle(directory);current=manifest['protocol']['thresholds']
        selected=(current['confidence'],current['bootstrap_resamples'],current['maximum_time_regression'])
        if limits is not None and limits!=selected:raise schema.Invalid('tracking thresholds differ; declare a shared policy')
        limits=selected
        for row in report['rows']:
            if row['dataset'] is not None:continue
            record=next(r for r in records if r['job']['id']==row['job'])
            build=schema.digest({'binary_sha256':record['process']['binary_sha256'],'sources':manifest['sources'],'toolchains':manifest['toolchains']})
            observations.append({'host':row['host'],'corpus_sha256':manifest['corpus_sha256'],'policy_sha256':schema.digest(manifest['protocol']['policy']),'case':row['case'],'engine':row['engine'],'build_sha256':build,'process_seconds':row['process_seconds'],'quality_passed':row['quality']['passed'],'status':row['status']})
    result=tracking_summary(observations,*limits);write_new(output,result);return result
