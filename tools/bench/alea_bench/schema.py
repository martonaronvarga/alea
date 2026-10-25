"""Strict, versioned JSON contracts. Unknown fields/versions are errors."""
import hashlib
import json
import math
import functools
from pathlib import PurePosixPath

VERSION=1
ENGINES=('alea','blackjax','stan','nuts-rs')
POLICY={'algorithm','geometry','adaptation','warmup','draws','depth','energy_limit','precision','chains','seeds','repetitions'}
LIMITS={'threads','timeout_seconds','maximum_output_bytes','maximum_cases'}
THRESHOLDS={'maximum_rhat','minimum_ess','maximum_divergence_rate','maximum_depth_rate','mean_mcse_multiplier','mean_absolute_tolerance','gradient_absolute_tolerance','gradient_relative_tolerance','minimum_speedup','confidence','bootstrap_resamples','maximum_failure_rate_increase','maximum_time_regression'}
CALIBRATION={'cases','datasets','seed','bins','maximum_rank_chi_square','minimum_coverage','maximum_coverage'}
PROTOCOL={'schema_version','id','purpose','corpus','engines','policy','resources','thresholds','calibration','excluded_claims','engine_differences'}
CASE={'id','kind','dimension','rows','data','partition','step','initial','family','measure','provenance'}
MANIFEST={'schema_version','protocol','corpus','sources','git','toolchains','ledgers','protocol_sha256','corpus_sha256','manifest_sha256'}
RESULT={'schema_version','manifest_sha256','job','host','engine','status','failure','process','phases','metrics','preflight','artifacts','availability'}


class Invalid(ValueError):
    """Schema, provenance, completeness or scientific input failure."""


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(value): return hashlib.sha256(canonical(value)).hexdigest()


def keys(value, expected, label):
    if not isinstance(value,dict) or set(value)!=expected: raise Invalid(f'{label}: fields differ: {set(value) ^ expected if isinstance(value,dict) else "not an object"}')


def integer(value, lo, hi, label):
    if type(value) is not int or not lo<=value<=hi: raise Invalid(f'{label}: integer outside [{lo},{hi}]')


def real(value, lo, hi, label):
    if type(value) not in (int,float) or not math.isfinite(value) or not lo<=value<=hi: raise Invalid(f'{label}: invalid finite number')


def relative(path):
    if not isinstance(path,str) or not path or PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts or '\\' in path: raise Invalid('path must stay relative to the experiment root')
    return path


def sha(value):
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value): raise Invalid('invalid SHA-256')


def _pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise Invalid(f'duplicate JSON key: {key}')
        result[key]=value
    return result


def load(path):
    data=path.read_bytes()
    if len(data)>16_777_216: raise Invalid('JSON byte budget exceeded')
    return json.loads(data,object_pairs_hook=_pairs,parse_constant=lambda _: (_ for _ in ()).throw(Invalid('nonfinite JSON')))


def guarded(function):
    @functools.wraps(function)
    def wrapped(*args,**kwargs):
        try:return function(*args,**kwargs)
        except (TypeError,KeyError,AttributeError,OverflowError) as error:raise Invalid(f'{function.__name__}: malformed typed JSON') from error
    return wrapped


@guarded
def protocol(value):
    keys(value,PROTOCOL,'protocol')
    if value['schema_version']!=VERSION or type(value['schema_version']) is not int: raise Invalid('unsupported protocol schema')
    if value['purpose'] not in ('validation','correctness_ci'): raise Invalid('unsupported experiment purpose')
    if not isinstance(value['id'],str) or not value['id']: raise Invalid('missing protocol identity')
    relative(value['corpus'])
    engines=value['engines']
    if not isinstance(engines,list) or not engines or len(set(engines))!=len(engines) or any(e not in ENGINES for e in engines) or 'alea' not in engines: raise Invalid('unsupported/duplicate engine')
    if set(value['engine_differences'])!=set(engines) or any(not isinstance(v,str) or not v for v in value['engine_differences'].values()):raise Invalid('engine policy differences must be predeclared')
    p=value['policy'];keys(p,POLICY,'policy')
    if (p['algorithm'],p['geometry'],p['adaptation'],p['precision'])!=('multinomial_nuts','identity','none','float64') or p['energy_limit']!=1000.0: raise Invalid('unsupported numerical policy')
    for name,lo,hi in [('chains',2,16),('warmup',0,1_000_000),('draws',4,1_000_000),('depth',1,12),('repetitions',1,100)]: integer(p[name],lo,hi,name)
    if 'nuts-rs' in engines and p['warmup']==0:raise Invalid('pinned nuts-rs requires positive native warmup')
    if len(p['seeds'])!=p['chains'] or len(set(p['seeds']))!=p['chains']: raise Invalid('chain seed count/identity')
    for seed in p['seeds']: integer(seed,1,2**31-1,'seed')
    r=value['resources'];keys(r,LIMITS,'resources')
    if r['threads']!=1: raise Invalid('v1 comparison is serial within a child process')
    integer(r['timeout_seconds'],1,86400,'timeout');integer(r['maximum_output_bytes'],4096,1_073_741_824,'output');integer(r['maximum_cases'],1,128,'cases')
    t=value['thresholds'];keys(t,THRESHOLDS,'thresholds')
    for name,v in t.items(): real(v,0,1e9,name)
    real(t['maximum_rhat'],1,10,'rhat');real(t['confidence'],.5,.9999,'confidence');integer(t['bootstrap_resamples'],100,100_000,'bootstrap')
    for n in ('maximum_divergence_rate','maximum_depth_rate','maximum_failure_rate_increase'): real(t[n],0,1,n)
    for n in ('minimum_ess','mean_mcse_multiplier','gradient_absolute_tolerance','minimum_speedup'):
        if t[n]<=0: raise Invalid('positive threshold required')
    c=value['calibration'];keys(c,CALIBRATION,'calibration');integer(c['datasets'],1,4096,'SBC datasets');integer(c['bins'],2,32,'SBC bins');integer(c['seed'],1,2**31-1,'SBC seed')
    for n in ('minimum_coverage','maximum_coverage'):real(c[n],0,1,n)
    if c['minimum_coverage']>c['maximum_coverage'] or c['maximum_rank_chi_square']<=0 or not c['cases'] or len(set(c['cases']))!=len(c['cases']): raise Invalid('invalid SBC policy')
    if not isinstance(value['excluded_claims'],list) or not value['excluded_claims'] or any(not isinstance(s,str) or not s for s in value['excluded_claims']):raise Invalid('excluded claims required before measurement')
    return value


@guarded
def corpus(value):
    keys(value,{'schema_version','id','cases','vendor'},'corpus')
    if type(value['schema_version']) is not int or value['schema_version']!=1 or not isinstance(value['cases'],list) or not value['cases']:raise Invalid('invalid corpus')
    seen=set()
    for c in value['cases']:
        keys(c,CASE,'case')
        if not isinstance(c['id'],str) or not c['id'] or any(ch not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for ch in c['id']) or c['id'] in seen:raise Invalid('duplicate/invalid case identity')
        seen.add(c['id']);integer(c['kind'],0,11,'model');integer(c['dimension'],1,128,'dimension');integer(c['rows'],0,4096,'rows');real(c['step'],1e-12,100,'step')
        if c['partition'] not in ('tuning','heldout') or not c['family'] or not c['measure'] or not c['provenance']:raise Invalid('missing measure/split/provenance')
        if not isinstance(c['initial'],list) or len(c['initial'])!=c['dimension'] or not isinstance(c['data'],list) or len(c['data'])>100_000:raise Invalid('data/initial shape')
        for v in c['initial']+c['data']:real(v,-1e100,1e100,'model value')
    return value


@guarded
def manifest(value):
    keys(value,MANIFEST,'manifest');protocol(value['protocol']);corpus(value['corpus'])
    if type(value['schema_version']) is not int or value['schema_version']!=1 or len(value['corpus']['cases'])>value['protocol']['resources']['maximum_cases']:raise Invalid('manifest version/case budget')
    if value['protocol_sha256']!=digest(value['protocol']) or value['corpus_sha256']!=digest(value['corpus']):raise Invalid('protocol/corpus digest mismatch')
    body={k:v for k,v in value.items() if k!='manifest_sha256'}
    if value['manifest_sha256']!=digest(body):raise Invalid('manifest seal mismatch')
    for path,expected in value['sources'].items():relative(path);sha(expected)
    p=value['protocol'];policy=p['policy']
    if any((c['dimension']+24)*64*(policy['warmup']+policy['draws'])*policy['chains']>p['resources']['maximum_output_bytes'] for c in value['corpus']['cases']):raise Invalid('planned CSV/native output exceeds admission byte budget')
    ids={c['id']:c for c in value['corpus']['cases']}
    for name in value['protocol']['calibration']['cases']:
        if name not in ids or ids[name]['kind'] not in (5,7,9):raise Invalid('unimplemented generative measure')
    return value


@guarded
def result(value):
    keys(value,RESULT,'result')
    if type(value['schema_version']) is not int or value['schema_version']!=1:raise Invalid('unsupported result schema')
    sha(value['manifest_sha256']);sha(value['host']['fingerprint']);sha(value['process']['binary_sha256'])
    if value['status'] not in ('completed','failed') or value['engine'] not in ENGINES:raise Invalid('invalid result status/engine')
    if (value['status']=='failed')!=(value['failure'] is not None):raise Invalid('failure record was lost or invented')
    if value['failure'] is not None:keys(value['failure'],{'kind','message'},'failure')
    keys(value['job'],{'id','case','engine','repetition','dataset','truth'},'job')
    if value['job']['engine']!=value['engine']:raise Invalid('engine identity mismatch')
    keys(value['process'],{'returncode','wall_seconds','peak_rss_kib','cpu_affinity','command','binary_sha256'},'process')
    if value['process']['wall_seconds'] is not None:real(value['process']['wall_seconds'],0,1e9,'process time')
    for path,artifact in value['artifacts'].items():relative(path);keys(artifact,{'sha256','bytes'},'artifact');sha(artifact['sha256']);integer(artifact['bytes'],0,2**63-1,'artifact size')
    if value['status']=='completed' and (value['process']['returncode']!=0 or value['metrics'] is None or value['phases'] is None):raise Invalid('incomplete success record')
    return value
