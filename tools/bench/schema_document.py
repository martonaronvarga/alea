"""Generate structural JSON Schemas; the library additionally checks digests/laws."""
import argparse
import json
from pathlib import Path
from alea_bench import schema

HERE=Path(__file__).resolve().parent

def object_schema(properties):
    return {'type':'object','additionalProperties':False,'required':sorted(properties),'properties':properties}

def document():
    number={'type':'number'};integer={'type':'integer','minimum':0};text={'type':'string'};nullable={'type':['number','null']};sha={'type':'string','pattern':'^[0-9a-f]{64}$'}
    array=lambda item:{'type':'array','items':item}
    ref=lambda name:{'$ref':'#/$defs/'+name}
    policy=object_schema({k:integer for k in schema.POLICY})
    policy['properties'].update(algorithm={'const':'multinomial_nuts'},geometry={'const':'identity'},adaptation={'const':'none'},precision={'const':'float64'},energy_limit={'const':1000.0},seeds=array(integer))
    case=object_schema({k:text for k in schema.CASE});case['properties'].update(kind={'type':'integer','minimum':0,'maximum':11},dimension={'type':'integer','minimum':1,'maximum':128},rows=integer,step={'type':'number','exclusiveMinimum':0},initial=array(number),data=array(number),partition={'enum':['tuning','heldout']})
    protocol=object_schema({k:text for k in schema.PROTOCOL});protocol['properties'].update(schema_version={'const':1},purpose={'enum':['validation','correctness_ci']},engines={'type':'array','uniqueItems':True,'minItems':1,'items':{'enum':list(schema.ENGINES)}},policy=ref('policy'),resources=object_schema({k:integer for k in schema.LIMITS}),thresholds=object_schema({k:number for k in schema.THRESHOLDS}),calibration=object_schema({k:(array(text) if k=='cases' else number) for k in schema.CALIBRATION}),excluded_claims={'type':'array','minItems':1,'items':text},engine_differences={'type':'object','additionalProperties':text})
    corpus=object_schema({'schema_version':{'const':1},'id':text,'cases':array(ref('case')),'vendor':{'type':'object'}})
    manifest=object_schema({k:sha for k in schema.MANIFEST});manifest['properties'].update(schema_version={'const':1},protocol=ref('protocol'),corpus=ref('corpus'),sources={'type':'object','additionalProperties':sha},git=object_schema({'head':text,'worktree_content_is_authoritative':{'const':True}}),toolchains={'type':'object','additionalProperties':text},ledgers={'type':'object'})
    diagnostic=object_schema({k:nullable for k in ('mean','rhat','bulk_ess','tail_ess','mcse') }|{'error':{'type':['string','null']}})
    metrics=object_schema({k:integer for k in ('schema_version','draws','divergences','depth_saturations','integration_steps','acceptance_observations','nonfinite_energy_errors')}|{'coordinates':array(ref('diagnostic')),'second_moments':array(ref('diagnostic')),'ebfmi':array(nullable),'mean_acceptance':nullable,'maximum_reported_energy_error':nullable})
    result=object_schema({k:{'type':'object'} for k in schema.RESULT});result['properties'].update(schema_version={'const':1},manifest_sha256=sha,job=object_schema({'id':text,'case':ref('case'),'engine':{'enum':list(schema.ENGINES)},'repetition':integer,'dataset':{'type':['integer','null']},'truth':{'type':['array','null'],'items':number}}),engine={'enum':list(schema.ENGINES)},status={'enum':['completed','failed']},failure={'oneOf':[{'type':'null'},object_schema({'kind':text,'message':text})]},process=object_schema({'returncode':{'type':['integer','null']},'wall_seconds':nullable,'peak_rss_kib':{'type':['integer','null']},'cpu_affinity':array(integer),'command':array(text),'binary_sha256':sha}),phases={'type':['object','null'],'additionalProperties':nullable},metrics={'oneOf':[{'type':'null'},ref('metrics')]},preflight={'type':['object','null']},artifacts={'type':'object','additionalProperties':object_schema({'sha256':sha,'bytes':integer})})
    return {'$schema':'https://json-schema.org/draft/2020-12/schema','$id':'urn:alea:bench:1','$comment':'Structural contract. alea_bench.schema enforces cross-field budgets, parameter/engine compatibility, digest seals, completeness and scientific laws. Availability is explicitly extensible, with measured values or null/reasons.','oneOf':[ref('manifest'),ref('result'),ref('protocol')],'$defs':{'policy':policy,'case':case,'protocol':protocol,'corpus':corpus,'manifest':manifest,'result':result,'diagnostic':diagnostic,'metrics':metrics}}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args();path=HERE/'schemas/v1.json';content=json.dumps(document(),indent=2,sort_keys=True)+'\n'
    if args.check:
        if path.read_text()!=content:raise SystemExit('structural schema drift; deliberately review the contract')
    else:path.write_text(content)
