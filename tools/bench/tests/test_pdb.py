"""Descriptive PosteriorDB contracts: failures, raw completeness and no decisions."""
import copy,csv,gzip,hashlib,io,json,sys,tarfile,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alea_bench import pdb,schema
from alea_bench.manifest import ROOT,file_hash,write_new
from alea_bench.pdb_external import json_init
from alea_bench.pdb_report import build_report,coordinate_summary,reference_comparison,interactive

PROTOCOL=ROOT/'tools/bench/protocols/posteriordb-v2.json'

def tar(path,files):
 with tarfile.open(path,'w:gz') as archive:
  for name,data in files.items():
   item=tarfile.TarInfo(name);item.size=len(data);archive.addfile(item,io.BytesIO(data))

def fixture(directory):
 p=copy.deepcopy(schema.load(PROTOCOL));p.update(methods=['nuts_cov_diag'],chains=2,seeds=[11,12],repetitions=1,warmup=16,draws=8)
 source=b'model {}';sha=hashlib.sha256(source).hexdigest();case={'id':'example','family':'test','directory':'example','files':{'model.stan':sha}}
 metadata={'dimension':1,'parameter_names':'theta','derivative_validation':{'hvp_error':None}}
 m={'schema_version':2,'purpose':'descriptive','protocol':p,'corpus':{'schema_version':2,'cases':[case]},'preparation':{'models':{'example':{'metadata':metadata,'builds':{'stan':{'returncode':0}}}}},'sources':{'example/model.stan':sha},'toolchains':{},'reference_defaults_changed':False}
 m['manifest_sha256']=schema.digest(m);write_new(directory/'manifest.json',m);job=pdb.jobs(m)[0]
 tar(directory/'sources.tar.gz',{'example/model.stan':source});tar(directory/'preparation.tar.gz',{})
 write_new(directory/'host.json',{'physical_host_id':'test'})
 write_new(directory/'plan.json',{'schema_version':2,'manifest_sha256':m['manifest_sha256'],'jobs':[job],'host_sha256':file_hash(directory/'host.json'),'sources_sha256':file_hash(directory/'sources.tar.gz'),'preparation_sha256':file_hash(directory/'preparation.tar.gz')})
 folder=directory/'jobs'/job['id'];folder.mkdir(parents=True)
 for filename,skip,names in [('draws.csv.gz',8,['q0']),('constrained.csv.gz',2,['theta'])]:
  with gzip.open(folder/filename,'wt',newline='') as out:
   writer=csv.writer(out);writer.writerow(['chain','draw']+(['divergent','saturated','steps','acceptance','energy','energy_error'] if skip==8 else [])+names)
   for c in range(2):
    for i in range(8):writer.writerow([c,i]+([0,0,1,.8,'',''] if skip==8 else [])+[float(i)])
 summary={'mean':3.5,'sd':2.366,'mcse':1.,'rhat':1.2,'bulk_ess':8.,'tail_ess':5.,'error':None}
 metrics={key:{'draws':16,'divergences':0,'depth_saturations':0,'integration_steps':16,'coordinates':[{'name':name,'summary':summary}]} for key,name in [('unconstrained','q0'),('constrained','theta')]};write_new(folder/'metrics.json',metrics)
 row={'schema_version':2,'manifest_sha256':m['manifest_sha256'],'job':job,'status':'completed','failure':None,'process':{'returncode':0,'timed_out':False,'wall_seconds':.001234567,'wall_nanoseconds':1234567,'peak_rss_kib':1024},'worker':{'phases':[{'warmup_seconds':.0004,'retained_seconds':.0005}]},'metrics':metrics,'artifacts':{x.name:{'sha256':file_hash(x),'bytes':x.stat().st_size} for x in folder.iterdir()}}
 write_new(folder/'result.json',row);write_new(directory/'index.json',{'schema_version':2,'manifest_sha256':m['manifest_sha256'],'records':{job['id']:file_hash(folder/'result.json')}})
 return m,row,folder

class ProtocolTests(unittest.TestCase):
 def test_descriptive_protocol_rejects_decision_fields_versions_and_unknown_methods(self):
  original=schema.load(PROTOCOL);pdb.validate_protocol(original)
  for mutation in [lambda p:p.update(promotion=True),lambda p:p.update(thresholds={}),lambda p:p.update(schema_version=True),lambda p:p.update(purpose='promotion'),lambda p:p.update(methods=['unknown']),lambda p:p.update(notes='missing'),lambda p:p.update(seeds=[2**31-1]*4)]:
   p=copy.deepcopy(original);mutation(p)
   with self.assertRaises(schema.Invalid):pdb.validate_protocol(p)
 def test_full_matrix_preserves_unsupported_cases_and_independent_repetitions(self):
  p=schema.load(PROTOCOL);corpus=schema.load(ROOT/p['corpus']);models={c['id']:{'metadata':{'dimension':26 if c['family']=='hierarchical_binomial' else 1,'derivative_validation':{'hvp_error':None}},'builds':{'stan':{'returncode':0}}} for c in corpus['cases']}
  jobs=pdb.jobs({'protocol':p,'corpus':corpus,'preparation':{'models':models}})
  self.assertEqual(len(jobs),1560);self.assertEqual(len({j['id'] for j in jobs}),1560)
  self.assertTrue(any(j['unsupported_reason'] for j in jobs));self.assertEqual({j['seeds'][0] for j in jobs},{18001,28001,38001})
 def test_hvp_failure_only_excludes_hvp_workflow(self):
  p=schema.load(PROTOCOL);p['methods']=['nuts_atlas_joint','nuts_cov_diag'];p['repetitions']=1
  m={'protocol':p,'corpus':{'cases':[{'id':'hmm','family':'hmm'}]},'preparation':{'models':{'hmm':{'metadata':{'dimension':4,'derivative_validation':{'hvp_error':'nonfinite'}},'builds':{}}}}}
  atlas,nuts=pdb.jobs(m);self.assertIsNotNone(atlas['unsupported_reason']);self.assertIsNone(nuts['unsupported_reason'])
 def test_vendored_sources_and_reference_draws_match_inventory(self):
  corpus=schema.load(ROOT/'tools/bench/corpus/pdb-v2.json');self.assertEqual(len(corpus['cases']),20)
  for case in corpus['cases']:
   for name,sha in case['files'].items():self.assertEqual(file_hash(ROOT/case['directory']/name),sha)
 def test_initial_parameters_preserve_matrix_indexes(self):
  self.assertEqual(json_init('a.1.1,a.2.1,a.1.2,a.2.2,b',[1,2,3,4,5]),{'a':[[1,3],[2,4]],'b':5})

class ArtifactTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.directory=Path(self.temp.name);self.m,self.row,self.folder=fixture(self.directory)
 def replace_record(self):
  (self.folder/'result.json').write_text(json.dumps(self.row));index=schema.load(self.directory/'index.json');index['records'][self.row['job']['id']]=file_hash(self.folder/'result.json');(self.directory/'index.json').write_text(json.dumps(index))
 def test_offline_report_is_descriptive_and_retains_nanosecond_precision(self):
  report=build_report(self.directory);self.assertEqual(report['summary']['planned_jobs'],1);self.assertEqual(report['rows'][0]['process_nanoseconds'],1234567);self.assertNotIn('promotion',report);self.assertNotIn('passed',report['groups'][0]);self.assertIn('log retained',interactive(report))
 def test_missing_failed_job_cannot_be_dropped(self):
  index=schema.load(self.directory/'index.json');index['records'].clear();(self.directory/'index.json').write_text(json.dumps(index))
  with self.assertRaises(schema.Invalid):pdb.load(self.directory)
 def test_raw_or_source_mutation_is_rejected(self):
  with (self.folder/'draws.csv.gz').open('ab') as f:f.write(b'changed')
  with self.assertRaises(schema.Invalid):pdb.load(self.directory)
 def test_lost_draw_is_rejected_even_if_artifact_digest_is_updated(self):
  path=self.folder/'draws.csv.gz'
  with gzip.open(path,'rt') as stream:lines=stream.readlines()
  with gzip.open(path,'wt') as stream:stream.writelines(lines[:-1])
  self.row['artifacts'][path.name]={'sha256':file_hash(path),'bytes':path.stat().st_size};self.replace_record()
  with self.assertRaises(schema.Invalid):pdb.load(self.directory)
 def test_timeout_cannot_be_declared_completed(self):
  self.row['process']['timed_out']=True;self.replace_record()
  with self.assertRaises(schema.Invalid):pdb.load(self.directory)
 def test_split_bundle_round_trip_and_changed_part_rejection(self):
  from alea_bench.pdb_bundle import pack,unpack
  bundle=self.directory.parent/(self.directory.name+'-bundle');extracted=self.directory.parent/(self.directory.name+'-extract')
  import shutil
  self.addCleanup(lambda:shutil.rmtree(bundle,ignore_errors=True));self.addCleanup(lambda:shutil.rmtree(extracted,ignore_errors=True))
  pack(self.directory,bundle,limit=1024);restored=unpack(bundle,extracted)
  self.assertEqual(build_report(restored),build_report(self.directory))
  part=next(bundle.glob('*.part*'))
  with part.open('ab') as stream:stream.write(b'changed')
  with self.assertRaises(schema.Invalid):unpack(bundle,extracted/'invalid')
 def test_failures_remain_visible_without_statistical_verdict(self):
  self.row.update(status='failed',failure={'kind':'NumericalError','message':'domain'});self.replace_record();report=build_report(self.directory)
  self.assertEqual(report['summary']['status_counts'],{'failed':1});self.assertEqual(report['groups'][0]['failed'],1);self.assertEqual(report['groups'][0]['completed'],0)
 def test_empirical_accuracy_and_undefined_diagnostics_are_separate(self):
  metric={'coordinates':[{'name':'theta.1','summary':{'mean':5.,'sd':2.,'mcse':None,'rhat':None,'bulk_ess':None,'tail_ess':None,'error':'constant chain'}}]}
  comparison=reference_comparison(metric,{'theta[1]':{'mean':1.,'sd':2.}})[0]
  self.assertEqual(comparison['mean_error_reference_sd'],2.);self.assertEqual(comparison['posterior_sd_ratio'],1.);self.assertEqual(coordinate_summary(metric)['undefined_diagnostics'],1);self.assertIsNone(coordinate_summary(metric)['minimum_bulk_ess'])

class VersionThreeTests(unittest.TestCase):
 def test_explicit_native_workflows_and_six_target_long_protocol(self):
  protocol=schema.load(ROOT/'tools/bench/protocols/posteriordb-v3.json');pdb.validate_protocol(protocol)
  self.assertEqual(len(protocol['cases']),20);self.assertEqual(len(protocol['methods']),28)
  self.assertNotIn('nuts_atlas_joint',protocol['methods']);self.assertNotIn('nuts_pathfinder',protocol['methods'])
  self.assertTrue({'atlas_full','nuts_saia_joint','nuts_pathfinder_raw','nuts_pathfinder_psis'}<=set(protocol['methods']))
  long=schema.load(ROOT/'tools/bench/protocols/posteriordb-focused-v3.json');pdb.validate_protocol(long)
  self.assertEqual(len(long['cases']),6);self.assertEqual((long['warmup'],long['draws']),(1000,2000))
 def test_versions_cannot_silently_mix_method_policies(self):
  for name,method in [('posteriordb-v2.json','nuts_saia_joint'),('posteriordb-v3.json','nuts_atlas_joint')]:
   protocol=schema.load(ROOT/'tools/bench/protocols'/name);protocol['methods']=[method]
   with self.assertRaises(schema.Invalid):pdb.validate_protocol(protocol)
 def test_third_ad_build_failures_stay_in_the_v3_denominator(self):
  p=schema.load(ROOT/'tools/bench/protocols/posteriordb-v3.json');p.update(methods=['rmhmc_softabs','nuts_cov_diag'],repetitions=1,cases=['earnings-earn_height'])
  model={'metadata':{'dimension':3,'derivative_validation':{'hvp_error':None}},'builds':{'stan':{'returncode':0}},'third_ad':{'returncode':1}}
  jobs=pdb.jobs({'protocol':p,'corpus':{'cases':[{'id':'earnings-earn_height','family':'linear_regression'}]},'preparation':{'models':{'earnings-earn_height':model}}})
  self.assertEqual(len(jobs),2);self.assertTrue(all(j['workflow_version']==3 for j in jobs))
  self.assertIsNotNone(jobs[0]['unsupported_reason']);self.assertIsNone(jobs[1]['unsupported_reason'])
 def test_v3_bundle_retains_versioned_identity(self):
  from alea_bench.pdb_bundle import pack,unpack
  with tempfile.TemporaryDirectory() as temp:
   directory=Path(temp)/'run';directory.mkdir();m,row,folder=fixture(directory)
   m['schema_version']=3;m['reference_defaults_changed']=True;m['protocol'].update(schema_version=3,cases=['example']);m['manifest_sha256']=schema.digest({k:v for k,v in m.items() if k!='manifest_sha256'})
   job=pdb.jobs(m)[0];row.update(schema_version=3,manifest_sha256=m['manifest_sha256'],job=job)
   (directory/'manifest.json').write_text(json.dumps(m));(folder/'result.json').write_text(json.dumps(row))
   plan=schema.load(directory/'plan.json');plan.update(schema_version=3,manifest_sha256=m['manifest_sha256'],jobs=[job]);(directory/'plan.json').write_text(json.dumps(plan))
   (directory/'index.json').write_text(json.dumps({'schema_version':3,'manifest_sha256':m['manifest_sha256'],'records':{job['id']:file_hash(folder/'result.json')}}))
   bundle=Path(temp)/'bundle';pack(directory,bundle,limit=1024);restored=unpack(bundle,Path(temp)/'restore')
   self.assertEqual(restored.name,'posteriordb-v3');self.assertEqual(build_report(restored),build_report(directory))

if __name__=='__main__':unittest.main()
