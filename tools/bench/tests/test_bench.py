"""Offline contracts and deliberate faults; external engines are not CI dependencies."""
import copy
import csv
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from alea_bench import models,schema
from alea_bench.manifest import ROOT,file_hash,seal,task_text,verify_sources,write_new
from alea_bench.runner import gradient_preflight,run
from alea_bench.report import build_report,interval,load_bundle,quality,verify_metrics
BINARY=Path(os.environ.get('ALEA_BENCH_TEST_BINARY',ROOT/'crates/target/debug/alea-bench'))
if os.environ.get('ALEA_BENCH_REQUIRE_WORKER')=='1' and not BINARY.is_file():
    raise RuntimeError('CI requires the built alea-bench worker; numerical tests must not skip')


class SchemaTests(unittest.TestCase):
    def test_unknown_versions_fields_types_duplicate_keys_and_nonfinite_json_are_rejected(self):
        original=schema.load(HERE/'protocols/smoke-v1.json')
        for change in (lambda p:p.update(schema_version=2),lambda p:p.update(extra=True),lambda p:p['policy'].update(draws=True),lambda p:p.update(engines=[{}]),lambda p:p['thresholds'].update(maximum_rhat=float('nan')),lambda p:p['policy'].update(algorithm='unimplemented_sampler')):
            value=copy.deepcopy(original);change(value)
            with self.assertRaises(schema.Invalid):schema.protocol(value)
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'invalid.json'
            for text in ('{"a":1,"a":2}','{"a":NaN}'):
                path.write_text(text)
                with self.assertRaises(schema.Invalid):schema.load(path)
    def test_corpus_generation_and_vendor_bytes_are_pinned(self):
        subprocess.run([sys.executable,str(HERE/'generate_corpus.py'),'--check'],check=True)
        subprocess.run([sys.executable,str(HERE/'schema_document.py'),'--check'],check=True)
        corpus=schema.corpus(schema.load(HERE/'corpus/v1.json'))
        self.assertEqual({c['partition'] for c in corpus['cases']},{'tuning','heldout'})
        self.assertTrue({'hierarchical','latent','wiener','posteriordb'}<={c['family'] for c in corpus['cases']})
        for name,info in corpus['vendor']['files'].items():self.assertEqual(file_hash(HERE/'corpus/vendor'/name),info['sha256'])
    def test_uncertainty_is_deterministic_and_requires_observations(self):
        self.assertIsNone(interval([],.95,200))
        band=interval([1.,2.,3.],.95,200)
        self.assertEqual(band,interval([1.,2.,3.],.95,200))
        self.assertLessEqual(band['lower'],band['median']);self.assertGreaterEqual(band['upper'],band['median'])
    def test_generative_measure_and_predictions_do_not_mutate_density_inputs(self):
        cases=schema.load(HERE/'corpus/v1.json')['cases']
        for case in (c for c in cases if c['kind'] in (5,7,9)):
            before=copy.deepcopy(case);generated,truth=models.generate(case,16011)
            self.assertEqual(case,before);self.assertTrue(math.isfinite(models.density(generated,truth)))
            self.assertEqual((generated,truth),models.generate(case,16011))
            self.assertEqual(len(models.predictive(case,[truth]*20,16012)),20)


@unittest.skipUnless(BINARY.is_file(),'build alea-bench first (mandatory in CI)')
class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.directory=Path(self.temp.name)
        self.case=copy.deepcopy(schema.load(HERE/'corpus/v1.json')['cases'][0]);self.policy=copy.deepcopy(schema.load(HERE/'protocols/smoke-v1.json')['policy']);self.policy['draws']=128
    def diagnose(self,shift=0.,scale=1.,stuck=False,lost=False,acceptance=.9):
        task=self.directory/'task';task.write_text(task_text(self.case,self.policy));raw=self.directory/'raw.csv';rng=random.Random(16111)
        with raw.open('x') as f:
            writer=csv.writer(f);writer.writerow(['chain','draw','divergent','saturated','steps','acceptance','energy','energy_error','q0','q1'])
            for chain in range(4):
                for draw in range(128-(1 if lost and chain==3 else 0)):
                    q=[shift if stuck else shift+scale*rng.gauss(0,1) for _ in range(2)]
                    writer.writerow([chain,draw,0,0,1,acceptance,'',0.,*q])
        output=self.directory/'diagnostics.json';p=subprocess.run([str(BINARY),'--analyze',str(task),str(raw),str(output)],capture_output=True)
        if lost:return p
        self.assertEqual(p.returncode,0,p.stderr.decode());metrics=schema.load(output);verify_metrics(metrics,self.case,self.policy)
        return quality({'status':'completed','job':{'case':self.case},'metrics':metrics},schema.load(HERE/'protocols/reference-v1.json')['thresholds'])
    def test_stuck_chains_fail_diagnostics(self):self.assertIn('undefined_diagnostic',self.diagnose(stuck=True)['reasons'])
    def test_biased_chains_fail_analytic_means(self):self.assertIn('analytic_mean',self.diagnose(shift=4.)['reasons'])
    def test_variance_bias_fails_second_moments_even_with_good_rhat_and_ess(self):self.assertIn('analytic_second_moment',self.diagnose(scale=.05)['reasons'])
    def test_lost_draw_is_an_error_instead_of_silent_truncation(self):self.assertNotEqual(self.diagnose(lost=True).returncode,0)
    def test_upstream_acceptance_roundoff_is_bounded(self):
        self.diagnose(acceptance=math.nextafter(1.,math.inf))
        # A fresh directory avoids reusing create-new analyzer outputs.
        self.directory=self.directory/'invalid';self.directory.mkdir()
        with self.assertRaises(AssertionError):self.diagnose(acceptance=1.0001)
    def test_wrong_gradient_is_caught_against_independent_density(self):
        original=subprocess.run
        def inject(command,**kwargs):
            p=original(command,**kwargs)
            if '--probe' in command:
                path=Path(command[-1]);value=schema.load(path);value['gradient'][0]+=1.;path.write_text(json.dumps(value))
            return p
        with patch('alea_bench.runner.subprocess.run',side_effect=inject):
            value=gradient_preflight(self.case,str(BINARY),self.policy,self.directory,schema.load(HERE/'protocols/reference-v1.json')['thresholds'])
        self.assertFalse(value['passed'])
    def test_every_corpus_density_and_gradient_matches_the_independent_oracle(self):
        thresholds=schema.load(HERE/'protocols/reference-v1.json')['thresholds']
        for case in schema.load(HERE/'corpus/v1.json')['cases']:
            with self.subTest(case=case['id']):
                directory=self.directory/case['id'];directory.mkdir()
                self.assertTrue(gradient_preflight(case,str(BINARY),self.policy,directory,thresholds)['passed'])
    def test_worker_refuses_unbounded_or_malformed_task(self):
        path=self.directory/'bad';path.write_text('1 0 128 0 0 1000000 5 .1 4 128 '+'0 '*300)
        p=subprocess.run([str(BINARY),'--worker',str(path),str(self.directory)],capture_output=True)
        self.assertNotEqual(p.returncode,0)


@unittest.skipUnless(BINARY.is_file(),'build alea-bench first (mandatory in CI)')
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Sealed corpus/protocol paths must remain repository-relative. A fresh
        # checkout has no target directory until this fixture creates it.
        (ROOT/'target').mkdir(exist_ok=True)
        cls.temp=tempfile.TemporaryDirectory(dir=ROOT/'target',prefix='alea-bench-test-');cls.base=Path(cls.temp.name)
        cls.addClassCleanup(cls.temp.cleanup)
        p=schema.load(HERE/'protocols/smoke-v1.json');case=next(c for c in schema.load(HERE/'corpus/v1.json')['cases'] if c['kind']==7)
        corpus=schema.load(HERE/'corpus/v1.json');corpus['cases']=[case];write_new(cls.base/'corpus.json',corpus)
        p['corpus']=(cls.base/'corpus.json').relative_to(ROOT).as_posix();p['calibration']['datasets']=2;write_new(cls.base/'protocol.json',p)
        cls.manifest=seal(cls.base/'protocol.json',cls.base/'manifest.json')
        config=dict(schema_version=1,binaries={'alea':str(BINARY)},binary_sha256={'alea':file_hash(BINARY)},records=[],rustc='CI debug worker')
        write_new(cls.base/'engines.json',config);cls.bundle=cls.base/'run'
        run(cls.base/'manifest.json',cls.base/'engines.json',cls.bundle)
    def test_second_reader_regenerates_identical_report_from_manifest_and_bundle(self):
        report=build_report(self.bundle);self.assertEqual(report,build_report(self.bundle));self.assertFalse(report['promotion']['ledger_changed'])
    def test_boolean_schema_versions_cannot_alias_integer_one(self):
        record=schema.load(next((self.bundle/'jobs').glob('*/result.json')))
        for validate,original in ((schema.corpus,self.manifest['corpus']),(schema.manifest,self.manifest),(schema.result,record)):
            value=copy.deepcopy(original);value['schema_version']=True
            if validate is schema.manifest:value['manifest_sha256']=schema.digest({k:v for k,v in value.items() if k!='manifest_sha256'})
            with self.assertRaises(schema.Invalid):validate(value)
    def test_measurements_reject_changed_sources_after_seal(self):
        path=self.base/'corpus.json';original=path.read_bytes()
        try:
            path.write_bytes(original+b' ')
            with self.assertRaises(schema.Invalid):verify_sources(self.manifest)
        finally:path.write_bytes(original)
    def test_lost_failure_record_or_schema_drift_cannot_improve_a_report(self):
        path=self.bundle/'index.json';original=path.read_bytes();value=schema.load(path);value['records'].pop(next(iter(value['records'])))
        try:
            path.write_text(json.dumps(value))
            with self.assertRaises(schema.Invalid):load_bundle(self.bundle)
        finally:path.write_bytes(original)
        result_path=next((self.bundle/'jobs').glob('*/result.json'));before=result_path.read_bytes();r=schema.load(result_path);r['schema_version']=2
        try:
            result_path.write_text(json.dumps(r))
            with self.assertRaises(schema.Invalid):load_bundle(self.bundle)
        finally:result_path.write_bytes(before)
    def test_threshold_edits_after_results_are_rejected_even_if_the_manifest_is_resealed(self):
        path=self.bundle/'manifest.json';original=path.read_bytes();value=schema.load(path);value['protocol']['thresholds']['maximum_rhat']=9.
        value['protocol_sha256']=schema.digest(value['protocol']);value['manifest_sha256']=schema.digest({k:v for k,v in value.items() if k!='manifest_sha256'})
        try:
            path.write_text(json.dumps(value))
            with self.assertRaises(schema.Invalid):load_bundle(self.bundle)
        finally:path.write_bytes(original)
    def test_backend_failure_is_persisted_and_included_in_promotion_denominator(self):
        config=schema.load(self.base/'engines.json');config['binaries']['alea']=shutil.which('false');config['binary_sha256']['alea']=file_hash(Path(config['binaries']['alea']));write_new(self.base/'failing-engines.json',config)
        directory=self.base/'failed';records=run(self.base/'manifest.json',self.base/'failing-engines.json',directory)
        self.assertTrue(all(r['status']=='failed' and r['failure'] for r in records));report=build_report(directory)
        self.assertEqual(report['failure_rates']['alea'],1.);self.assertEqual(report['promotion']['decision'],'not_eligible')

class TrackingTests(unittest.TestCase):
    def test_hosts_are_separate_and_regression_bands_never_fail_correctness_ci(self):
        from alea_bench.report import tracking_summary
        observations=[]
        for host,build,seconds in [('a','old',1.),('a','new',2.),('b','old',10.),('b','new',9.)]:
            for _ in range(3):observations.append(dict(host=host,corpus_sha256='corpus',policy_sha256='policy',case='normal',engine='alea',build_sha256=build,process_seconds=seconds,quality_passed=True,status='completed'))
        result=tracking_summary(observations,resamples=200)
        self.assertEqual(result['hosts'],['a','b']);self.assertEqual(len(result['strata']),4)
        self.assertTrue(result['correctness_ci_is_timing_independent'])
        self.assertTrue(next(v for v in result['regressions'] if v['host']=='a')['informational_alert'])
        self.assertFalse(next(v for v in result['regressions'] if v['host']=='b')['informational_alert'])
        observations[0]['quality_passed']=False
        result=tracking_summary(observations,resamples=200)
        self.assertIsNone(next(v for v in result['regressions'] if v['host']=='a')['time_ratio'])


if __name__=='__main__':unittest.main()
