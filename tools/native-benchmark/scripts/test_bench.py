import importlib.util
from pathlib import Path
import tempfile
import copy
import json
import subprocess
import sys
from unittest.mock import patch
import compare
import kernel_compare
import unittest
from compare import performance_regressions, compare_work
from kernel_compare import compatible as kernel_compatible, summarize as summarize_kernels
from kernel_compare import SUITES
from compare import summarize as summarize_paired
from replay import rebuild, pack, unpack, capsule_index, relative

spec = importlib.util.spec_from_file_location('bench', Path(__file__).with_name('bench.py'))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class SummaryTests(unittest.TestCase):
    def test_host_metadata_records_the_measurement_host_without_tuning_it(self):
        with patch.object(bench.platform, 'node', return_value='measurement-host'), \
             patch.object(bench.os, 'sched_getaffinity', return_value={3, 1}), \
             patch.object(bench.os, 'getloadavg', return_value=(0.1, 0.2, 0.3)):
            metadata=bench.host_metadata()
        self.assertEqual(metadata['hostname'],'measurement-host')
        self.assertEqual(metadata['affinity'],[1,3])
        self.assertEqual(metadata['load_average'],[0.1,0.2,0.3])
        self.assertIsInstance(metadata['governors'],dict)
        self.assertIsInstance(metadata['power_supplies'],dict)

    def test_capsule_index_rejects_malformed_metadata_before_restoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir()
            (source/'data').write_bytes(b'original')
            index=pack(source,root/'bundle')
            for invalid in [None, [], {'bytes':1,'sha256':False}]:
                changed=copy.deepcopy(index);changed['files']['data']=invalid
                (root/'bundle/capsule.json').write_text(json.dumps(changed))
                with self.assertRaises(ValueError):unpack(root/'bundle',root/'uncreated')
                self.assertFalse((root/'uncreated').exists())
            (root/'valid.json').write_text(json.dumps(index))
            (root/'bundle/capsule.json').unlink()
            (root/'bundle/capsule.json').symlink_to(root/'valid.json')
            with self.assertRaisesRegex(ValueError,'symlink'):capsule_index(root/'bundle')
            for name in [None, 1, '../bad', '/absolute', 'a/./b', 'a\\b']:
                with self.assertRaises(ValueError):relative(name)

    def test_generic_capsule_round_trip_seals_every_byte_and_refuses_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir()
            (source/'failures.json').write_text('{"failed":true}\n')
            (source/'nested').mkdir();(source/'nested/data.bin').write_bytes(bytes(range(256))*8)
            index=pack(source,root/'bundle',limit=128)
            self.assertGreater(len(index['parts']),1)
            restored=unpack(root/'bundle',root/'restore')
            for name in index['files']:
                self.assertEqual((source/name).read_bytes(),(restored/name).read_bytes())
            with self.assertRaises(FileExistsError):unpack(root/'bundle',root/'restore')
            part=root/'bundle'/index['parts'][0]['name'];part.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):unpack(root/'bundle',root/'bad')
            self.assertFalse((root/'bad').exists())
            with self.assertRaises(ValueError):pack(source,source/'recursive')
            (source/'link').symlink_to(source/'failures.json')
            with self.assertRaisesRegex(ValueError,'symlink'):pack(source,root/'linked')

    def test_conformance_replay_retains_matrix_and_separates_target_runtime_and_sampling(self):
        cases=[dict(id='a',family='hierarchy',reference_available=True)]
        manifest=dict(protocol=dict(cases=['a'],methods=['nuts'],repetitions=5),corpus=dict(cases=cases),
                      preparation=dict(native=dict(conformance={'a':{'status':'validated'}})))
        def seal():
            unsigned={k:v for k,v in manifest.items() if k!='manifest_sha256'}
            manifest['manifest_sha256']=__import__('hashlib').sha256(json.dumps(unsigned,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        seal()
        diag=dict(maximum_rhat=1.001,minimum_bulk_ess=800.,minimum_tail_ess=700.,undefined_diagnostics=0)
        base=dict(case='a',method='nuts',status='completed',chains=4,diagnostics=diag,
                  constrained_diagnostics=diag,divergences=0,process_seconds=2.,backend='native',guarantee='reference')
        rows=[dict(base,repetition=0),dict(base,repetition=1,divergences=1),
              dict(base,repetition=2,status='failed',failure={'kind':'BackendFailure'}),
              dict(base,repetition=3,diagnostics=None)]
        report=dict(manifest_sha256=manifest['manifest_sha256'],rows=rows)
        result=bench.conformance_summary(manifest,report)
        self.assertEqual(result['planned'],5)
        self.assertEqual(result['classifications'],dict(qualified_sampling=1,sampling_unqualified=1,runtime_failure=1,diagnostics_unavailable=1,unstarted=1))
        qualified=[r for r in result['rows'] if r['classification']=='qualified_sampling'][0]
        self.assertEqual(qualified['qualified_bulk_ess_per_process_second'],400.)
        self.assertTrue(all(r['qualified_bulk_ess_per_process_second'] is None for r in result['rows'] if r is not qualified))
        report['rows'].append(dict(base,repetition=0))
        with self.assertRaisesRegex(ValueError,'duplicate'):bench.conformance_summary(manifest,report)
        report['rows'].pop();manifest['preparation']['native']['conformance']['a']={'status':'failed','reason':'derivative preflight'};seal()
        report['manifest_sha256']=manifest['manifest_sha256']
        self.assertEqual(bench.conformance_summary(manifest,report)['rows'][0]['classification'],'target_failure')
        manifest['preparation']['native']['conformance']['a']={'status':'unsupported','reason':'missing source rule'};seal()
        report['manifest_sha256']=manifest['manifest_sha256']
        unsupported=bench.conformance_summary(manifest,report)['rows'][0]
        self.assertEqual(unsupported['classification'],'unsupported')
        self.assertEqual(unsupported['representation'],'unsupported')
        self.assertEqual(unsupported['target'],'unavailable')
        manifest['preparation']['native']['conformance']['a']={'status':'pending'};seal()
        report['manifest_sha256']=manifest['manifest_sha256']
        self.assertEqual(bench.conformance_summary(manifest,report)['rows'][0]['classification'],'target_unqualified')
        manifest['protocol']['repetitions']=6
        with self.assertRaisesRegex(ValueError,'identity'):bench.conformance_summary(manifest,report)
    def test_fusion_suite_keeps_every_checkpoint_control_and_exact_dag_admission(self):
        cases = SUITES['tensor-fusion']
        self.assertEqual(cases[:14], SUITES['tensor-checkpoint'])
        self.assertEqual(cases[14:], [('tensor_dag',1000,10000),
                                     ('tensor_dag',10000,1000),('tensor_dag',100000,100)])
        self.assertEqual(len(cases), len(set(cases)))
        for row in summarize_kernels([], cases, reps=7):
            self.assertIsNone(row['paired_gain'])
            self.assertTrue(all(engine['unstarted']==7 for engine in row['engines'].values()))
        a=dict(case='tensor_dag',size=10000,calls=1000,checksum=1.)
        self.assertFalse(kernel_compatible(a,dict(a,checksum=1.+1e-12),'tensor_dag'))

    def test_tensor_suite_keeps_scalar_matrix_branch_and_scan_controls(self):
        cases = SUITES['tensor']
        self.assertEqual(len(cases), 10)
        self.assertEqual({case for case, _, _ in cases},
                         {'tensor', 'tensor_scalar', 'tensor_matmul', 'tensor_cond', 'tensor_scan'})
        self.assertEqual(len(set(cases)), len(cases))
        for row in summarize_kernels([], cases, reps=7):
            self.assertIsNone(row['paired_gain'])
            self.assertTrue(all(engine['unstarted']==7 for engine in row['engines'].values()))
        a=dict(case='tensor', size=1000, calls=10000, checksum=1.)
        self.assertFalse(kernel_compatible(a,dict(a,checksum=1.+1e-12),'tensor'))

    def test_checkpoint_suite_retains_short_scan_and_unrelated_controls(self):
        cases = SUITES['tensor-checkpoint']
        self.assertEqual(cases[:10], SUITES['tensor'])
        self.assertEqual(cases[10:], [('tensor_scan',1025,1000), ('tensor_scan',10000,100),
                                    ('tensor_scan_checkpoint',1025,1000),
                                    ('tensor_scan_checkpoint',10000,100)])
        self.assertEqual(len(cases), len(set(cases)))
        for row in summarize_kernels([], cases, reps=7):
            self.assertIsNone(row['paired_gain'])
            self.assertTrue(all(engine['unstarted']==7 for engine in row['engines'].values()))

    def test_sufficient_statistics_suite_keeps_common_and_distinct_controls_separate(self):
        cases = SUITES['sufficient-statistics']
        self.assertEqual(len(cases), 6)
        self.assertEqual({(case, size) for case, size, _ in cases},
                         {(case, size) for case in ['poisson', 'poisson_common', 'poisson_scalar']
                          for size in [1000, 10000]})
        for row in summarize_kernels([], cases, reps=7):
            self.assertIsNone(row['paired_gain'])
            for engine in row['engines'].values():
                self.assertEqual(engine['unstarted'], 7)
                self.assertIsNone(engine['seconds'])

    def test_common_poisson_checksum_does_not_admit_changed_work_or_nonfinite_results(self):
        a = dict(case='poisson_common', calls=100, size=1000, checksum=-100.)
        self.assertTrue(kernel_compatible(a, dict(a, checksum=-100.+1e-12), 'poisson_common'))
        for b in [dict(a, calls=101), dict(a, size=10000), dict(a, case='poisson'),
                  dict(a, checksum=float('nan')), dict(a, checksum=-99.)]:
            self.assertFalse(kernel_compatible(a, b, 'poisson_common'))

    def test_kernel_suites_preserve_the_original_protocol_and_declare_new_work(self):
        self.assertEqual(len(SUITES['base']),15)
        self.assertEqual(len(SUITES['mixture-scores']),10)
        self.assertEqual(SUITES['all'],SUITES['base']+SUITES['mixture-scores'])
        self.assertEqual(len(set(SUITES['all'])),25)
        rows=summarize_kernels([],SUITES['mixture-scores'])
        self.assertEqual(len(rows),10)
        self.assertTrue(all(r['case'].startswith('mixture_') for r in rows))

    def test_kernel_checks_require_work_finite_values_and_exact_controls(self):
        a=dict(case='norm',calls=100,size=32,checksum=10.)
        self.assertTrue(kernel_compatible(a,a,'norm'))
        self.assertFalse(kernel_compatible(a,dict(a,calls=101),'norm'))
        self.assertFalse(kernel_compatible(a,dict(a,checksum=10.000000000001),'norm'))
        a=dict(a,case='poisson')
        self.assertTrue(kernel_compatible(a,dict(a,checksum=10.000000000001),'poisson'))
        self.assertFalse(kernel_compatible(a,dict(a,checksum=11.),'poisson'))
        self.assertFalse(kernel_compatible(dict(a,checksum=float('inf')),dict(a,checksum=float('inf')),'poisson'))

    def test_kernel_summary_excludes_failed_attempts(self):
        rows=[dict(case='norm',size=4,engine='before',rep=0,success=False),
              dict(case='norm',size=4,engine='after',rep=0,success=True,
                   metrics=dict(seconds=2.,construction_seconds=0.1))]
        result=next(r for r in summarize_kernels(rows) if r['case']=='norm' and r['size']==4)
        self.assertEqual(result['engines']['before']['failed'],1)
        self.assertIsNone(result['engines']['before']['seconds'])
        self.assertIsNone(result['paired_gain'])
        self.assertIsNone(result['engines']['after']['seconds'])
        self.assertEqual(result['engines']['after']['unvalidated_or_incompatible'],1)

    def qualified(self):
        return dict(case='mean',engine='alea',rep=0,success=True,returncode=0,mode='sampling',wall_seconds=2.,
                    metrics=dict(chains=4,completed=4000,dimension=2,evaluations=100,
                        trajectory_hash='a',retained_leapfrog_steps=96,retained_divergences=0,
                        diagnostics=[dict(rhat=1.001,bulk_ess=200,tail_ess=150) for _ in range(2)]))

    def test_ess_requires_complete_finite_all_coordinate_multichain_diagnostics(self):
        row = self.qualified()
        self.assertEqual(bench.summarize([row])[0]['min_bulk_ess_per_second']['median'],100.)
        mutations = [dict(rhat=1.2),dict(rhat=float('nan')),dict(rhat=float('inf')),
                     dict(rhat=None),dict(bulk_ess=99),dict(tail_ess=99),dict(bulk_ess=True)]
        for mutation in mutations:
            bad = copy.deepcopy(row)
            bad['metrics']['diagnostics'][1].update(mutation)
            bad['diagnostic_pass'] = True  # cached historical flags are not authority
            with self.subTest(mutation=mutation):
                self.assertIsNone(bench.summarize([bad])[0]['min_bulk_ess_per_second'])
                self.assertIsNotNone(bench.summarize([bad])[0]['wall_seconds'])
        for mutation in [dict(chains=1),dict(completed=3999),dict(diagnostics=[]),
                         dict(retained_divergences=1),dict(diagnostics=[row['metrics']['diagnostics'][0]]),
                         dict(diagnostics=[{},{}])]:
            bad = copy.deepcopy(row)
            bad['metrics'].update(mutation)
            self.assertIsNone(bench.summarize([bad])[0]['min_bulk_ess_per_second'])
        bad = copy.deepcopy(row)
        bad['metrics']['retained_divergences']=1
        mixed=bench.summarize([row,bad])[0]
        self.assertEqual(mixed['wall_seconds']['n'],2)
        self.assertEqual(mixed['min_bulk_ess_per_second']['n'],1)

    def test_paired_ratios_require_validation_and_ess_requires_both_eligible(self):
        row=self.qualified()
        pair=[dict(row,engine=engine) for engine in ['baseline','current']]
        plan=dict(modes=['sampling'],cases=['mean'],chains=4,reps=7)
        decision=dict(mode='sampling',case='mean',rep=0,compatible=True)
        self.assertIsNone(summarize_paired(pair,plan,[])[0]['baseline_over_current'])
        self.assertIsNone(summarize_paired(pair,plan,[dict(decision,compatible=False)])[0]['baseline_over_current'])
        summary=summarize_paired(pair,plan,[decision])[0]
        self.assertEqual(summary['engines']['baseline']['unstarted'],6)
        self.assertEqual(summary['current_over_baseline_ess_per_second']['median'],1.)
        pair[1]=copy.deepcopy(pair[1])
        pair[1]['metrics']['retained_divergences']=1
        summary=summarize_paired(pair,plan,[decision])[0]
        self.assertIsNotNone(summary['baseline_over_current'])
        self.assertIsNone(summary['current_over_baseline_ess_per_second'])

    def test_kernel_journal_replay_never_promotes_pending_or_incompatible_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'manifest.json').write_text(json.dumps(dict(cases=[['norm',4,100]],reps=7)))
            rows=[dict(case='norm',size=4,rep=0,engine=e,success=True,
                       metrics=dict(seconds=2.,construction_seconds=.1)) for e in ['before','after']]
            (root/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            self.assertIsNone(rebuild(root,'kernel')[0]['paired_gain'])
            for valid in [False,True]:
                (root/'validation.jsonl').write_text(json.dumps(dict(case='norm',size=4,rep=0,compatible=valid))+'\n')
                summary=rebuild(root,'kernel')[0]
                self.assertEqual(summary['engines']['after']['unstarted'],6)
                self.assertEqual(summary['paired_gain'] is not None,valid)
            (root/'records.jsonl').write_text(json.dumps(rows[0])+'\n')
            self.assertIsNone(rebuild(root,'kernel')[0]['engines']['before']['seconds'])
            (root/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            # A torn trailing write is not a completed record. Duplicate completed
            # identities and malformed newline-terminated JSON are hard errors.
            with (root/'records.jsonl').open('a') as stream: stream.write('{')
            self.assertIsNotNone(rebuild(root,'kernel')[0]['paired_gain'])
            with (root/'records.jsonl').open('a') as stream: stream.write('\n')
            with self.assertRaises(ValueError): rebuild(root,'kernel')
            (root/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows+rows))
            with self.assertRaisesRegex(ValueError,'duplicate'): rebuild(root,'kernel')

    def test_malformed_child_metrics_cannot_be_successful(self):
        for text in ['[]','null','{"x":NaN}','{"x":1e999}','{"x":1,"x":2}']:
            with self.assertRaises(ValueError): bench.read_metrics(text)

    def test_stan_requires_every_chain_complete_and_launch_errors_stay_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'draws.csv'
            path.write_text('n_leapfrog__,divergent__,treedepth__\n1,0,1\n')
            with self.assertRaisesRegex(ValueError,'incomplete Stan chain'): bench.parse_stan([path])
            path.write_text('n_leapfrog__,n_leapfrog__\n')
            with self.assertRaisesRegex(ValueError,'duplicate Stan'): bench.parse_stan([path])
        with patch.object(bench.subprocess,'run',side_effect=OSError('injected launch failure')):
            process=bench.run_process(['absent'])
        self.assertNotEqual(process.returncode,0)
        self.assertIn('injected launch failure',process.stderr)

    def test_runner_mismatch_preserves_summary_and_replays_identically(self):
        for module,kind in [(compare,'paired'),(kernel_compare,'kernel')]:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                before,after=root/'before',root/'after'
                before.write_text('before'); after.write_text('after')
                output=root/'run'
                if kind=='paired':
                    argv=['compare.py','--baseline',str(before),'--current',str(after),
                          '--output',str(output),'--reps','1','--modes','evaluation','--cases','mean']
                else:
                    argv=['kernel_compare.py','--before',str(before),'--after',str(after),
                          '--output',str(output),'--reps','1']
                def child(command, **kwargs):
                    checksum=1. if command[0]==str(before) else 2.
                    metrics=(dict(evaluations=10000,evaluation_seconds=1.,checksum=checksum)
                        if kind=='paired' else dict(case='poisson',size=1000,calls=10000,
                            seconds=1.,construction_seconds=.1,checksum=checksum))
                    return subprocess.CompletedProcess(command,0,json.dumps(metrics),'')
                with patch.object(sys,'argv',argv), patch.object(module.os,'sched_setaffinity',create=True), \
                     patch.object(module.subprocess,'run',side_effect=child), \
                     patch.object(module.subprocess,'check_output',return_value='fixture'):
                    with self.assertRaises(AssertionError): module.main()
                summary=json.loads((output/'summary.json').read_text())
                self.assertEqual(summary,rebuild(output,kind))
                self.assertTrue(all(r['success'] for r in map(json.loads,(output/'records.jsonl').read_text().splitlines())))
                self.assertFalse(json.loads((output/'validation.jsonl').read_text())['compatible'])
                key='baseline_over_current' if kind=='paired' else 'paired_gain'
                self.assertIsNone(summary[0][key])
                if kind=='kernel': self.assertEqual(summary[-1]['engines']['after']['unstarted'],1)

    def test_changed_arithmetic_requires_explicit_opt_in(self):
        a = dict(checksum=10.,evaluations=100)
        b = dict(checksum=10.000000000001,evaluations=100)
        with self.assertRaises(AssertionError):
            compare_work(a,b,'evaluation')
        self.assertFalse(compare_work(a,b,'evaluation',True)['checksum'])
        with self.assertRaises(AssertionError):
            compare_work(a,dict(checksum=11.,evaluations=100),'evaluation',True)
        a = dict(trajectory_hash='a',evaluations=100,retained_leapfrog_steps=90,retained_divergences=0)
        b = dict(a,trajectory_hash='b',evaluations=101)
        with self.assertRaises(AssertionError):
            compare_work(a,b,'sampling')
        self.assertFalse(compare_work(a,b,'sampling',True)['evaluations'])

    def test_optional_performance_alert_ignores_failed_fits(self):
        rows = [dict(mode='sampling',case='failed',baseline_over_current=None),
                dict(mode='evaluation',case='slow',baseline_over_current=dict(median=0.5))]
        self.assertEqual(performance_regressions(rows,None),[])
        self.assertEqual(performance_regressions(rows,1.2),[dict(mode='evaluation',case='slow',slowdown=2.)])
        self.assertEqual(performance_regressions(rows,2.),[])

    def test_unstarted_jobs_remain_in_plan(self):
        rows = bench.summarize([],dict(cases=['mean'],reps=7))
        self.assertEqual(len(rows),2)
        self.assertTrue(all(r['planned']==7 and r['unstarted']==7 and r['attempted']==0 for r in rows))

    def test_failures_have_no_eligible_times_and_missing_calls_stay_missing(self):
        rows = [dict(case='linear',engine='alea',success=False,wall_seconds=0.001),
                dict(case='linear',engine='stan',success=True,wall_seconds=2.,
                     metrics=dict(evaluations=None,completed=1000))]
        result = bench.summarize(rows)
        self.assertIsNone(result[0]['wall_seconds'])
        self.assertEqual(result[0]['failed'],1)
        self.assertIsNone(result[1]['evaluations_per_second'])
        self.assertEqual(result[1]['iterations_per_second']['median'],500.)

    def test_successful_subset_and_repeat_spread_are_retained(self):
        rows = [dict(case='mean',engine='alea',success=s,wall_seconds=t,
                     metrics=dict(evaluations=20,completed=10)) for s,t in [(True,1.),(False,.01),(True,3.)]]
        row = bench.summarize(rows)[0]
        self.assertEqual((row['planned'],row['successful'],row['failed']),(3,2,1))
        self.assertEqual(row['wall_seconds']['median'],2.)
        self.assertEqual(row['wall_seconds']['minimum'],1.)
        self.assertEqual(row['wall_seconds']['maximum'],3.)


if __name__ == '__main__':
    unittest.main()
