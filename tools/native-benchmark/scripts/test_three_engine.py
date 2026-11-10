import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import three_engine as bench


class ThreeEngineTests(unittest.TestCase):
    def test_compiler_ad_is_opted_in_per_enzyme_cell_without_rewriting_user_controls(self):
        base={'RUSTFLAGS':'-Ctarget-cpu=native'}
        self.assertEqual(bench.feature_environment(base,[]),base)
        self.assertIn('-Zautodiff=Enable',bench.feature_environment(base,['enzyme'])['RUSTFLAGS'])
        self.assertNotIn('autodiff',base['RUSTFLAGS'])
        explicit={'RUSTFLAGS':'-Zautodiff=Enable -Ctarget-cpu=native'}
        for features in [[],['enzyme']]: self.assertEqual(bench.feature_environment(explicit,features),explicit)
        encoded={'CARGO_ENCODED_RUSTFLAGS':'-Ctarget-cpu=native\x1f-Cembed-bitcode=yes'}
        self.assertTrue(bench.feature_environment(encoded,['enzyme'])['CARGO_ENCODED_RUSTFLAGS'].endswith('\x1f-Zautodiff=Enable'))
        self.assertEqual(bench.feature_environment(encoded,[]),encoded)

    def test_state_banks_require_ten_finite_rows_and_enter_input_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'states.txt'
            path.write_text(' '.join(str(i/10) for i in range(20)))
            bench.validate_state_bank(path,2)
            plan=dict(cases=[],variants=[],state_banks={'control':str(path)})
            self.assertIn(path,bench.matrix_input_paths(plan))
            for text in ['0 1',' '.join(['NaN']*20),' '.join(['inf']*20),'bad']:
                path.write_text(text)
                with self.assertRaises(ValueError): bench.validate_state_bank(path,2)

    def test_cargo_report_reuses_estimates_preserves_failed_cells_and_checks_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for i,value in enumerate([20.,10.]):
                cell=root/f'cell-{i}';path=cell/'criterion/alea-math/primitives/example/new';path.mkdir(parents=True)
                (cell/'features.txt').write_text('' if i==0 else 'simd')
                (cell/'run.log').write_text('complete')
                (path/'estimates.json').write_text(json.dumps(dict(median=dict(point_estimate=value,
                    confidence_interval=dict(lower_bound=value-1,upper_bound=value+1,confidence_level=.95)))))
                (path/'benchmark.json').write_text(json.dumps(dict(full_id='same-work',throughput={'Elements':4})))
                (path/'sample.json').write_text(json.dumps(dict(iters=[10,20],times=[100,200])))
            failed=root/'cell-2';failed.mkdir();(failed/'features.txt').write_text('bad');(failed/'BUILD_FAILED').touch()
            result=bench.cargo_bench_report(root)
            self.assertEqual(result['comparisons'][0]['reference_over_candidate'],2.)
            self.assertEqual(result['cells'][2]['status'],'build-failed')
            self.assertEqual(result['measurements'][0]['iterations'],30)
            self.assertEqual(result['measurements'][0]['namespace'],'alea-math/primitives')
            (path/'benchmark.json').write_text(json.dumps(dict(full_id='same-work',throughput={'Elements':8})))
            with self.assertRaisesRegex(ValueError,'work units'):bench.cargo_bench_report(root)
            (path/'sample.json').write_text(json.dumps(dict(iters=[0],times=[100])))
            with self.assertRaisesRegex(ValueError,'numeric'):bench.cargo_bench_report(root)

    def test_cargo_inventory_exposes_real_targets_and_uncovered_modules_without_timings(self):
        inventory=bench.cargo_bench_inventory(bench.REPO)
        self.assertEqual(len(inventory['packages']),10)
        packages={p['package']:p for p in inventory['packages']}
        self.assertIn('workflow',{t['name'] for t in packages['alea-runtime']['targets']})
        self.assertIn('crates/alea-smc/src/lib.rs',
            [m['path'] for m in packages['alea-smc']['modules'] if m['status']=='declaration-or-test'])
        self.assertTrue(inventory['gaps']) # Never turn package presence into 100% coverage.
        self.assertTrue(all((bench.REPO/t['path']).is_file() for p in packages.values() for t in p['targets']))

    def test_route_inventory_separates_planning_execution_and_higher_order_admission(self):
        def receipt(requested, resolved, order, admission, checkpoint, status, reason='explicit reason'):
            return '\t'.join(['ALEA_BASELINE_ROUTE_V1','fixture',requested,resolved,order,
                              'fixture-default','rust',admission,checkpoint,status,reason]) + '\n'
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cell=root/'cell-0';cell.mkdir()
            (cell/'features.txt').write_text('simd,faer')
            log=cell/'run.log'
            planned=receipt('H','H','first','H','true','not-run')
            log.write_text(planned + receipt('H','H','first','H','true','supported')
                + receipt('H','L','hvp','H','true','not-run')
                + receipt('L','H','first','H','true','unsupported')
                + receipt('N','none','third','N','false','unsupported')
                + receipt('H','L','value','none','true','not-run'))
            report=bench.cargo_bench_report(root)
            self.assertEqual(report['cells'][0]['route_inventory_status'],'recorded')
            routes=report['route_inventory']
            self.assertEqual(len(routes),5)
            self.assertEqual(routes[0]['status'],'supported')
            self.assertEqual(routes[1]['resolved_route'],'L')
            self.assertEqual(routes[1]['first_order_admission_route'],'H')
            self.assertFalse(routes[1]['higher_order_checkpointed'])
            self.assertEqual(routes[2]['status'],'unsupported')
            self.assertEqual(routes[2]['resolved_route'],'H') # Never relabel hybrid as pure L.
            self.assertEqual(routes[4]['first_order_admission_route'],'none') # Primal-only export.
            log.write_text(planned + receipt('H','H','first','H','true','failed'))
            (cell/'RUN_FAILED').touch()
            report=bench.cargo_bench_report(root)
            self.assertEqual(report['cells'][0]['status'],'run-failed')
            self.assertEqual(report['route_inventory'][0]['status'],'failed')
            log.write_text(receipt('L','L','first','H','false','supported',
                                   'compiler derivative with extra checked tensor admission')
                + receipt('L','L','hvp','H','false','supported'))
            report=bench.cargo_bench_report(root)
            self.assertEqual(report['route_inventory'][0]['resolved_route'],'L')
            self.assertEqual(report['route_inventory'][0]['first_order_admission_route'],'H')
            self.assertEqual(report['route_inventory'][1]['status'],'supported')
            log.write_text('old log without route receipts')
            report=bench.cargo_bench_report(root)
            self.assertEqual(report['route_inventory'],[])
            self.assertEqual(report['cells'][0]['route_inventory_status'],'not-recorded')
            for bad in [planned + planned,
                        receipt('H','H','hvp','H','true','supported'),
                        receipt('C','H','first','H','true','supported'),
                        receipt('C','C','first','H','false','supported'),
                        receipt('L','L','first','H','true','supported'),
                        planned.replace('fixture-default','standard'),
                        planned.replace('\trust\t','\tfaer\t'),
                        receipt('H','H','first','H','true','not-run',''),
                        planned.replace('V1','V2'),planned.replace('not-run','guessed')]:
                log.write_text(bad)
                with self.subTest(bad=bad),self.assertRaises(ValueError):
                    bench.cargo_bench_report(root)

    def test_scheduler_affinity_uses_allocated_cpu_ids_and_checks_thread_budget(self):
        with patch.object(bench.os,'sched_getaffinity',return_value={8,10,12}):
            self.assertEqual(bench.allocation_affinity({}),( [8],1))
            self.assertEqual(bench.allocation_affinity(dict(cpus=[12,8],threads=2)),([8,12],2))
            for plan in [dict(cpu=0),dict(cpus=[]),dict(cpus=[8,8]),
                         dict(cpus=[8],threads=2),dict(cpus=[True]),dict(threads=0)]:
                with self.assertRaises(ValueError):bench.allocation_affinity(plan)

    def test_slurm_dry_run_quotes_command_and_does_not_submit(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'submission'
            args=SimpleNamespace(output=out,job_command=['worker','a b','$(touch wrong)'],
                cpus=18,walltime='24:00:00',partition='hpc2019',account=None,
                shared_node=False,dry_run=True,exclude_nodes=['cn18-28'],afterany=['2776558','2776559'])
            with patch.object(bench.subprocess,'run') as run:bench.slurm_submit(args)
            run.assert_not_called()
            script=(out/'job.sh').read_text()
            self.assertIn('SLURM_JOB_ID',script)
            self.assertIn("'$(touch wrong)'",script)
            submission=json.loads((out/'submission.json').read_text())
            self.assertIn('--exclusive',submission['submission'])
            self.assertIn('--exclude=cn18-28',submission['submission'])
            self.assertIn('--dependency=afterany:2776558:2776559',submission['submission'])
            self.assertEqual(submission['status'],'prepared')

    def test_matrix_plan_separates_compiled_features_from_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);builds=root/'variants.json'
            builds.write_text(json.dumps([dict(name='enzyme',features=['enzyme'],binary='/saved/enzyme',
                                               build_receipt='/saved/qualification.json',env={'ALEA_BENCH_BACKEND':'enzyme'})]))
            args=SimpleNamespace(builds=[builds],stan_evaluate=root/'stan',output=root/'plan.json',
                                 cases=['mean','covariance'],reps=7,evaluations=10000,cpu=None)
            bench.matrix_plan(args)
            plan=json.loads(args.output.read_text());variants=plan['variants']
            self.assertEqual(len(variants),5)
            self.assertEqual(variants[0]['env']['ALEA_BENCH_BACKEND'],'native')
            self.assertEqual(variants[1]['env']['ALEA_BENCH_REGRESSION'],'scalar')
            self.assertEqual(variants[2]['cases'],['covariance'])
            self.assertEqual(variants[2]['env']['ALEA_BENCH_BACKEND'],'enzyme')
            self.assertEqual(variants[0]['build_receipt'],'/saved/qualification.json')

    def test_pairwise_feature_ratios_use_matched_repetitions_only(self):
        plan=dict(cases=['mean'],variants=[dict(name=n) for n in ['stan','a','b']],reps=2)
        rows=[dict(case='mean',variant=n,rep=r,admitted=True,
                   metrics=dict(evaluation_seconds=t,evaluations=10))
              for n,r,t in [('stan',0,8.),('a',0,2.),('b',0,4.),('b',1,.01)]]
        result=bench.matrix_summary(plan,rows)
        self.assertEqual(result[1]['pairwise_reference_over_variant']['b']['median'],2.)

    def test_suite_uses_existing_workers_and_reports_source_inference_as_unsupported(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'results';builds=root/'variants.json'
            builds.write_text(json.dumps([dict(name='enzyme',features=['enzyme'],binary=str(root/'enzyme'))]))
            args=SimpleNamespace(output=out,builds=[builds],stan_evaluate=root/'stan',
                cases=['covariance'],reps=7,evaluations=10000,cpu=None,
                phases=['sampling'],cmdstan=root/'cmdstan',cxx=None,job_timeout=10)
            with patch.object(bench,'child',return_value=dict(returncode=0)), \
                 patch.object(bench.os,'sched_getaffinity',return_value={8}), \
                 patch.object(bench,'suite_report') as report:
                bench.suite_run(args)
            progress=json.loads((out/'progress.json').read_text())
            self.assertEqual(len(progress),4)
            self.assertEqual(progress[-1]['status'],'unsupported')
            self.assertTrue(all(row.get('status')=='preflight_failed' for row in progress[1:3]))
            report.assert_called_once_with(out)

    def test_stan_build_aliases_share_the_same_model_executable(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'build';commands=[]
            main=root/'cmdstan/src/cmdstan/main.cpp';main.parent.mkdir(parents=True);main.write_text('int main(){}')
            args=SimpleNamespace(output=out,cmdstan=root/'cmdstan',cxx='/usr/bin/g++',
                                 cases=['linear','linear_large'],native_cpu=False)
            def child(command,*unused,**kwargs):
                commands.append(list(map(str,command)))
                if '-MMD' in command:(out/'stan-evaluate.d').write_text('worker: ')
                return dict(returncode=0)
            with patch.object(bench,'child',side_effect=child),patch.object(bench,'command_output',return_value='compiler'):
                bench.matrix_stan_build(args)
            self.assertEqual(len(commands),2)
            self.assertEqual(commands[1][-1],str(out/'linear'))
            self.assertIn(f'CMDSTAN_MAIN={out}/cmdstan_main.cpp',commands[1])
            self.assertEqual((out/'cmdstan_main.cpp').read_text(),main.read_text())

    def test_profile_requires_an_explicit_local_exception_without_compute_allocation(self):
        args=SimpleNamespace(local_profile=False)
        with patch.dict(bench.os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,'compute allocation'):bench.profile_run(args)

    def test_profile_forwards_evaluation_mode_after_the_optional_privilege_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);plan=root/'plan.json'
            plan.write_text(json.dumps(dict(cases=['funnel'],variants=[dict(name='native',binary='/saved/alea',env={})])))
            args=SimpleNamespace(local_profile=True,matrix_plan=plan,cpu=None,
                perf_command='sudo -n perf',evaluations=10,output=root/'profiles',
                cases=['funnel'],profile_variants=None)
            commands=[]
            def child(command,*unused,**kwargs):
                commands.append(list(map(str,command)))
                return dict(returncode=0)
            with patch.object(bench,'child',side_effect=child),patch.object(bench,'host_metadata',return_value={}),patch.object(bench.os,'sched_getaffinity',return_value={3}):
                bench.profile_run(args)
            self.assertEqual(len(commands),3)
            for command in commands[:2]:
                target=command[command.index('--')+1:]
                self.assertEqual(target[:4],['taskset','-c','3','env'])
                self.assertIn('ALEA_BENCH_EVALUATIONS=10',target)
            self.assertIn('not comparative timing',json.loads((args.output/'profile.json').read_text())['scope'])
            args.output=root/'failed-profiles'
            with patch.object(bench,'child',return_value=dict(returncode=1)),patch.object(bench,'host_metadata',return_value={}),patch.object(bench.os,'sched_getaffinity',return_value={3}):
                with self.assertRaisesRegex(ValueError,'profiling incomplete'):bench.profile_run(args)
            self.assertEqual(json.loads((args.output/'profile.json').read_text())['rows'][0]['record']['returncode'],1)

    def test_per_variant_success_cannot_bypass_the_build_freeze_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);binary=root/'default';binary.write_bytes(b'compiled')
            source=root/'source.json';source.write_text('{}')
            variants=root/'variants.json';variants.write_text('[]')
            variant=dict(name='default',binary=str(binary),source_sha256=bench.sha(source),
                         build_receipt=str(root/'qualification.json'),build_returncode=0)
            self.assertIn('unqualified build',bench.build_admission(variant))
            for status in ['pending','source_changed']:
                (root/'qualification.json').write_text(json.dumps(dict(status=status)))
                self.assertIn('incomplete',bench.build_admission(variant))
            receipt=dict(status='complete',source_sha256=bench.sha(source),variants_sha256=bench.sha(variants),
                         variants={'default':dict(returncode=0,binary_sha256=bench.sha(binary))},files={})
            (root/'qualification.json').write_text(json.dumps(receipt))
            self.assertIsNone(bench.build_admission(variant))
            binary.write_bytes(b'changed')
            self.assertIn('changed',bench.build_admission(variant))

    def test_cohort_admission_hashes_each_file_once_and_does_not_cache_across_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source=root/'source.json';source.write_text('{}')
            listing=root/'variants.json';listing.write_text('[]')
            binary=root/'default';binary.write_bytes(b'compiled')
            receipt=dict(status='complete',source_sha256=bench.sha(source),variants_sha256=bench.sha(listing),
                variants={'default':dict(returncode=0,binary_sha256=bench.sha(binary))},
                files={p.name:bench.sha(p) for p in [source,listing,binary]})
            (root/'qualification.json').write_text(json.dumps(receipt))
            variants=[dict(name=str(i),binary=str(binary),source_sha256=receipt['source_sha256'],
                           build_receipt=str(root/'qualification.json')) for i in range(8)]
            with patch.object(bench,'sha',wraps=bench.sha) as digest:
                self.assertEqual(set(bench.build_admissions(variants).values()),{None})
                self.assertEqual(digest.call_count,3)
            binary.write_bytes(b'changed')
            self.assertTrue(all('changed' in reason for reason in bench.build_admissions(variants).values()))

    def test_matrix_input_discovery_reads_each_receipt_once_before_hashing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);binary=root/'binary';binary.write_bytes(b'compiled')
            receipt=root/'qualification.json';receipt.write_text(json.dumps(dict(files={'binary':'digest'})))
            variants=[dict(name=str(i),binary=str(binary),build_receipt=str(receipt)) for i in range(8)]
            with patch.object(bench,'read_metrics',wraps=bench.read_metrics) as read, patch.object(bench,'sha') as digest:
                paths=bench.matrix_input_paths(dict(variants=variants,cases=[]))
                self.assertEqual(paths,{binary,receipt})
                self.assertEqual(read.call_count,1)
                digest.assert_not_called()

    def test_feature_replay_uses_raw_preflight_and_loop_outputs_not_cached_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            plan=dict(cases=['mean'],variants=[dict(name='stan'),dict(name='alea')],reps=2,evaluations=100)
            (out/'manifest.json').write_text(json.dumps(dict(plan=plan,files={})))
            probes=[dict(logp=-3.,gradient=[1.,2.]) for _ in range(8)]
            for i,delta in enumerate([1e-5,-1e-5,2e-5,-2e-5],start=4): probes[i]['logp']+=delta
            def raw(folder,value):
                p=out/folder;p.mkdir(parents=True,exist_ok=True)
                (p/'process.json').write_text('{"returncode":0}')
                (p/'stdout.txt').write_text(json.dumps(value))
            for i,value in enumerate(probes):raw(f'preflight/mean-stan-{i}',value)
            raw('preflight/mean-alea',probes)
            receipt=dict(evaluations=10,logp=-2.,checksum=-20.,gradient_checksum=50.,gradient=[1.,2.])
            raw('preflight/mean-stan-loop',receipt)
            actual=dict(receipt,evaluations=100,evaluation_seconds=.2,checksum=-200.,gradient_checksum=500.)
            for name in ['stan','alea']:raw(f'mean-0-{name}',actual)
            rows=[dict(case='mean',variant=name,rep=0,admitted=False,metrics={'evaluation_seconds':1e-30}) for name in ['stan','alea']]
            (out/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result=bench.matrix_report(out)
            self.assertEqual(result[1]['admitted'],1)
            self.assertEqual(result[1]['unstarted'],1)
            self.assertEqual(result[1]['seconds']['median'],.002)
            raw('mean-0-alea',dict(actual,gradient=[1.,3.]))
            self.assertEqual(bench.matrix_report(out)[1]['admitted'],0)
            (out/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in [*rows,rows[0]]))
            with self.assertRaisesRegex(ValueError,'duplicate'):bench.matrix_report(out)

    def test_feature_receipts_require_full_gradient_and_changing_state_work(self):
        reference=dict(evaluations=10,logp=-2.,checksum=-20.,gradient_checksum=30.,gradient=[1.,2.])
        actual=dict(reference,evaluations=100,evaluation_seconds=.1,checksum=-200.,gradient_checksum=300.)
        self.assertTrue(bench.matrix_receipt(actual,reference,100,2))
        for key,value in [('gradient',[1.]),('gradient',[1.,3.]),('checksum',-199.),
                          ('gradient_checksum',301.),('evaluation_seconds',float('inf')),('evaluations',99)]:
            self.assertFalse(bench.matrix_receipt(dict(actual,**{key:value}),reference,100,2))

    def test_feature_summary_preserves_failed_and_unstarted_jobs_and_withholds_ratios(self):
        plan=dict(cases=['covariance'],variants=[dict(name='stan'),dict(name='enzyme')],reps=3)
        m=dict(evaluation_seconds=.1,evaluations=100)
        rows=[dict(case='covariance',variant='stan',rep=0,admitted=True,metrics=m),
              dict(case='covariance',variant='enzyme',rep=0,admitted=False,reason='unsupported')]
        result=bench.matrix_summary(plan,rows)
        self.assertEqual(result[1]['unstarted'],2)
        self.assertEqual(result[1]['admitted'],0)
        self.assertIsNone(result[1]['stan_over_variant'])
        self.assertEqual(result[1]['rejected'],['unsupported'])

    def test_child_uses_blocking_wait_not_quantized_timeout_polling(self):
        with tempfile.TemporaryDirectory() as temp:
            process=MagicMock();process.wait.return_value=0
            with patch.object(bench.subprocess,'Popen',return_value=process), patch.object(bench.threading,'Timer') as timer:
                row=bench.child(['worker'],Path(temp)/'run',{},timeout=1)
            process.wait.assert_called_once_with()
            timer.return_value.start.assert_called_once_with()
            timer.return_value.cancel.assert_called_once_with()
            timer.return_value.join.assert_called_once_with()
            self.assertEqual(row['returncode'],0)
            self.assertIsNone(row['error'])

    def test_watchdog_timeout_is_retained_as_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            process=MagicMock();process.poll.return_value=None
            def wait():
                timer.call_args.args[1]()
                return -9
            process.wait.side_effect=wait
            with patch.object(bench.subprocess,'Popen',return_value=process), patch.object(bench.threading,'Timer') as timer:
                row=bench.child(['worker'],Path(temp)/'run',{},timeout=1)
            process.kill.assert_called_once_with()
            self.assertEqual(row['returncode'],124)
            self.assertIn('exceeded',row['error'])

    def test_build_reuse_requires_all_compiled_inputs_not_just_head(self):
        before={'a.rs':'one','Cargo.lock':'two','b.cpp':'three','model.stan':'four','worker.py':'old'}
        after={**before,'worker.py':'new'}
        self.assertEqual(bench.compiled_inputs(before),bench.compiled_inputs(after))
        for name in ('a.rs','Cargo.lock','b.cpp','model.stan'):
            changed={**after,name:'different'}
            self.assertNotEqual(bench.compiled_inputs(before),bench.compiled_inputs(changed))
    def test_all_scores_and_work_are_required_for_fixed_work_admission(self):
        reference=dict(evaluations=100,logp=-2.,checksum=-200.,gradient_checksum=3.,gradient=[1.,2.])
        self.assertTrue(bench.kernel_compatible(reference,reference))
        for key,value in [('evaluations',99),('logp',0.),('checksum',float('nan')),
                          ('gradient_checksum',0.),('gradient',[1.]),('gradient',[1.,3.])]:
            changed=copy.deepcopy(reference);changed[key]=value
            self.assertFalse(bench.kernel_compatible(reference,changed),key)

    def test_replay_never_trusts_cached_success_or_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); folder=root/'run';folder.mkdir()
            row=dict(folder='run',engine='alea',phase='evaluate',case='mean',returncode=0,
                     success=True,metrics={'evaluations':100,'evaluation_seconds':.1})
            (folder/'stdout.txt').write_text('not JSON\n')
            self.assertFalse(bench.rederive(root,row)['success'])
            (folder/'stdout.txt').write_text('{"evaluations":100,"evaluation_seconds":0.1}\n')
            row['success']=False;row['metrics']={}
            self.assertTrue(bench.rederive(root,row)['success'])
            row['returncode']=1
            self.assertFalse(bench.rederive(root,row)['success'])

    def test_specialized_stan_glm_is_not_silently_substituted_for_original(self):
        self.assertEqual(bench.engines('mean'),['alea','stan','jax'])
        self.assertEqual(bench.engines('linear'),['alea','stan','stan_glm','jax'])
        self.assertEqual(bench.engines('logistic_large'),['alea','stan','stan_glm','jax'])


if __name__=='__main__': unittest.main()
