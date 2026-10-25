"""Native lanes preserve coverage, failures, coordinate identity and raw evidence."""
import copy
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alea_bench import pdb, schema
from alea_bench.manifest import ROOT
from alea_bench.native_conformance import points, relative
from alea_bench.native_ports import generate, rust_port


class NativeTests(unittest.TestCase):
    def test_gradient_profile_and_trace_scope_are_sealed_without_changing_old_jobs(self):
        protocol = schema.load(ROOT/'tools/bench/protocols/posteriordb-native-pilot-v3.json')
        protocol.update(cases=['Rate_2_data-Rate_2_model'], methods=['nuts_grad_diag','nuts_rs'])
        case = dict(id=protocol['cases'][0],family='binomial')
        model = dict(metadata=dict(dimension=2),builds={})
        manifest = dict(protocol=protocol,corpus={'cases':[case]},preparation=dict(models={case['id']:model}))
        pdb.validate_protocol(protocol)
        self.assertTrue(all('record_warmup_trace' not in j for j in pdb.jobs(manifest)))
        protocol['record_warmup_trace'] = False
        pdb.validate_protocol(protocol)
        self.assertTrue(all(j['record_warmup_trace'] is False for j in pdb.jobs(manifest)))
        protocol['record_warmup_trace'] = 0
        with self.assertRaisesRegex(schema.Invalid,'record_warmup_trace must be boolean'):
            pdb.validate_protocol(protocol)

    def test_two_chain_pilot_uses_exactly_two_prepared_starts_for_every_engine(self):
        starts=[[.1,.2],[.3,.4],[.5,.6],[.7,.8]]
        metadata={'initials':copy.deepcopy(starts)}
        selected=pdb.select_initials(metadata,2)
        self.assertEqual(selected,starts[:2])
        self.assertEqual(len(selected),2)
        selected[0][0]=99.
        self.assertEqual(metadata['initials'],starts)
        with self.assertRaisesRegex(schema.Invalid,'not enough sealed initial points'):
            pdb.select_initials(metadata,5)

    def test_failed_native_job_retains_declared_backend_without_a_false_stan_fallback(self):
        from test_pdb import fixture
        from alea_bench.pdb_report import summarize
        with tempfile.TemporaryDirectory() as tmp:
            m,row,_=fixture(Path(tmp))
            m['preparation']['native']={'rust_backend':'enzyme'}
            row.update(status='failed',worker=None,metrics=None,process=None,failure={'kind':'Invalid','message':'admission'})
            report,_=summarize(m,[row],{'example':{}})
            result=report['rows'][0]
            self.assertIn('Enzyme',result['backend'])
            self.assertIn('Enzyme',result['planned_backend'])
            self.assertIsNone(result['observed_backend'])


    def test_every_declared_model_has_source_and_stan_data_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'ports'; records = generate(folder)
            self.assertEqual(len(records), 20)
            self.assertEqual(len({r['case'] for r in records}), 20)
            for row in records:
                self.assertGreater(row['dimension'], 0)
                self.assertIn('fn norm(', (folder/row['primal']).read_text())
                self.assertEqual(len(row['data_sha256']), 64)
                self.assertEqual(len(row['stan_sha256']), 64)
            with self.assertRaises(FileExistsError): generate(folder)

    def test_unknown_port_never_falls_back(self):
        with self.assertRaisesRegex(ValueError, 'no native Rust translation'):
            rust_port('other-unknown', {})

    def test_protocols_are_descriptive_and_baseline_parallel_has_no_cohort_sampler(self):
        for path in (ROOT/'tools/bench/protocols').glob('posteriordb-native-*-v3.json'):
            value = pdb.validate_protocol(schema.load(path))
            self.assertEqual(value['purpose'], 'descriptive')
            self.assertTrue(value['report_work_metrics'])
            self.assertNotIn('promotion', value)
            if 'parallel' in path.name: self.assertNotIn('chees', value['methods'])

    def test_failed_conformance_is_an_explicit_unsupported_native_job(self):
        protocol = schema.load(ROOT/'tools/bench/protocols/posteriordb-native-pilot-v3.json')
        protocol.update(cases=['Rate_2_data-Rate_2_model'], methods=['nuts_cov_diag','nuts_rs','blackjax_nuts','stan_nuts_diag'])
        model = dict(metadata=dict(dimension=2), builds={'stan':{'returncode':0}})
        manifest = dict(protocol=protocol, corpus={'cases':[dict(id=protocol['cases'][0],family='binomial')]},
                        preparation=dict(models={protocol['cases'][0]:model}, native=dict(rust_backend='enzyme',execution='serial',conformance={})))
        jobs = pdb.jobs(manifest)
        for job in jobs:
            self.assertEqual(job['backend_lane'], 'enzyme')
            self.assertEqual(job['unsupported_reason'] is None, job['method']=='stan_nuts_diag')
        manifest['preparation']['native']['conformance'][protocol['cases'][0]] = {'status':'validated'}
        self.assertTrue(all(j['unsupported_reason'] is None for j in pdb.jobs(manifest)))

    def test_native_hvp_preflight_replaces_old_backend_limit(self):
        protocol = schema.load(ROOT/'tools/bench/protocols/posteriordb-native-pilot-v3.json')
        protocol.update(cases=['Rate_2_data-Rate_2_model'], methods=['atlas_full','rmhmc_softabs'])
        case = dict(id=protocol['cases'][0],family='binomial')
        model = dict(metadata=dict(dimension=2,derivative_validation={'hvp_error':'old Stan backend limitation'}),builds={})
        manifest = dict(protocol=protocol,corpus={'cases':[case]},preparation=dict(models={case['id']:model},native=dict(rust_backend='enzyme',execution='serial',conformance={case['id']:{'status':'validated'}})))
        self.assertTrue(all(j['unsupported_reason'] is None for j in pdb.jobs(manifest)))

    def test_conformance_points_are_deterministic_and_include_shared_starts(self):
        starts = [[.1,.2],[-.2,.3]]
        self.assertEqual(points(starts), points(starts))
        self.assertEqual(points(starts)[:2], starts)
        self.assertEqual(len(points(starts)), 7)
        self.assertGreater(relative([float('nan')],[1.]), 1.)
        self.assertGreater(relative([1.,2.],[1.]), 1.)

    @unittest.skipUnless(hasattr(os, 'pidfd_open'), 'Linux pidfd execution')
    def test_parallel_execution_merges_logical_chain_ids_and_seals_child_artifacts(self):
        from alea_bench.native_parallel import run
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            worker = directory/'fake.py'
            worker.write_text('''import csv,json,sys
from pathlib import Path
j=json.loads(Path(sys.argv[1]).read_text()); p=Path(j['directory'])
for name,header in [('draws.csv',['chain','draw','q0']),('constrained.csv',['chain','draw','theta'])]:
 with (p/name).open('w') as out:
  w=csv.writer(out);w.writerow(header)
  for i in range(j['draws']):w.writerow([0,i,j['seeds'][0]])
(p/'worker.json').write_text(json.dumps(dict(backend='fixture',work={'gradient_api_calls':7},phases=[{'chain':0,'retained_seconds':.1}])))
''')
            cpu = min(os.sched_getaffinity(0))
            job = dict(directory=str(directory), seeds=[101,102], initials=[[0.],[0.]], draws=4,
                       parallel_cpus=[cpu,cpu], parallel_command=[sys.executable,str(worker)],timeout_seconds=10)
            run(job)
            with (directory/'draws.csv').open() as source:
                rows = list(csv.DictReader(source))
            self.assertEqual([r['chain'] for r in rows], ['0']*4+['1']*4)
            self.assertEqual([r['q0'] for r in rows], ['101']*4+['102']*4)
            record = json.loads((directory/'worker.json').read_text())
            self.assertEqual(record['work']['gradient_api_calls'], 14)
            self.assertEqual([p['chain'] for p in record['phases']], [0,1])
            self.assertTrue((directory/'chain-artifacts.tar.gz').is_file())

    @unittest.skipUnless(hasattr(os, 'pidfd_open'), 'Linux pidfd execution')
    def test_parallel_deadline_kills_the_whole_cohort(self):
        from alea_bench.process import isolated
        from alea_bench.runner import environment
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);worker=directory/'sleep.py'
            worker.write_text("import os,sys,json,time; from pathlib import Path; j=json.loads(Path(sys.argv[1]).read_text()); Path(j['directory'],'pid').write_text(str(os.getpid())); time.sleep(30)\n")
            cpu=min(os.sched_getaffinity(0))
            job=dict(directory=str(directory),seeds=[101,102],initials=[[0.],[0.]],draws=4,
                     parallel_cpus=[cpu,cpu],parallel_command=[sys.executable,str(worker)],timeout_seconds=1)
            path=directory/'job.json';path.write_text(json.dumps(job))
            with (directory/'process.log').open('w') as log:
                result=isolated([sys.executable,'-m','alea_bench.native_parallel',str(path)],
                    cwd=ROOT,env=environment(ROOT),log=log,timeout=1.)
            self.assertTrue(result['timed_out'])
            pids=list(directory.glob('chain-*/pid'));self.assertEqual(len(pids),2)
            for p in pids:
                stat=Path('/proc')/p.read_text()/'stat'
                def state():
                    try:return stat.read_text().split()[2]
                    except FileNotFoundError:return None
                # The session leader can become pidfd-ready before another
                # CPU schedules its children to consume the pending SIGKILL.
                deadline=time.monotonic()+2.
                while state() not in (None,'Z') and time.monotonic()<deadline:time.sleep(.01)
                self.assertIn(state(),(None,'Z'),'parallel worker escaped its measured process group')

    def test_comparison_keeps_failed_pairs_and_ratio_direction(self):
        from alea_bench.native_report import comparison
        row=dict(case='case',method='nuts_cov_diag',repetition=0,status='completed',backend='Enzyme',
                 diagnostics={'maximum_rhat':1.},divergences=0,process_seconds=3.,warmup_seconds=1.,retained_seconds=2.,
                 minimum_bulk_ess_per_process_second=10.,minimum_bulk_ess_per_retained_second=15.)
        other=dict(row,method='nuts_rs',process_seconds=2.)
        pairs=comparison({'rows':[row,other]})
        self.assertEqual(pairs[0]['alea_over_competitor_process_seconds'],1.5)
        other['status']='failed';pairs=comparison({'rows':[row,other]})
        self.assertIsNone(pairs[0]['alea_over_competitor_process_seconds'])
        self.assertIsNotNone(pairs[0]['undefined_reason'])


if __name__ == '__main__': unittest.main()
