import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import three_engine as bench


class ThreeEngineTests(unittest.TestCase):
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
