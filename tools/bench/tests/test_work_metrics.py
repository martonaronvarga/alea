import copy,json,unittest
from pathlib import Path
from alea_bench import schema
from alea_bench.pdb import validate_protocol
from alea_bench.pdb_report import work_metrics
ROOT=Path(__file__).resolve().parents[3]
class WorkMetrics(unittest.TestCase):
 def test_actual_total_and_retained_calls_have_separate_ratios(self):
  worker={'work':{'gradient_api_calls':120,'hvp_api_calls':0},'phases':[
   {'work_before_retained':{'gradient_api_calls':20},'work_after':{'gradient_api_calls':50}},
   {'work_before_retained':{'gradient_api_calls':90},'work_after':{'gradient_api_calls':120}}]}
  r=work_metrics(worker,240,True)
  self.assertEqual(r['minimum_bulk_ess_per_gradient_call'],2)
  self.assertEqual(r['minimum_bulk_ess_per_retained_gradient_call'],4)
  self.assertIsNone(r['minimum_bulk_ess_per_hvp_call']);self.assertEqual(r['hvp_call_ratio_unavailable_reason'],'no calls of this capability')
  self.assertIsNone(r['total_third_ad_api_calls']);self.assertEqual(r['third_ad_call_ratio_unavailable_reason'],'API count unavailable')
 def test_failed_and_undefined_jobs_never_get_fabricated_efficiency(self):
  for complete,ess in [(False,100),(True,None)]:
   r=work_metrics({'work':{'gradient_api_calls':4}},ess,complete)
   self.assertIsNone(r['minimum_bulk_ess_per_gradient_call'])
  r=work_metrics({'phases':[{'gradient_api_calls':3},{'gradient_api_calls':7}]},20,True)
  self.assertEqual(r['minimum_bulk_ess_per_gradient_call'],2)
  self.assertIsNone(r['retained_gradient_api_calls'])
 def test_work_metrics_are_explicitly_opted_in_and_old_protocols_stay_valid(self):
  old=json.loads((ROOT/'tools/bench/protocols/posteriordb-v3.json').read_text());validate_protocol(old)
  new=copy.deepcopy(old);new['report_work_metrics']=True;validate_protocol(new)
  new['report_work_metrics']=1
  with self.assertRaises(schema.Invalid):validate_protocol(new)
