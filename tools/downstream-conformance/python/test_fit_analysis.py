import copy
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

import fit_analysis


class FitAnalysisTests(unittest.TestCase):
    def fixture(self):
        return {"format":"alea-fit", "schema_version":1, "kind":"point_estimate",
                "provenance":{"model":"m", "data":"d", "source":"s", "method":"L-BFGS", "settings":{"seed":str(2**64-1)}},
                "objective":{"kind":"likelihood", "measure":"Lebesgue"},
                "parameters":[{"name":'σ,"λ"\n', "dimension":None, "coordinate":None, "block":None}],
                "unconstrained_dimension":1,
                "numerical":{"algorithm":"lbfgs_armijo", "termination":"score_tolerance", "line_search_failure":None,
                             "options":{"iteration_limit":10, "evaluation_limit":2**64-1, "history_limit":3,
                                        "line_search_limit":10, "score_tolerance":1e-8, "scales":[2.]},
                             "iterations":0, "evaluations":2**53+1, "domain_trials":2**64-1, "skipped_updates":0},
                "candidate":{"free_position":[-0.], "free_gradient":[math.ulp(0.)],
                             "value":float.fromhex("0x1.fffffffffffffp+1023"), "constrained_estimates":[1.]},
                "assessment":{"existence":"unresolved", "budget":2**64-1}}

    def test_exact_numbers_names_and_opaque_evidence_no_posterior_axes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"fit.json"
            path.write_text(json.dumps(self.fixture()),encoding="utf-8")
            r=fit_analysis.load(path)
        self.assertEqual(r,self.fixture())
        self.assertEqual(struct.pack(">d",r["candidate"]["free_position"][0]),struct.pack(">d",-0.))
        self.assertEqual(r["numerical"]["evaluations"],2**53+1)
        self.assertFalse(set(r)&{"chains","draws","covariance","standard_errors"})

    def test_all_stopping_reasons_and_no_candidate_are_retained(self):
        for term in ("score_tolerance","iteration_limit","evaluation_limit","cancelled","line_search_failed","numerical_direction","evaluation_failed"):
            r=self.fixture();r["numerical"]["termination"]=term
            if term=="iteration_limit":r["numerical"]["iterations"]=10
            if term=="evaluation_limit":r["numerical"]["evaluations"]=2**64-1
            if term=="line_search_failed":r["numerical"]["line_search_failure"]="no_representable_step"
            self.assertIs(fit_analysis.validate(r),r)
        for term in ("cancelled","evaluation_failed"):
            r=self.fixture();r["numerical"]["termination"]=term;r["candidate"]=None
            r["numerical"].update(evaluations=int(term=="evaluation_failed"),domain_trials=0)
            self.assertIsNone(fit_analysis.validate(r)["candidate"])

    def test_malformed_and_unsupported_records_are_not_coerced(self):
        mutations=[lambda r:r.update(schema_version=True),lambda r:r.update(schema_version=2),
                   lambda r:r.update(kind="sampling"),lambda r:r.update(chains=[]),
                   lambda r:r["parameters"].append(copy.deepcopy(r["parameters"][0])),
                   lambda r:r["numerical"].update(evaluations=True),
                   lambda r:r["numerical"].update(iterations=11),
                   lambda r:r["numerical"].update(line_search_failure="backtrack_limit"),
                   lambda r:r["candidate"].update(free_gradient=[1.]),
                   lambda r:r["candidate"].update(free_position=[float("nan")]),
                   lambda r:r["candidate"].update(value=True),
                   lambda r:r["candidate"].update(constrained_estimates=[]),
                   lambda r:r["numerical"]["options"].update(scales=[]),
                   lambda r:r["numerical"]["options"].update(unknown=1),
                   lambda r:r["candidate"].update(free_gradient=[10**300]),
                   lambda r:r["assessment"].update(value=float("inf")),
                   lambda r:r.update(candidate=None)]
        for i,mutate in enumerate(mutations):
            with self.subTest(case=i):
                r=self.fixture();mutate(r)
                with self.assertRaises(ValueError):fit_analysis.validate(r)

    def test_duplicate_fields_truncation_and_overflow_json_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"fit.json"
            encoded=json.dumps(self.fixture())
            for data in (encoded[:-1],encoded.replace('"schema_version": 1','"schema_version": 1, "schema_version": 1'),
                         encoded.replace('"unresolved"','1e999')):
                path.write_text(data,encoding="utf-8")
                with self.assertRaises(ValueError):fit_analysis.load(path)


if __name__=="__main__":unittest.main()
