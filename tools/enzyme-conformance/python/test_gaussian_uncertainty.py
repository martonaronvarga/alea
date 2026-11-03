import copy
import json
import math
import unittest
import gaussian_uncertainty as checker


def record(case="base"):
    xs, ys = checker.CASES[case]
    offsets = [(i+1)*.25*(-1 if i % 2 else 1)*ys for i in range(6)]
    response = [(1.5-.75*x+e)*ys+o for x, e, o in zip(range(-2, 4), (1,-2,1,1,-2,1), offsets)]
    covariance = [38/105, -6/(105*xs), -6/(105*xs), 12/(105*xs*xs)]
    covariance = [v*ys*ys for v in covariance]
    intervals=[]
    for level,critical in checker.CRITICAL_DF4.items():
        intervals.append([level,critical,*[bound for beta,var in zip((1.5*ys,-.75*ys/xs),(covariance[0]*1.5,covariance[3]*1.5)) for bound in (beta-critical*math.sqrt(var),beta+critical*math.sqrt(var))]])
    return {"case": case, "rows": 6, "columns": 2, "rank": 2, "df": 4,
            "design": [v for x in range(-2, 4) for v in (1., x*xs)],
            "response": response, "offsets": offsets,
            "batch_estimates": [1.5*ys, -.75*ys/xs, math.sqrt(2)*ys],
            "generated_estimates": [1.5*ys, -.75*ys/xs, math.sqrt(2)*ys],
            "svd_coefficients": [1.5*ys, -.75*ys/xs], "mle_variance": 2*ys*ys,
            "residual_variance": 3*ys*ys, "coefficient_covariance": covariance,
            "joint_covariance": [covariance[0], covariance[1], 0., covariance[2], covariance[3], 0., 0., 0., ys*ys/6],
            "corrected_covariance": [v*1.5 for v in covariance], "intervals":intervals}


class Qualification(unittest.TestCase):
    def test_exact_fixtures_and_complete_cohort(self):
        records = checker.read("\n".join(json.dumps(record(k)) for k in checker.CASES))
        for item in records:
            checker.check(item)

    def test_wrong_divisor_covariance_sign_projection_or_sigma_rejects(self):
        for key in ("mle_variance", "residual_variance", "coefficient_covariance", "corrected_covariance", "joint_covariance"):
            r = record()
            if isinstance(r[key], list):
                r[key][1] = -r[key][1] if key != "joint_covariance" else .1
            else:
                r[key] *= 1.5
            with self.subTest(key=key), self.assertRaises(ValueError):
                checker.check(r)
        r = record()
        r["joint_covariance"][8] *= 1.5
        with self.assertRaises(ValueError):
            checker.check(r)

    def test_shape_nonfinite_boolean_metadata_and_data_mutations_reject(self):
        for key, value in (("rows", True), ("df", 6), ("rank", 1), ("columns", 3), ("extra", 0),
                           ("batch_estimates", [1., 2.]), ("joint_covariance", [math.nan]*9),
                           ("design", [1.]*12), ("offsets", [0.]*6), ("response", [0.]*6)):
            r = copy.deepcopy(record())
            r[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                checker.check(r)

    def test_missing_duplicate_truncated_and_unknown_cases_reject(self):
        valid = [record(k) for k in checker.CASES]
        for records in (valid[:2], [valid[0]]*3, [*valid, valid[0]]):
            with self.assertRaises(ValueError):
                checker.read("\n".join(json.dumps(r) for r in records))
        with self.assertRaises(ValueError):
            checker.read('{"case":"base","case":"small"}')
        with self.assertRaises(ValueError):
            checker.read('{"case":')

    def test_interval_level_critical_value_endpoint_and_missing_level_reject(self):
        for index,value in ((0,.8),(1,1.959963984540054),(2,999.),(3,math.inf),(4,True)):
            r=record(); r["intervals"][0][index]=value
            with self.subTest(index=index),self.assertRaises(ValueError): checker.check(r)
        r=record(); r["intervals"].pop()
        with self.assertRaises(ValueError): checker.check(r)


if __name__ == "__main__":
    unittest.main()
