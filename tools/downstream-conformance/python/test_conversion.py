"""Optional, executed external conversions and independent predictive oracles."""
import copy
import json
import os
from pathlib import Path
import subprocess
import unittest
import warnings
import analysis
import test_analysis as fixtures
from test_analysis import bound_fixture

EXTERNAL = os.environ.get("ALEA_ANALYSIS_EXTERNAL") == "1"
if EXTERNAL:
    import numpy as np
    import xarray as xr
    from scipy.stats import norm
    import conversion


@unittest.skipUnless(EXTERNAL, "run in the locked analysis shell with ALEA_ANALYSIS_EXTERNAL=1")
class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.base = fixtures.AnalysisTests()
        self.base.setUp()
        self.addCleanup(self.base.temp.cleanup)

    def load(self, manifest, rows):
        return analysis.load(*self.base.files(manifest, rows))

    def test_datatree_netcdf_preserves_bits_dtypes_units_and_predictive_meaning(self):
        m, rows = bound_fixture(self.base)
        a = self.load(m, rows)
        tree = conversion.to_datatree(a)
        self.assertNotIn("energy", tree["sample_stats"].data_vars)
        self.assertIn("initial_energy", tree["sample_stats"].data_vars)
        self.assertNotIn("n_steps", tree["sample_stats"].data_vars)
        self.assertIn("integration_steps", tree["sample_stats"].data_vars)
        self.assertIn("future", tree["generated_quantities"].data_vars)
        self.assertNotIn("future", tree["posterior_predictive"].data_vars)
        self.assertEqual(tree["posterior_predictive"]["y"].shape, (1, 2, 2))
        name = m["parameters"][0]["name"]
        self.assertEqual(tree["posterior"][name].attrs["alea_units"], "seconds")
        path = self.base.root / "fit.nc"
        conversion.write_netcdf(tree, path)
        with xr.open_datatree(path, engine="h5netcdf") as restored:
            before = tree["posterior"][name].values
            after = restored["posterior"][name].values
            self.assertEqual(after.dtype, np.dtype("float64"))
            np.testing.assert_array_equal(after.view(np.uint64), before.view(np.uint64))
            self.assertEqual(restored["constant_data"]["unsigned"].item(), 2**64-1)
            self.assertEqual(restored["constant_data"]["signed"].item(), -(2**63))
            self.assertEqual(restored["constant_data"]["label"].item(), 'λ,"\r\nlabel')
            self.assertEqual(restored["sample_stats"]["diverging"].dtype, np.dtype("bool"))
            np.testing.assert_array_equal(restored["posterior_predictive"]["y"], tree["posterior_predictive"]["y"])
        with self.assertRaises(ValueError): conversion.write_netcdf(tree, path)

    def test_mixed_statistics_are_not_filled_or_misrepresented(self):
        m, rows = bound_fixture(self.base)
        rows[0][m["csv"]["columns"].index("stat:energy")] = ""
        a = self.load(m, rows)
        with self.assertRaisesRegex(ValueError, "mixed statistic"):
            conversion.to_datatree(a)
        self.assertEqual(len(a.records), 2)

    def test_explicit_subset_retains_all_failure_messages_and_warmup_identity(self):
        m, rows = bound_fixture(self.base)
        other = copy.deepcopy(m["chains"][0]); other["id"] = 2**64-1
        m["chains"].append(other)
        second = copy.deepcopy(rows)
        for row in second: row[0] = str(2**64-1)
        for chain in m["chains"]:
            chain["warmup"] = {"first": 40, "last": 40, "rows": 1, "failures": 0}
        warmup = []
        for id_ in (9, 2**64-1):
            row = rows[0].copy(); row[:3] = [str(id_), "warmup", "40"]; warmup.append(row)
        failed = copy.deepcopy(other); failed.update(id=3, status="failed", reason="reporting error")
        failed["warmup"] = {"first": None, "last": None, "rows": 0, "failures": 0}
        failed["sampling"] = {"first": 0, "last": 0, "rows": 1, "failures": 1}
        m["chains"].append(failed)
        failure = ["3", "sampling", "0", "failure", 'error,"\r\ndetail'] + [""]*(len(rows[0])-5)
        a = self.load(m, warmup + rows + second + [failure])
        with self.assertRaises(ValueError): conversion.to_datatree(a)
        tree = conversion.to_datatree(a, chains=[9, 2**64-1])
        self.assertEqual(tree["posterior"].coords["chain"].values.tolist(), [9, 2**64-1])
        self.assertEqual(tree["warmup_posterior"].coords["draw"].values.tolist(), [40])
        self.assertEqual(tree["analysis_failures"]["message"].item(), 'error,"\r\ndetail')
        self.assertEqual(len(json.loads(tree.attrs["alea_manifest_json"])["chains"]), 3)
        other["warmup"] = copy.deepcopy(failed["warmup"])
        a = self.load(m, warmup[:1] + rows + second + [failure])
        with self.assertRaises(ValueError): conversion.to_datatree(a, chains=[9, 2**64-1])
        tree = conversion.to_datatree(a, chains=[9, 2**64-1], include_warmup=False)
        self.assertNotIn("warmup_posterior", tree.children)

    def test_r_named_tables_reuse_strict_reader_and_preserve_limits_and_absence(self):
        m, rows = bound_fixture(self.base)
        other = copy.deepcopy(m["chains"][0]); other.update(id=2**64-1, status="failed", reason="initialization")
        other["sampling"] = {"first": None, "last": None, "rows": 0, "failures": 0}
        m["chains"].append(other)
        mp, cp = self.base.files(m, rows)
        directory = Path(__file__).resolve().parent
        code = r'''args <- commandArgs(TRUE)
source(args[1]); x <- read_alea(args[2], args[3], args[4])
stopifnot(nrow(x$posterior) == 2L, is.character(x$coordinates$chain))
stopifnot(x$manifest$chains[[2]]$id == "18446744073709551615")
stopifnot(x$manifest$constant_data[[1]]$values[[1]] == "-9223372036854775808")
stopifnot(x$manifest$constant_data[[2]]$values[[1]] == "18446744073709551615")
stopifnot(identical(names(x$posterior), "μ,\"x\"\r\nnext"))
stopifnot(is.infinite(1/x$posterior[[1]][1]), 1/x$posterior[[1]][1] < 0)
stopifnot(inherits(try(alea_rectangular(x), silent=TRUE), "try-error"))
r <- alea_rectangular(x, chains="9")
stopifnot(identical(r$draw, c("7","8")))
'''
        result = subprocess.run(["Rscript", "-e", code, str(directory.parent / "r/analysis.R"), str(mp), str(cp), str(directory / "analysis.py")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        # A changed CSV is rejected by the same authoritative validator through R.
        cp.write_bytes(cp.read_bytes() + b"bad")
        bad = subprocess.run(["Rscript", "-e", code, str(directory.parent / "r/analysis.R"), str(mp), str(cp), str(directory / "analysis.py")], capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)

    def conjugate(self):
        m, _ = self.base.fixture()
        y = np.asarray([-0.3, 0.1, 0.8, 1.2])
        n = len(y)
        m["parameters"][0]["name"] = "mu"; m["units"] = {}
        m["predictive_units"] = [{"name": f"obs{i}", "factors": [f"density{i}"], "observations": [f"obs{i}"], "conditioned_on": ["mu"], "integrated_out": []} for i in range(n)]
        m["observed_data"] = [{"array": {"name": "y", "axes": [{"name": "observation", "coordinates": [f"obs{i}" for i in range(n)]}], "dtype": "float64", "values": y.tolist(), "unit": None}, "observation_dimension": "observation"}]
        m["csv"]["columns"] = ["chain","phase","draw","kind","error","parameter:mu","unconstrained:0"] + analysis.STATS + [f"log_likelihood:obs{i}" for i in range(n)] + ["prediction:future"]
        m["chains"] = []
        rows = []
        rng = np.random.default_rng(4857)
        for chain in range(4):
            m["chains"].append({"id": chain,"status":"completed","reason":None,"warmup":{"first":None,"last":None,"rows":0,"failures":0},"sampling":{"first":0,"last":999,"rows":1000,"failures":0}})
            for draw, mu in enumerate(rng.normal(y.sum()/(n+1), 1/np.sqrt(n+1), 1000)):
                rows.append([str(chain),"sampling",str(draw),"draw","",str(mu),str(mu),"1","1","","","","",""] + [str(v) for v in norm.logpdf(y,mu,1)] + ["0"])
        return conversion.to_datatree(self.load(m, rows)), y

    def test_psis_loo_matches_exact_conjugate_leave_one_out(self):
        tree, y = self.conjugate()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = conversion.loo_report(tree, reff=1.0)  # independent posterior draws
        exact = norm.logpdf(y, (y.sum()-y)/len(y), np.sqrt(1+1/len(y)))
        np.testing.assert_allclose(report["elpd_i"], exact, atol=0.04, rtol=0)
        self.assertEqual(report["pareto_k"].shape, (4,))
        self.assertTrue(np.isfinite(report["pareto_k"]).all())
        self.assertEqual(report.attrs["n_samples"], 4000)
        path = self.base.root / "loo.nc"
        report.to_netcdf(path, engine="h5netcdf")
        with xr.open_dataset(path, engine="h5netcdf") as restored:
            np.testing.assert_array_equal(restored["pareto_k"], report["pareto_k"])

    def test_high_pareto_diagnostics_and_warnings_are_retained(self):
        tree, _ = self.conjugate()
        # Inject a pathological importance-weight tail; this is a diagnostic test,
        # not a valid alternative posterior model or a quality-passing fit.
        tree["log_likelihood"]["likelihood"].values[:,:,0] = -np.arange(4000).reshape(4,1000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = conversion.loo_report(tree, reff=1.0)
        self.assertGreater(float(report["pareto_k"].values[0]), 0.7)
        self.assertEqual(report.attrs["warning"], 1)
        self.assertTrue(json.loads(report.attrs["warning_messages_json"]))
        self.assertEqual(report["pareto_k"].shape, (4,))

        path = self.base.root / "bad-loo.nc"
        report.to_netcdf(path, engine="h5netcdf")
        with xr.open_dataset(path, engine="h5netcdf") as restored:
            np.testing.assert_array_equal(restored["pareto_k"], report["pareto_k"])
