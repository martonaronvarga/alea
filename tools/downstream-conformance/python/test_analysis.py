"""Independent wire tests; fixture construction uses Python's standard CSV/JSON."""
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
import analysis


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self):
        columns = ["chain", "phase", "draw", "kind", "error", 'parameter:μ,"x"\r\nnext', 'unconstrained:0']
        columns += analysis.STATS + ["log_likelihood:joint", "prediction:future"]
        rows = [["9", "sampling", "7", "draw", "", "-0", "-0", "0", "0.75", "", "0", "", "", "", "-1.25", "5e-324"],
                ["9", "sampling", "8", "draw", "", "1.7976931348623157e308", "1.7976931348623157e308", "1", "1", "0", "1", "0", "0", "3", "-0.25", "-0"]]
        empty = {"first": None, "last": None, "rows": 0, "failures": 0}
        m = {"format": "alea-analysis", "schema_version": 2, "kind": "sampling", "csv": {"columns": columns, "sha256": ""},
             "parameters": [{"name": 'μ,"x"\r\nnext', "dimension": None, "coordinate": None, "block": "group"}],
             "unconstrained_dimension": 1, "save_unconstrained": True, "predictions": ["future"],
             "provenance": {"model": "digest:model", "data": "digest:data", "source": "revision", "method": "NUTS", "settings": {"seed": "7"}},
             "units": {'parameter:μ,"x"\r\nnext': "seconds"},
             "predictive_units": [{"name": "joint", "factors": ["first", "second"], "observations": ["row1", "row2"], "conditioned_on": ["mu"], "integrated_out": ["effect"]}],
             "observed_data": [], "constant_data": [], "posterior_predictive": [],
             "chains": [{"id": 9, "status": "completed", "reason": None, "warmup": dict(empty), "sampling": {"first": 7, "last": 8, "rows": 2, "failures": 0}}]}
        return m, rows

    def files(self, manifest, rows):
        m = copy.deepcopy(manifest)
        table = io.StringIO(newline="")
        w = csv.writer(table, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
        w.writerow(m["csv"]["columns"])
        w.writerows(rows)
        payload = table.getvalue().encode()
        m["csv"]["sha256"] = hashlib.sha256(payload).hexdigest()
        mp, cp = self.root / "manifest.json", self.root / "draws.csv"
        mp.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
        cp.write_bytes(payload)
        return mp, cp

    def test_full_precision_unicode_quoted_names_and_statistic_availability(self):
        m, rows = self.fixture()
        a = analysis.load(*self.files(m, rows))
        x, y = a.records
        self.assertEqual(struct.pack(">d", x.values['parameter:μ,"x"\r\nnext']), struct.pack(">d", -0.))
        self.assertEqual(x.values["prediction:future"], math.ulp(0.))
        self.assertEqual(y.values['parameter:μ,"x"\r\nnext'], float.fromhex("0x1.fffffffffffffp+1023"))
        self.assertIsNone(x.values["stat:energy"])
        self.assertIs(x.values["stat:divergent"], False)
        self.assertEqual(y.values["stat:integration_steps"], 0)
        self.assertEqual(a.rectangular()[:2], ([9], [7, 8]))

    def test_partial_unequal_and_empty_planned_chains_are_never_dropped(self):
        m, rows = self.fixture()
        other = copy.deepcopy(m["chains"][0])
        other.update(id=2**64-1, status="interrupted", reason="cancelled")
        other["sampling"].update(last=7, rows=1)
        m["chains"].append(other)
        row = rows[0].copy(); row[0] = str(2**64-1); rows.append(row)
        a = analysis.load(*self.files(m, rows))
        self.assertEqual(a.records[-1].chain, 2**64-1)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            a.rectangular()
        self.assertEqual(a.rectangular(chains=[9])[:2], ([9], [7, 8]))
        other.update(status="completed", reason=None)
        a = analysis.load(*self.files(m, rows))
        with self.assertRaisesRegex(ValueError, "unequal"):
            a.rectangular()
        other.update(status="failed", reason="initialization")
        other["sampling"].update(first=None, last=None, rows=0)
        a = analysis.load(*self.files(m, rows[:2]))
        self.assertEqual(len(a.manifest["chains"]), 2)
        with self.assertRaises(ValueError):
            a.rectangular()

    def test_failure_records_retain_error_and_absent_payload(self):
        m, rows = self.fixture()
        error = 'kernel failed, "detail"\r\nsecond line'
        rows[1] = ["9", "sampling", "8", "failure", error] + [""] * (len(rows[1])-5)
        m["chains"][0]["sampling"]["failures"] = 1
        a = analysis.load(*self.files(m, rows))
        self.assertEqual(a.records[1].error, error)
        self.assertTrue(all(v is None for v in a.records[1].values.values()))
        with self.assertRaisesRegex(ValueError, "failures"):
            a.rectangular()

    def test_exact_csv_bytes_are_sealed(self):
        m, rows = self.fixture()
        mp, cp = self.files(m, rows)
        cp.write_bytes(cp.read_bytes().replace(b"0.75", b"0.76"))
        with self.assertRaisesRegex(ValueError, "digest"):
            analysis.load(mp, cp)

    def test_numeric_grammar_does_not_admit_missing_nonfinite_or_ambiguous_values(self):
        for value in ["NaN", "-inf", "1_0", " 0", "", "1e999", "True"]:
            with self.subTest(value=value):
                m, rows = self.fixture(); rows[0][5] = value
                with self.assertRaises(ValueError): analysis.load(*self.files(m, rows))
        for column, value in [("stat:accepted", "2"), ("stat:tree_depth", "1.0"), ("stat:energy", "nan"), ("stat:acceptance_probability", "1.1")]:
            m, rows = self.fixture(); rows[0][m["csv"]["columns"].index(column)] = value
            with self.assertRaises(ValueError): analysis.load(*self.files(m, rows))

    def test_duplicate_gapped_out_of_order_and_missing_records_fail(self):
        for case in range(5):
            m, rows = self.fixture()
            if case == 0: rows[1][2] = "7"
            elif case == 1: rows[1][2] = "9"
            elif case == 2: rows[1][1] = "warmup"
            elif case == 3: rows.pop()
            else: rows[1].pop()
            with self.assertRaises(ValueError): analysis.load(*self.files(m, rows))

    def test_unknown_and_inconsistent_manifest_fields_fail(self):
        for case in range(10):
            m, rows = self.fixture()
            if case == 0: m["unknown"] = "ignored?"
            elif case == 1: m["schema_version"] = True
            elif case == 2: m["kind"] = "optimization"
            elif case == 3: m["chains"][0]["sampling"]["rows"] = True
            elif case == 4: m["chains"][0]["reason"] = "unexpected"
            elif case == 5: m["chains"][0]["status"] = "failed"
            elif case == 6: m["predictive_units"][0]["integrated_out"] = ["mu"]
            elif case == 7: m["predictive_units"][0]["observations"] = ["row1", "row1"]
            elif case == 8: m["chains"][0]["id"] = 2**64
            else: m["provenance"]["source"] = ""
            with self.assertRaises(ValueError): analysis.load(*self.files(m, rows))

    def test_duplicate_json_keys_and_nonfinite_constants_fail(self):
        m, rows = self.fixture(); mp, cp = self.files(m, rows)
        text = mp.read_text(); mp.write_text(text.replace('"schema_version": 2', '"schema_version": 2, "schema_version": 2'))
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            analysis.load(mp, cp)
        mp.write_text(text.replace('"schema_version": 2', '"schema_version": NaN'))
        with self.assertRaisesRegex(ValueError, "nonfinite JSON"):
            analysis.load(mp, cp)


if __name__ == "__main__":
    unittest.main()


def bound_fixture(base):
    m, rows = base.fixture()
    for row in rows:
        for name in ("stat:energy", "stat:depth_saturated", "stat:integration_steps", "stat:tree_depth"):
            row[m["csv"]["columns"].index(name)] = "0"
    axes = [{"name": "observation", "coordinates": ["row1", "row2"]}]
    m["observed_data"] = [{"array": {"name": "y", "axes": axes, "dtype": "float64", "values": [0.4, 0.8], "unit": "seconds"}, "observation_dimension": "observation"}]
    m["constant_data"] = [{"name": "signed", "axes": [], "dtype": "int64", "values": [-(2**63)], "unit": None},
                           {"name": "unsigned", "axes": [], "dtype": "uint64", "values": [2**64-1], "unit": None},
                           {"name": "label", "axes": [], "dtype": "string", "values": ['λ,"\r\nlabel'], "unit": None}]
    m["predictions"] += ["replica0", "replica1"]
    m["csv"]["columns"] += ["prediction:replica0", "prediction:replica1"]
    for row in rows:
        row += ["0.5", "0.7"]
    m["posterior_predictive"] = [{"variable": "y", "columns": ["replica0", "replica1"]}]
    return m, rows


class DataBindingTests(unittest.TestCase):
    def setUp(self):
        self.base = AnalysisTests()
        self.base.setUp()
        self.addCleanup(self.base.temp.cleanup)

    def test_typed_limits_and_r_bridge_preserve_identity(self):
        m, rows = bound_fixture(self.base)
        a = analysis.load(*self.base.files(m, rows))
        bridge = analysis.r_payload(a)
        self.assertEqual(bridge["manifest"]["constant_data"][0]["values"], [str(-(2**63))])
        self.assertEqual(bridge["manifest"]["constant_data"][1]["values"], [str(2**64-1)])
        self.assertEqual(bridge["records"][0]["chain"], "9")
        self.assertEqual(bridge["records"][1]["values"]["stat:tree_depth"], "0")

    def test_bound_data_shape_types_ids_and_predictive_columns_are_checked(self):
        for case in range(12):
            m, rows = bound_fixture(self.base)
            a = m["observed_data"][0]["array"]
            if case == 0: a["values"].pop()
            elif case == 1: a["dtype"] = "unknown"
            elif case == 2: a.update(dtype="bool", values=[True, 0.5])
            elif case == 3: a.update(dtype="int64", values=[0, 2**63])
            elif case == 4: a.update(dtype="uint64", values=[-1, 1])
            elif case == 5: a["axes"][0]["name"] = "draw"
            elif case == 6: a["axes"][0]["coordinates"][1] = "missing"
            elif case == 7: m["posterior_predictive"][0]["columns"][0] = "missing"
            elif case == 8: m["posterior_predictive"][0]["columns"].pop()
            elif case == 9: m["posterior_predictive"][0]["variable"] = "missing"
            elif case == 10: a.update(dtype="bool", values=[True, False])
            else: m["schema_version"] = 1
            with self.subTest(case=case), self.assertRaises(ValueError):
                analysis.load(*self.base.files(m, rows))
