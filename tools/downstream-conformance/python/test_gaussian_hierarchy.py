"""Reference mathematics tests; actual native reports are checked by Rust integration."""
import math
import unittest
import gaussian_hierarchy as hierarchy


class HierarchyReferenceTests(unittest.TestCase):
    def test_zero_group_scale_reduces_to_one_conjugate_mean(self):
        _,moments = hierarchy.conditional(0.)
        m = dict(zip(hierarchy.NAMES,moments))
        # Six observations, total1.5, unit prior and observation variance.
        self.assertAlmostEqual(m["mu"],1.5/7)
        self.assertAlmostEqual(m["mu2"],1/7+(1.5/7)**2)
        for name in ("theta0","theta1"):self.assertAlmostEqual(m[name],m["mu"])
        for name in ("theta0_2","theta1_2","theta01"):self.assertAlmostEqual(m[name],m["mu2"])
        self.assertEqual(m["same_group_diff2"],2.)
        self.assertEqual(m["different_group_diff2"],2.)

    def test_refined_reference_preserves_group_and_prediction_identities(self):
        r = hierarchy.reference()
        self.assertTrue(all(math.isfinite(v) for v in r.values()))
        self.assertGreater(r["tau"],0)
        self.assertGreater(r["tau2"],r["tau"]**2)
        self.assertLess(r["theta0"],r["mu"])
        self.assertLess(r["mu"],r["theta1"])
        self.assertAlmostEqual(r["existing_y2"],r["theta0_2"]+1)
        self.assertAlmostEqual(r["new_y2"],r["mu2"]+r["tau2"]+1)
        self.assertAlmostEqual(r["different_group_diff2"],2*(r["tau2"]+1))
        self.assertAlmostEqual(r["same_group_diff2"],2)
        with self.assertRaises(ValueError):hierarchy.integrate(12.,3)


if __name__ == "__main__":unittest.main()
