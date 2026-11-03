"""Independent oracle tests; Rust integration supplies actual native reports."""
import copy
import math
import unittest
import glm_workflow as workflow


class GlmReferenceTests(unittest.TestCase):
    def test_saturated_group_estimates_include_offsets_and_exposure(self):
        for family in workflow.FAMILIES:
            design,y,offsets=workflow.fixture(family,"qualified")
            beta=workflow.beta_reference(family,"qualified")
            # At the independent two-group solution every score vanishes.
            for coordinate in range(2):
                p,m=beta.copy(),beta.copy();p[coordinate]+=1e-5;m[coordinate]-=1e-5
                derivative=(workflow.density(family,design,y,offsets,p)-workflow.density(family,design,y,offsets,m))/2e-5
                self.assertLess(abs(derivative),1e-7)
            eta=[beta[0]+g*beta[1]+offsets[4*g] for g in range(2)]
            means=[1/(1+math.exp(-v)) if family=="logistic" else math.exp(v) for v in eta]
            for observed,expected in zip(means,(.25,.75) if family=="logistic" else (1.,2.)):
                self.assertAlmostEqual(observed,expected)
            design,y,offsets=workflow.fixture(family,"zero_no_intercept")
            at_zero=workflow.density(family,design,y,offsets,[0.,0.])
            for direction in ((1.,0.),(-1.,0.),(0.,1.),(0.,-1.)):
                self.assertLess(workflow.density(family,design,y,offsets,direction),at_zero)

    def test_exact_integer_recession_replay_does_not_round_through_float(self):
        for family,direction,kind in (("logistic",["-1","2"],"complete_separation"),("poisson",["-1","0"],"poisson_boundary")):
            design,y,_=workflow.fixture(family,"boundary")
            e=dict(status="no_finite_maximum",kind=kind,direction=direction,improving_rows=8,flat_rows=0)
            workflow.replay_witness(e,design,y,family)
            scaled=copy.deepcopy(e)
            scaled["direction"]=[str(int(v)*2**2048) for v in direction]
            workflow.replay_witness(scaled,design,y,family)
            for bad in (["0","0"],["1","-2"],["-01","2"],[True,"2"]):
                mutated=copy.deepcopy(e);mutated["direction"]=bad
                with self.subTest(family=family,direction=bad),self.assertRaises(ValueError):
                    workflow.replay_witness(mutated,design,y,family)


if __name__=="__main__":unittest.main()
