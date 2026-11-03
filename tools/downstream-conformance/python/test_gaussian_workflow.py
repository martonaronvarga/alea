import copy
import json
import math
import unittest
import gaussian_workflow as workflow


def record(case="qualified"):
    design,response,offsets=workflow.fixture(case)
    rank=1 if case=="rank_deficient" else 2
    rss=21.84375 if rank==1 else (0. if case=="perfect_fit" else 12.)
    beta=[.5625,.5625] if rank==1 else [1.5,-.75]
    sigma=math.sqrt(rss/6) if rss else None
    assessment=dict(numerical_rank=rank,residual_df=6-rank,coefficients=beta,residual_norm=math.sqrt(rss),
                    mle_sigma=sigma,issues=["numerical_rank_deficiency"] if rank==1 else (["vanishing_residual"] if not rss else []))
    intervals=dict(status="unavailable",estimator="separate_fixed_design_least_squares",options=dict(workflow.POLICY),assessment=assessment,error_chain=["unqualified intervals"])
    if case not in ("rank_deficient","perfect_fit"):
        covariance=[57/105,-9/105,-9/105,18/105]
        se=[math.sqrt(covariance[0]),math.sqrt(covariance[3])]
        intervals=dict(status="available",estimator="separate_fixed_design_least_squares",options=dict(workflow.POLICY),assessment=assessment,
            residual_variance=3.,covariance=covariance,standard_errors=se,critical_value=workflow.CRITICAL_DF4,
            bounds=[[b-workflow.CRITICAL_DF4*s,b+workflow.CRITICAL_DF4*s] for b,s in zip(beta,se)])
    term={"qualified":"score_tolerance","budget":"evaluation_limit","cancelled":"cancelled","initial_transform":"evaluation_failed",
          "rank_deficient":"score_tolerance","perfect_fit":"evaluation_failed"}[case]
    native=None
    if case not in ("cancelled","initial_transform"):
        estimates=[.1,-.2,1.] if case=="budget" else [*beta,sigma or .1]
        native=dict(free_position=[*estimates[:2],math.log(estimates[2])],constrained_estimates=estimates,
                    free_gradient=[0.,0.,0.],value=-3*(math.log(2*math.pi)+1+math.log(2)))
    status="no_candidate" if native is None else ("not_stationary" if term!="score_tolerance" else ("uncertainty_unavailable" if rank==1 else "consistent"))
    evidence=dict(format="gaussian-workflow",schema_version=1,case=case,assumptions=workflow.SCOPE,coefficient_names=["intercept","slope"],
        training=dict(rows=6,columns=2,design=design,response=response,offsets=offsets),native_evaluation_error=["original transform failure"] if term=="evaluation_failed" else None,
        intervals=intervals,comparison=dict(status=status,relative_tolerance=1e-5),
        predictions=dict(kind="plug_in_conditional_mean",status="available" if status=="consistent" else "withheld",
            rows=[dict(id=f"new-{i}",design=[1.,x],offset=o,mean=1.5-.75*x+o if status=="consistent" else None)
                  for i,(x,o) in enumerate(((-3.,.5),(0.,-.25),(4.,1.)))]))
    return dict(format="alea-fit",schema_version=1,kind="point_estimate",
        provenance=dict(model="gaussian-fixed-effect-workflow:v1",data=f"gaussian-workflow:{case}:v1",source="public-native-gaussian-workflow:v1",method="fixture",settings=dict(case=case)),
        objective=dict(kind="likelihood",measure="Lebesgue in intercept, slope and positive sigma"),
        parameters=[dict(name=n,dimension=None,coordinate=None,block=None) for n in ("intercept","slope","sigma")],unconstrained_dimension=3,
        numerical=dict(algorithm="lbfgs_armijo",termination=term,line_search_failure=None,iterations=0,
            evaluations=0 if case=="cancelled" else (2 if case=="perfect_fit" else 1),domain_trials=0,skipped_updates=0,
            options=dict(iteration_limit=200,evaluation_limit=1 if case=="budget" else 2000,history_limit=10,line_search_limit=25,score_tolerance=1e-6,scales=None)),
        candidate=native,assessment=evidence)


class GaussianWorkflowTests(unittest.TestCase):
    def test_all_outcomes_keep_distinct_native_and_ols_estimates(self):
        records=workflow.read("\n".join(json.dumps(record(c)) for c in workflow.CASES))
        self.assertEqual(len(records),6)
        for case in ("budget","cancelled","initial_transform"):
            r=workflow.validate(record(case))
            self.assertEqual(r["assessment"]["intervals"]["status"],"available")
            self.assertEqual(r["assessment"]["predictions"]["status"],"withheld")

    def test_no_false_native_success_or_posterior_prediction(self):
        for case in workflow.CASES[1:]:
            r=record(case);r["assessment"]["comparison"]["status"]="consistent"
            with self.subTest(case=case),self.assertRaises(ValueError):workflow.validate(r)
        r=record();r["assessment"]["predictions"]["kind"]="posterior"
        with self.assertRaises(ValueError):workflow.validate(r)

    def test_wrong_interval_mean_df_sigma_and_new_data_alignment_reject(self):
        mutations=[lambda a:a["intervals"].update(estimator="native_candidate"),
            lambda a:a["intervals"]["assessment"].update(mle_sigma=math.sqrt(3)),
            lambda a:a["intervals"]["assessment"].update(residual_df=6),
            lambda a:a["intervals"]["covariance"].__setitem__(1,.2),
            lambda a:a["intervals"].update(critical_value=1.96),
            lambda a:a["intervals"]["bounds"][0].__setitem__(1,999),
            lambda a:a["predictions"]["rows"][0].update(id="new-1"),
            lambda a:a["predictions"]["rows"][0].update(offset=0),
            lambda a:a["predictions"]["rows"][2].update(offset=True),
            lambda a:a["predictions"]["rows"][0].update(mean=0),
            lambda a:a["training"]["offsets"].__setitem__(0,0),
            lambda a:a.update(extra=0),lambda a:a.update(schema_version=True),
            lambda a:a["comparison"].update(relative_tolerance=True)]
        for i,mutate in enumerate(mutations):
            r=record();mutate(r["assessment"])
            with self.subTest(mutation=i),self.assertRaises(ValueError):workflow.validate(r)

    def test_unavailable_uncertainty_and_error_chains_cannot_be_manufactured(self):
        r=record("rank_deficient");r["assessment"]["intervals"]["bounds"]=[[0,1],[0,1]]
        with self.assertRaises(ValueError):workflow.validate(r)
        for case in ("initial_transform","perfect_fit"):
            r=record(case);r["assessment"]["native_evaluation_error"]=None
            with self.subTest(case=case),self.assertRaises(ValueError):workflow.validate(r)
        r=record("cancelled");r["candidate"]=copy.deepcopy(record()["candidate"])
        with self.assertRaises(ValueError):workflow.validate(r)

    def test_missing_duplicate_truncated_or_blank_records_fail_the_planned_cohort(self):
        valid=[json.dumps(record(c)) for c in workflow.CASES]
        for text in ("\n".join(valid[:-1]),"\n".join(valid+[valid[0]]),"\n".join(valid[:1]*6),"\n".join(valid)+"\n\n",valid[0][:-1],valid[0].replace('"schema_version": 1','"schema_version": 1, "schema_version": 1')):
            with self.assertRaises(ValueError):workflow.read(text)


if __name__=="__main__":unittest.main()
