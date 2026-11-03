"""Independent fixed-fixture GLM reference, exact witness replay and fit consumer.

No numerical termination is silently promoted to MLE existence/identification.
Only point fits/conditional means are admitted; no covariance or posterior axes.
"""
import argparse
from fractions import Fraction
import json
import math
import re
import sys
from analysis import keys, require, unique_object, reject_constant
import fit_analysis
from fit_analysis import finite

FAMILIES = ("logistic","poisson")
CASES = ("qualified","budget","cancelled","evaluation_failure","rank_deficient","boundary","unresolved","zero_no_intercept")
NAMES = ["coefficient0","coefficient1"]
SCOPE = "independent_fixed_design_known_offsets_unconstrained_no_priors_weights_penalties"


def close(a,b,tol=1e-9):
    require(finite(a) and finite(b) and abs(a-b) <= tol*(1+abs(b)),"reference mismatch")


def array(a,n):
    require(type(a) is list and len(a)==n and all(finite(x) for x in a),"numeric array")
    return a


def fixture(family,case):
    design = [v for i in range(8) for v in (1.,1. if case=="rank_deficient" else float(i>=4))]
    y = [False,False,False,True,False,True,True,True] if family=="logistic" else [0,1,2,1,1,2,3,2]
    offsets = [.25]*4+[-.5]*4 if family=="logistic" else [math.log(2)]*4+[-math.log(2)]*4
    if case=="boundary":y=[i>=4 for i in range(8)] if family=="logistic" else [0]*8
    if case=="zero_no_intercept":
        design=[1.,0.,-1.,0.,0.,1.,0.,-1.];y=[False]*4 if family=="logistic" else [0]*4;offsets=[0.]*4
    return design,y,offsets


def beta_reference(family,case):
    if case=="zero_no_intercept":return [0.,0.]
    # Independent two-group saturated likelihood: p_g=k_g/n_g or
    # rate_per_exposure_g=sum(y_g)/sum(exposure_g); no optimizer oracle needed.
    return [math.log(1/3)-.25,2*math.log(3)+.75] if family=="logistic" else [-math.log(2),math.log(8)]


def replay_witness(e,design,y,family):
    keys(e,["status","kind","direction","improving_rows","flat_rows"])
    require(type(e["direction"]) is list and len(e["direction"])==2 and all(type(s) is str and re.fullmatch(r"(?:0|-?[1-9][0-9]*)",s) for s in e["direction"]),"canonical exact integer direction")
    direction=list(map(int,e["direction"]))
    improving=flat=0
    for i,event in enumerate(y):
        dot=sum(Fraction(v)*d for v,d in zip(design[2*i:2*i+2],direction))
        if family=="logistic":
            margin=(1 if event else -1)*dot
            require(margin>=0,"invalid logistic witness sign")
            improving+=margin>0;flat+=margin==0
        else:
            require(dot==0 if event>0 else dot<=0,"invalid Poisson witness sign/equality")
            improving+=event==0 and dot<0;flat+=event>0 or dot==0
    require(improving>0 and type(e["improving_rows"]) is int and e["improving_rows"]==improving
            and type(e["flat_rows"]) is int and e["flat_rows"]==flat,"witness counts")
    require(e["kind"]==("poisson_boundary" if family=="poisson" else ("complete_separation" if flat==0 else "quasi_complete_separation")),"witness kind")


def density(family,design,y,offsets,beta):
    terms=[]
    for i,(event,o) in enumerate(zip(y,offsets)):
        eta=math.fsum((o,*(x*b for x,b in zip(design[2*i:2*i+2],beta))))
        if family=="logistic":
            z=eta if event else -eta
            terms.append(-math.log1p(math.exp(-z)) if z>=0 else z-math.log1p(math.exp(z)))
        else:terms.append(event*eta-math.exp(eta)-math.lgamma(event+1))
    return math.fsum(terms)


def validate(r):
    fit_analysis.validate(r)
    a=r["assessment"]
    keys(a,["format","schema_version","family","case","assumptions","training","assessment","admission","native_evaluation_error","uncertainty","predictions"])
    require(a["format"]=="glm-workflow" and type(a["schema_version"]) is int and a["schema_version"]==1,"workflow version")
    family,case=a["family"],a["case"]
    require(type(family) is str and family in FAMILIES and type(case) is str and case in CASES and a["assumptions"]==SCOPE,"workflow scope")
    require(r["parameters"]==[dict(name=n,dimension=None,coordinate=None,block=None) for n in NAMES] and type(r["unconstrained_dimension"]) is int and r["unconstrained_dimension"]==2,"coefficient coordinates")
    require(r["objective"]==dict(kind="likelihood",measure="Lebesgue in two independent real coefficients"),"objective measure")
    require(r["provenance"]==dict(model=f"native-{family}-workflow:v1",data=f"glm-workflow:{family}:{case}:v1",source="public-native-glm-workflow:v1",method="native batch L-BFGS; separate exact-cone and SVD assessment",settings=dict(family=family,case=case)),"provenance")
    design,y,offsets=fixture(family,case)
    t=a["training"];keys(t,["rows","columns","design","response","offsets"])
    require(type(t["rows"]) is int and t["rows"]==len(y) and type(t["columns"]) is int and t["columns"]==2,"dimensions")
    require(array(t["design"],len(design))==design,"design orientation")
    require(type(t["response"]) is list and len(t["response"])==len(y) and all(type(v) is (bool if family=="logistic" else int) for v in t["response"]) and t["response"]==y,"response type/value")
    for v,expected in zip(array(t["offsets"],len(offsets)),offsets):close(v,expected,1e-15)
    e=a["assessment"];keys(e,["coefficient_names","existence","rank","search"])
    require(e["coefficient_names"]==NAMES,"evidence coefficient order")
    search=e["search"];keys(search,["max_columns","max_rows","max_coefficient_bits"])
    require(all(type(v) is int for v in search.values()) and search==dict(max_columns=32,max_rows=0 if case=="unresolved" else 16384,max_coefficient_bits=8192),"search budget")
    rank=e["rank"];keys(rank,["rows","columns","numerical_rank","relative_tolerance"])
    require(all(type(rank[k]) is int for k in ("rows","columns","numerical_rank")) and rank["rows"]==len(y) and rank["columns"]==2 and rank["numerical_rank"]==(1 if case=="rank_deficient" else 2),"rank policy/result")
    close(rank["relative_tolerance"],1e-12,0.)
    existence="no_finite_maximum" if case=="boundary" else ("unresolved" if case=="unresolved" else "finite_maximum")
    require(type(e["existence"]) is dict and e["existence"].get("status")==existence,"finite-attainment evidence")
    if case=="boundary":replay_witness(e["existence"],design,y,family)
    else:keys(e["existence"],["status"])
    options=r["numerical"]["options"]
    require(options["iteration_limit"]==200 and options["evaluation_limit"]==(1 if case=="budget" else 2000) and options["scales"] is None,"fit policy")
    close(options["score_tolerance"],1e-6,0.)
    term=r["numerical"]["termination"];candidate=r["candidate"]
    error=a["native_evaluation_error"]
    require((type(error) is list and error and all(type(s) is str and s for s in error)) if term=="evaluation_failed" else error is None,"original backend error chain")
    if candidate is not None:
        require(candidate["free_position"]==candidate["constrained_estimates"],"real-coordinate projection")
        close(candidate["value"],density(family,design,y,offsets,candidate["free_position"]))
    admission=("no_finite_maximum" if existence=="no_finite_maximum" else "existence_unresolved" if existence=="unresolved" else "nonidentified" if case=="rank_deficient" else "no_candidate" if candidate is None else "not_stationary" if term!="score_tolerance" else "admitted_point_fit")
    require(a["admission"]==admission,"numerical/existence/identification separation")
    if case in ("qualified","zero_no_intercept"):
        require(admission=="admitted_point_fit","positive point-fit case")
        for v,expected in zip(candidate["constrained_estimates"],beta_reference(family,case)):close(v,expected,1e-5)
    if case=="budget":require(term=="evaluation_limit" and candidate["free_position"]==[.1,-.2],"retained budget candidate")
    if case in ("cancelled","evaluation_failure"):
        require(candidate is None and term==("cancelled" if case=="cancelled" else "evaluation_failed"),"initial stopping outcome")
    require(a["uncertainty"]==dict(status="not_computed",reason="no_explicit_likelihood_hvp_supplied"),"no invented covariance/intervals")
    p=a["predictions"];keys(p,["kind","quantity","status","rows"])
    admitted=admission=="admitted_point_fit"
    require(p["kind"]=="plug_in_conditional_mean" and p["quantity"]==("event_probability" if family=="logistic" else "expected_count") and p["status"]==("available" if admitted else "withheld"),"prediction meaning/status")
    require(type(p["rows"]) is list and len(p["rows"])==3,"planned new-data rows")
    for i,(row,x,o) in enumerate(zip(p["rows"],(0.,1.,.5),(0.,0.,0.) if family=="logistic" else (2*math.log(2),-2*math.log(2),0.))):
        keys(row,["id","design","offset","mean"])
        require(row["id"]==f"new-{i}" and array(row["design"],2)==[1.,x],"new-data identity/orientation")
        close(row["offset"],o,1e-15)
        if admitted:
            beta=candidate["constrained_estimates"];eta=math.fsum((beta[0],x*beta[1],o))
            close(row["mean"],1/(1+math.exp(-eta)) if family=="logistic" else math.exp(eta))
        else:require(row["mean"] is None,"unqualified forecast")
    return r


def read(text,plan=None):
    plan=plan or [(f,c) for f in FAMILIES for c in CASES]
    records=[validate(json.loads(line,object_pairs_hook=unique_object,parse_constant=reject_constant)) for line in text.splitlines()]
    require(len(records)==len(plan) and {(r["assessment"]["family"],r["assessment"]["case"]) for r in records}==set(plan),"planned pairs/duplicates/incomplete cohort")
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--family",choices=FAMILIES);p.add_argument("--case",choices=CASES);args=p.parse_args()
    require(bool(args.family)==bool(args.case),"supply both family and case")
    records=read(sys.stdin.read(),[(args.family,args.case)] if args.family else None)
    identity=lambda r:f'{r["assessment"]["family"]}/{r["assessment"]["case"]}'
    print(json.dumps(dict(planned=len(records),completed=len(records),admitted=[identity(r) for r in records if r["assessment"]["admission"]=="admitted_point_fit"],
        numerical_score_termination=[identity(r) for r in records if r["numerical"]["termination"]=="score_tolerance"],
        outcomes={identity(r):r["assessment"]["admission"] for r in records}),sort_keys=True))


if __name__=="__main__":main()
