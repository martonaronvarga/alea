"""Independent marginal reference and sealed-artifact hierarchy qualification.

This is a test/application oracle, not a production estimator or inference API.
The mean/scale prior and normal random effects integrate to one scalar tau integral.
All seeds, budgets and gates are fixed before inspecting native sampler results.
"""
import functools
import io
import json
import math
import sys
import analysis
from analysis import require, keys
from fit_analysis import finite

CASES = ("complete", "cancelled", "initial_transform")
NAMES = ("mu", "tau", "tau2", "theta0", "theta1", "theta0_2", "theta1_2", "mu2", "theta01",
         "existing_y", "existing_y2", "new_y", "new_y2", "same_group_diff2", "different_group_diff2")
PREDICTIONS = ["theta0", "theta1", "existing_y", "new_a_y0", "new_a_y1", "new_b_y0"]
INITIAL = [[-.5,-.7,-.2,.3],[0.,0.,0.,0.],[.5,.4,.3,-.2],[1.,-.3,-.3,.2]]


def conditional(tau):
    """Integrate mu and group effects analytically at fixed physical tau.

    Group means (-.5,1) have covariance V*I+11', V=tau²+1/3.
    This direct two-group algebra is independent of Alea's four-dimensional graph.
    """
    v = tau*tau+1/3
    mu, mu_var = .5/(v+2), v/(v+2)
    h = (1/3)/v
    theta = [(1-h)*y+h*mu for y in (-.5,1.)]
    theta_var = tau*tau*h+h*h*mu_var
    mu2 = mu*mu+mu_var
    t02, t12 = (t*t+theta_var for t in theta)
    cross = theta[0]*theta[1]+h*h*mu_var
    quadratic = 1.25/v-.25/(v*(v+2))
    weight = math.exp(-.5*(tau*tau+quadratic))/math.sqrt(v*(v+2))
    moments = [mu,tau,tau*tau,*theta,t02,t12,mu2,cross,
               theta[0],t02+1,mu,mu2+tau*tau+1,2.,2*(tau*tau+1)]
    return weight, moments


def integrate(limit, panels):
    require(type(panels) is int and panels > 0 and panels % 2 == 0, "even quadrature panels")
    columns = [[] for _ in range(1+len(NAMES))]
    for i in range(panels+1):
        w, moments = conditional(limit*i/panels)
        coefficient = 1 if i in (0,panels) else (4 if i % 2 else 2)
        for column, value in zip(columns, [1.,*moments]):
            column.append(coefficient*w*value)
    integrals = [math.fsum(c)*(limit/panels)/3 for c in columns]
    return dict(zip(NAMES,[v/integrals[0] for v in integrals[1:]]))


@functools.lru_cache(maxsize=1)
def reference():
    # Refinement AND domain extension, not a rigorous quadrature error enclosure.
    coarse, fine, extended = integrate(12.,6144), integrate(12.,12288), integrate(20.,20480)
    require(all(abs(fine[k]-coarse[k]) < 1e-9 and abs(fine[k]-extended[k]) < 1e-9 for k in NAMES), "reference quadrature refinement/domain extension")
    return extended


def close(a,b,tolerance=1e-10):
    require(finite(a) and finite(b) and abs(a-b) <= tolerance*(1+abs(b)), "numeric replay mismatch")


def observable(row):
    v = row.values
    mu,tau,z0,z1 = [v["parameter:"+k] for k in ("mu","tau","z0","z1")]
    require(tau > 0, "positive physical tau")
    close(math.log(tau),v["unconstrained:1"])
    for i,x in ((0,mu),(2,z0),(3,z1)):
        close(x,v[f"unconstrained:{i}"],0.)
    theta = [mu+tau*z0,mu+tau*z1]
    for g in range(2):
        close(theta[g],v[f"prediction:theta{g}"])
        y = ((-1.,-.5,0.),(.5,1.,1.5))[g]
        close(v[f"log_likelihood:existing_group{g}"],math.fsum(-.5*(x-theta[g])**2-.5*math.log(2*math.pi) for x in y))
    existing,a0,a1,b0 = [v["prediction:"+k] for k in PREDICTIONS[2:]]
    return [mu,tau,tau*tau,*theta,theta[0]**2,theta[1]**2,mu*mu,theta[0]*theta[1],
            existing,existing**2,a0,a0*a0,(a0-a1)**2,(a0-b0)**2]


def check_summaries(summaries, values):
    keys(summaries,NAMES)
    truth = reference()
    maximum_z = 0.
    for i,name in enumerate(NAMES):
        s = summaries[name]
        keys(s,["mean","rhat","bulk_ess","tail_ess","mean_ess","mcse_mean"])
        require(all(finite(x) for x in s.values()), "finite diagnostics")
        require(0 < s["rhat"] < 1.05 and min(s["bulk_ess"],s["tail_ess"],s["mean_ess"]) > 200, "predeclared Rhat/ESS gate")
        x = [v[i] for v in values]
        mean = math.fsum(x)/len(x)
        close(s["mean"],mean)
        variance = math.fsum((v-mean)**2 for v in x)/(len(x)-1)
        close(s["mcse_mean"],math.sqrt(variance/s["mean_ess"]))
        require(s["mcse_mean"] > 0, "positive MCSE")
        z = abs(mean-truth[name])/s["mcse_mean"]
        require(z < 6, f"reference moment gate: {name}, z={z}")
        maximum_z = max(maximum_z,z)
    return maximum_z


def validate(r):
    keys(r,["format","schema_version","case","plan","session_status","sink_error","chains","summaries","manifest","csv"])
    require(r["format"] == "gaussian-hierarchy-workflow" and type(r["schema_version"]) is int and r["schema_version"] == 1, "workflow version")
    case = r["case"]
    require(type(case) is str and case in CASES, "planned workflow case")
    plan = r["plan"]
    initial = [q.copy() for q in INITIAL]
    if case == "initial_transform":
        for q in initial:q[1] = -1000.
    expected = dict(chains=4,warmup=1000,draws=2000,initial=initial,seed=[83]*32,step_size=.1,max_depth=8,target_acceptance=.9,metric="diagonal",search=None)
    keys(plan,expected)
    require(all(type(plan[k]) is int for k in ("chains","warmup","draws","max_depth"))
            and type(plan["seed"]) is list and all(type(x) is int for x in plan["seed"])
            and all(finite(plan[k]) for k in ("step_size","target_acceptance"))
            and type(plan["initial"]) is list and all(type(q) is list and all(finite(x) for x in q) for q in plan["initial"]), "plan numeric types")
    require(plan == expected, "declared plan/seed/budget")
    require(r["sink_error"] is None and type(r["csv"]) is str, "sink/CSV failure")
    artifact = analysis.load_stream(r["manifest"],io.BytesIO(r["csv"].encode("utf-8")))
    m = artifact.manifest
    require(m["provenance"] == dict(model="gaussian-hierarchy:half-normal-tau:fixed-residual-scale:v1",data="two-groups:[-1,-.5,0],[.5,1,1.5]:v1",source="public-native-gaussian-hierarchy:v1",method="session NUTS; diagonal covariance warmup",
            settings={"seed":"[83;32]; logical chain streams","prediction_seed":"83001+chain; SmallRng","case":case}), "workflow provenance")
    require([p["name"] for p in m["parameters"]] == ["mu","tau","z0","z1"] and m["save_unconstrained"] and m["unconstrained_dimension"] == 4, "coordinates")
    require(m["predictions"] == PREDICTIONS, "prediction alignment")
    coordinates = [f"group{g}/row{i}" for g in range(2) for i in range(3)]
    axes = [dict(name="observation",coordinates=coordinates)]
    require(m["observed_data"] == [dict(array=dict(name="y",axes=axes,values=[-1.,-.5,0.,.5,1.,1.5],dtype="float64",unit=None),observation_dimension="observation")], "observed values/alignment")
    require(m["constant_data"] == [dict(name="group",axes=axes,values=[0,0,0,1,1,1],dtype="uint64",unit=None)] and m["posterior_predictive"] == [], "group alignment/generic generated quantities")
    require([u["name"] for u in m["predictive_units"]] == ["existing_group0","existing_group1"], "pointwise units")
    for g,u in enumerate(m["predictive_units"]):
        require(u == dict(name=f"existing_group{g}",factors=[f"group{g}/row{i}" for i in range(3)],observations=[f"group{g}/row{i}" for i in range(3)],conditioned_on=["mu","tau",f"z{g}"],integrated_out=[]), "conditional likelihood is not marginal new-group likelihood")
    require([c["id"] for c in m["chains"]] == list(range(4)), "all planned chains")
    require(type(r["chains"]) is list and len(r["chains"]) == 4, "chain outcome count")
    for i,(c,manifest_chain) in enumerate(zip(r["chains"],m["chains"])):
        keys(c,["id","warmup_completed","completed","cancelled","error_chain","numerical_rejections","final_step_size"])
        require(type(c["id"]) is int and c["id"] == i and type(c["cancelled"]) is bool, "chain identity/cancellation")
        require(all(type(c[k]) is int and c[k] >= 0 for k in ("warmup_completed","completed","numerical_rejections")), "chain counts")
        require(c["completed"] == manifest_chain["sampling"]["rows"], "native/export count agreement")
        if case == "complete":
            require(c["warmup_completed"] == 1000 and c["completed"] == 2000 and not c["cancelled"] and c["error_chain"] is None and c["numerical_rejections"] == 0, "incomplete/failed native chain")
            require(finite(c["final_step_size"]) and c["final_step_size"] > 0, "adapted step")
        else:
            require(c["completed"] == 0 and c["warmup_completed"] == 0 and c["final_step_size"] is None, "empty control chain")
            if case == "cancelled":
                require(c["cancelled"] and c["error_chain"] is None and manifest_chain["status"] == "interrupted", "cancelled outcome")
            else:
                require(not c["cancelled"] and type(c["error_chain"]) is list and c["error_chain"] and all(type(s) is str and s for s in c["error_chain"]) and manifest_chain["status"] == "failed", "original initial failure")
    if case != "complete":
        require(r["session_status"] == ("cancelled" if case == "cancelled" else "backend_failure") and r["summaries"] is None and not artifact.records, "failed/cancelled outcomes must not acquire inference")
        return dict(case=case,session_status=r["session_status"],planned_chains=4,completed_chains=0,qualified=False)
    require(r["session_status"] in ("success","diagnostic_warning"), "native session failure")
    _,indices,chains = artifact.rectangular()
    require(indices == list(range(2000)), "retained draw coordinates")
    rows = [row for c in chains for row in c]
    require(all(row.values["stat:divergent"] is not None and row.values["stat:depth_saturated"] is not None for row in rows), "missing transition diagnostics")
    divergences = sum(row.values["stat:divergent"] for row in rows)
    saturated = sum(row.values["stat:depth_saturated"] for row in rows)
    require(divergences/len(rows) <= .01 and saturated/len(rows) <= .01, "predeclared transition gate")
    z = check_summaries(r["summaries"],[observable(row) for row in rows])
    return dict(case=case,session_status=r["session_status"],planned_chains=4,completed_chains=4,qualified=True,retained=len(rows),divergences=divergences,depth_saturated=saturated,maximum_moment_mcse_units=z,
                maximum_rhat=max(s["rhat"] for s in r["summaries"].values()),minimum_bulk_ess=min(s["bulk_ess"] for s in r["summaries"].values()),minimum_tail_ess=min(s["tail_ess"] for s in r["summaries"].values()))


def main():
    report = json.load(sys.stdin,object_pairs_hook=analysis.unique_object,parse_constant=analysis.reject_constant)
    print(json.dumps(validate(report),sort_keys=True))


if __name__ == "__main__":main()
