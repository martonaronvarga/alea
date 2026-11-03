"""Strict independent consumer of the six-case fixed-effect Gaussian workflow.

Uses the generic fit reader plus exact dyadic fixture algebra. Failed native
optimization does not erase valid *separate* OLS intervals; neither it nor small
scores on a rank-deficient model authorize native plug-in forecasts.
"""
import argparse
from fractions import Fraction as F
import json
import math
import sys
from analysis import keys, require, unique_object, reject_constant
import fit_analysis

CASES = ("qualified", "budget", "cancelled", "initial_transform", "rank_deficient", "perfect_fit")
SCOPE = "independent_homoskedastic_gaussian_fixed_design_known_offsets_unconstrained_no_priors_weights_penalties"
POLICY = dict(confidence_level=.95, relative_rank_tolerance=1e-12,
              relative_residual_tolerance=1e-12, quantile_relative_tolerance=1e-8)
CRITICAL_DF4 = 2.7764451051977934  # Independent SciPy1.17.1 t.isf(.025,4).


def array(value, count):
    require(type(value) is list and len(value) == count and all(fit_analysis.finite(v) for v in value), "numeric array")
    return value


def close(actual, expected, tolerance=1e-10):
    require(fit_analysis.finite(actual), "finite number")
    require(abs(actual-float(expected)) <= tolerance*(1+abs(float(expected))), "reference mismatch")


def fixture(case):
    x = list(range(-2, 4))
    offsets = [(i+1)*.25*(-1 if i % 2 else 1) for i in range(6)]
    errors = (0,)*6 if case == "perfect_fit" else (1, -2, 1, 1, -2, 1)
    response = [1.5-.75*z+o+e for z, o, e in zip(x, offsets, errors)]
    design = [v for z in x for v in (1., 1. if case == "rank_deficient" else float(z))]
    return design, response, offsets


def reference(design, response, offsets):
    z = list(map(F, design[1::2]))
    y = [F(a)-F(b) for a, b in zip(response, offsets)]
    sx, sxx, sy, sxy = sum(z), sum(a*a for a in z), sum(y), sum(a*b for a, b in zip(z, y))
    determinant = 6*sxx-sx*sx
    if determinant == 0:
        # This fixture has two identical unit columns; minimum-norm SVD solution.
        beta, rank = [sy/12]*2, 1
    else:
        beta, rank = [(sxx*sy-sx*sxy)/determinant, (6*sxy-sx*sy)/determinant], 2
    rss = sum((a-beta[0]-b*beta[1])**2 for a, b in zip(y, z))
    covariance = None if determinant == 0 else [rss/4*v/determinant for v in (sxx, -sx, -sx, F(6))]
    return beta, rank, rss, covariance


def chain(value):
    require(type(value) is list and value and all(type(v) is str and v for v in value), "error chain")


def validate(record):
    fit_analysis.validate(record)
    a = record["assessment"]
    keys(a, ["format", "schema_version", "case", "assumptions", "coefficient_names", "training",
             "native_evaluation_error", "intervals", "comparison", "predictions"])
    require(a["format"] == "gaussian-workflow" and type(a["schema_version"]) is int and a["schema_version"] == 1, "workflow version")
    case = a["case"]
    require(type(case) is str and case in CASES and a["assumptions"] == SCOPE, "workflow scope/case")
    require(a["coefficient_names"] == ["intercept", "slope"], "coefficient order")
    require([p["name"] for p in record["parameters"]] == ["intercept", "slope", "sigma"] and record["unconstrained_dimension"] == 3, "model coordinates")
    require(record["objective"] == {"kind":"likelihood", "measure":"Lebesgue in intercept, slope and positive sigma"}, "objective measure")
    provenance = record["provenance"]
    require(provenance["model"] == "gaussian-fixed-effect-workflow:v1" and provenance["data"] == f"gaussian-workflow:{case}:v1"
            and provenance["source"] == "public-native-gaussian-workflow:v1" and provenance["settings"] == {"case":case}, "provenance")
    t = a["training"]
    keys(t, ["rows", "columns", "design", "response", "offsets"])
    require(type(t["rows"]) is int and t["rows"] == 6 and type(t["columns"]) is int and t["columns"] == 2, "dimensions")
    expected = fixture(case)
    for key, values in zip(("design", "response", "offsets"), expected):
        require(array(t[key], len(values)) == values, "fixture data/offset/orientation")
    beta, rank, rss, covariance = reference(*expected)
    u = a["intervals"]
    available = case not in ("rank_deficient", "perfect_fit")
    keys(u, ["status", "estimator", "options", "assessment"] + (["residual_variance", "covariance", "standard_errors", "critical_value", "bounds"] if available else ["error_chain"]))
    require(u["status"] == ("available" if available else "unavailable") and u["estimator"] == "separate_fixed_design_least_squares", "interval status/estimator")
    keys(u["options"], POLICY)
    require(all(fit_analysis.finite(u["options"][key]) and u["options"][key] == value for key, value in POLICY.items()), "interval policy")
    d = u["assessment"]
    keys(d, ["numerical_rank", "residual_df", "coefficients", "residual_norm", "mle_sigma", "issues"])
    require(type(d["numerical_rank"]) is int and d["numerical_rank"] == rank and type(d["residual_df"]) is int and d["residual_df"] == 6-rank, "rank/residual df")
    for actual, value in zip(array(d["coefficients"], 2), beta):
        close(actual, value)
    close(d["residual_norm"], math.sqrt(float(rss)))
    if case == "perfect_fit":
        require(d["mle_sigma"] is None and d["issues"] == ["vanishing_residual"], "perfect-fit boundary")
    else:
        close(d["mle_sigma"], math.sqrt(float(rss/6)))
        require(d["issues"] == (["numerical_rank_deficiency"] if rank == 1 else []), "assessment issues")
    if available:
        close(u["residual_variance"], rss/4)
        close(u["critical_value"], CRITICAL_DF4)
        for actual, value in zip(array(u["covariance"],4), covariance):
            close(actual, value)
        require(type(u["bounds"]) is list and len(u["bounds"]) == 2, "interval shape")
        for j, actual in enumerate(array(u["standard_errors"],2)):
            se = math.sqrt(float(covariance[3*j]))
            close(actual, se)
            lo, hi = array(u["bounds"][j],2)
            close(lo, float(beta[j])-CRITICAL_DF4*se, 1e-9)
            close(hi, float(beta[j])+CRITICAL_DF4*se, 1e-9)
    else:
        chain(u["error_chain"])
    term = record["numerical"]["termination"]
    native = record["candidate"]
    if term == "evaluation_failed":
        chain(a["native_evaluation_error"])
    else:
        require(a["native_evaluation_error"] is None, "unexpected backend error")
    if native is not None:
        q, estimates = native["free_position"], native["constrained_estimates"]
        close(q[0], estimates[0], 0.)
        close(q[1], estimates[1], 0.)
        require(estimates[2] > 0, "positive native sigma")
        close(math.log(estimates[2]), q[2], 1e-12)
    keys(a["comparison"], ["status", "relative_tolerance"])
    close(a["comparison"]["relative_tolerance"], 1e-5, 0.)
    status = "no_candidate" if native is None else ("not_stationary" if term != "score_tolerance" else (
        "uncertainty_unavailable" if not available else ("consistent" if all(abs(x-y) <= 1e-5*(1+abs(y))
            for x,y in zip(native["constrained_estimates"], [*map(float,beta),d["mle_sigma"]])) else "mismatch")))
    require(a["comparison"]["status"] == status, "native/reference consistency classification")
    if case == "qualified":
        require(term == "score_tolerance" and status == "consistent", "qualified native fit")
        close(native["value"], -3*(math.log(2*math.pi)+1+math.log(2)), 1e-9)
    elif case == "budget":
        require(term == "evaluation_limit" and native["free_position"] == [.1,-.2,0.] and record["numerical"]["evaluations"] == 1, "budget retention")
    elif case in ("cancelled", "initial_transform"):
        require(native is None and term == ("cancelled" if case == "cancelled" else "evaluation_failed"), "initial stopping outcome")
    p = a["predictions"]
    keys(p, ["kind", "status", "rows"])
    require(p["kind"] == "plug_in_conditional_mean" and p["status"] == ("available" if status == "consistent" else "withheld"), "prediction provenance/status")
    require(type(p["rows"]) is list and len(p["rows"]) == 3, "planned prediction rows")
    for i,(row,(x,offset)) in enumerate(zip(p["rows"],((-3.,.5),(0.,-.25),(4.,1.)))):
        keys(row,["id","design","offset","mean"])
        require(row["id"] == f"new-{i}" and array(row["design"],2) == [1.,x]
                and fit_analysis.finite(row["offset"]) and row["offset"] == offset, "new-data alignment")
        if status == "consistent":
            close(row["mean"], float(beta[0]+F(x)*beta[1])+offset, 1e-5)
            close(row["mean"], math.fsum((native["constrained_estimates"][0],x*native["constrained_estimates"][1],offset)), 1e-12)
        else:
            require(row["mean"] is None, "unqualified native forecasts must remain withheld")
    return record


def read(text, cases=CASES):
    records = [validate(json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_constant)) for line in text.splitlines()]
    require(len(records) == len(cases) and {r["assessment"]["case"] for r in records} == set(cases), "planned cases/duplicates/incomplete workflow")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES)
    args = parser.parse_args()
    records = read(sys.stdin.read(), (args.case,) if args.case else CASES)
    print(json.dumps({"planned":len(records), "completed":len(records),
        "native_numerical_successes":[r["assessment"]["case"] for r in records if r["numerical"]["termination"] == "score_tolerance"],
        "native_reference_consistent":[r["assessment"]["case"] for r in records if r["assessment"]["comparison"]["status"] == "consistent"],
        "intervals_available_for_separate_ols":[r["assessment"]["case"] for r in records if r["assessment"]["intervals"]["status"] == "available"],
        "native_failures_or_incomplete":[r["assessment"]["case"] for r in records if r["numerical"]["termination"] != "score_tolerance"]}))


if __name__ == "__main__":
    main()
