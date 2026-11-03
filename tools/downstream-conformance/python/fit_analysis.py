"""Strict, standard-library reader for Alea's point-estimation JSON reports.

Returns the entire validated record, including unsuccessful numerical outcomes
and opaque caller assessments. Never creates posterior/chain axes or uncertainty.
Does not reconstruct charts, verify source digests or establish estimator validity.
"""
import json
import math
from pathlib import Path

from analysis import keys, names, reject_constant, require, uint, unique_object


def finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def finite_json(value):
    if type(value) is float:
        require(math.isfinite(value), "nonfinite JSON payload")
    elif type(value) is list:
        for item in value:
            finite_json(item)
    elif type(value) is dict:
        for item in value.values():
            finite_json(item)


def validate(record):
    finite_json(record)
    keys(record, ["format", "schema_version", "kind", "provenance", "objective",
                  "parameters", "unconstrained_dimension", "numerical", "candidate", "assessment"])
    require(record["format"] == "alea-fit" and type(record["schema_version"]) is int
            and record["schema_version"] == 1 and record["kind"] == "point_estimate",
            "unsupported fit format")
    provenance = record["provenance"]
    keys(provenance, ["model", "data", "source", "method", "settings"])
    require(all(type(provenance[k]) is str and provenance[k] for k in
                ("model", "data", "source", "method")), "invalid provenance")
    require(type(provenance["settings"]) is dict and all(type(k) is str and k and type(v) is str
            for k, v in provenance["settings"].items()), "invalid settings")
    objective = record["objective"]
    keys(objective, ["kind", "measure"])
    require(objective["kind"] in ("likelihood", "posterior", "other") and
            type(objective["measure"]) is str and objective["measure"], "invalid objective")
    parameters = record["parameters"]
    require(type(parameters) is list, "invalid parameter schema")
    for p in parameters:
        keys(p, ["name", "dimension", "coordinate", "block"])
        require(all(p[k] is None or type(p[k]) is str for k in
                    ("dimension", "coordinate", "block")), "invalid coordinate metadata")
    names([p["name"] for p in parameters])
    dimension = uint(record["unconstrained_dimension"])
    numerical = record["numerical"]
    keys(numerical, ["algorithm", "termination", "line_search_failure", "options",
                     "iterations", "evaluations", "domain_trials", "skipped_updates"])
    require(numerical["algorithm"] == "lbfgs_armijo", "unsupported optimization algorithm")
    options = numerical["options"]
    keys(options, ["iteration_limit", "evaluation_limit", "history_limit", "line_search_limit",
                   "score_tolerance", "scales"])
    for key in ("iteration_limit", "evaluation_limit", "history_limit", "line_search_limit"):
        require(uint(options[key]) > 0, "empty optimization budget")
    require(finite(options["score_tolerance"]) and options["score_tolerance"] > 0, "invalid score tolerance")
    scales = options["scales"]
    require(scales is None or (type(scales) is list and len(scales) == dimension
            and all(finite(v) and v > 0 for v in scales)), "invalid coordinate scales")
    for key in ("iterations", "evaluations", "domain_trials", "skipped_updates"):
        uint(numerical[key])
    require(numerical["iterations"] <= options["iteration_limit"] and
            numerical["evaluations"] <= options["evaluation_limit"] and
            numerical["skipped_updates"] <= numerical["iterations"], "inconsistent fit counters")
    term = numerical["termination"]
    require(term in ("score_tolerance", "iteration_limit", "evaluation_limit", "cancelled",
                    "line_search_failed", "numerical_direction", "evaluation_failed"), "unknown termination")
    reason = numerical["line_search_failure"]
    require((reason in ("backtrack_limit", "no_representable_step")) if term == "line_search_failed"
            else reason is None, "inconsistent line-search reason")
    candidate = record["candidate"]
    require(numerical["evaluations"] >= numerical["iterations"] + int(candidate is not None),
            "inconsistent evaluation count")
    if candidate is None:
        require(term in ("cancelled", "evaluation_failed") and numerical["iterations"] == 0
                and numerical["domain_trials"] == 0
                and numerical["evaluations"] <= int(term == "evaluation_failed"),
                "missing candidate for stopping evidence")
    else:
        keys(candidate, ["free_position", "value", "free_gradient", "constrained_estimates"])
        for key, count in (("free_position", dimension), ("free_gradient", dimension),
                           ("constrained_estimates", len(parameters))):
            require(type(candidate[key]) is list and len(candidate[key]) == count
                    and all(finite(v) for v in candidate[key]), "invalid candidate dimensions or values")
        require(finite(candidate["value"]), "invalid objective value")
    if term == "score_tolerance":
        require(all(finite(score * (1 if scales is None else scales[i]))
                    and abs(score * (1 if scales is None else scales[i])) <= options["score_tolerance"]
                    for i, score in enumerate(candidate["free_gradient"])), "score condition not met")
    elif term == "iteration_limit":
        require(numerical["iterations"] == options["iteration_limit"], "iteration limit not reached")
    elif term == "evaluation_limit":
        require(numerical["evaluations"] == options["evaluation_limit"], "evaluation limit not reached")
    return record


def load(path):
    return validate(json.loads(Path(path).read_text(encoding="utf-8"),
                               object_pairs_hook=unique_object, parse_constant=reject_constant))
