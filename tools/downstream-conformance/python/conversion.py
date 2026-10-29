"""Optional conversion of validated Alea artifacts to an ArviZ-compatible DataTree.

Qualified with the versions in requirements-analysis.txt / the locked Nix analysis
shell. Native Alea and the strict CSV/JSON reader need none of these dependencies.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import xarray as xr
from analysis import Artifact, load, require

STAT_NAMES = {
    "accepted": "accepted", "acceptance_probability": "acceptance_rate",
    # Alea records initial Hamiltonian, not accepted-proposal Hamiltonian.
    "energy": "initial_energy", "divergent": "diverging",
    "depth_saturated": "reached_max_treedepth",
    # A macrostep need not be one leapfrog step for a nonreference integrator.
    "integration_steps": "integration_steps", "tree_depth": "tree_depth",
}
DTYPES = {"float64": np.float64, "int64": np.int64, "uint64": np.uint64,
          "bool": np.bool_, "string": str}


def data_array(spec):
    dims = [a["name"] for a in spec["axes"]]
    coords = {a["name"]: a["coordinates"] for a in spec["axes"]}
    shape = tuple(len(a["coordinates"]) for a in spec["axes"])
    values = np.asarray(spec["values"], dtype=DTYPES[spec["dtype"]]).reshape(shape)
    attrs = {"alea_dtype": spec["dtype"]}
    if spec["unit"] is not None:
        attrs["alea_units"] = spec["unit"]
    return xr.DataArray(values, dims=dims, coords=coords, attrs=attrs)


def to_datatree(artifact: Artifact, chains=None, include_warmup=True):
    """Preserve semantics and require explicit, completed rectangular samples.

    Full chain plans and predictive declarations remain in root metadata, including
    excluded chains after an explicit selection. All failure records remain in the
    analysis_failures group. Source CSV/digest remains the authoritative full table.
    """
    m = artifact.manifest
    selected, indices, rows = artifact.rectangular(chains=chains)
    groups = {"/": xr.Dataset(attrs={
        "creation_library": "alea", "creation_library_language": "Rust",
        "alea_manifest_json": json.dumps(m, ensure_ascii=False, allow_nan=False),
        "alea_selection_json": json.dumps({"chains": [str(c) for c in selected],
                                            "include_warmup": include_warmup}),
    })}
    bound_columns = {c for b in m["posterior_predictive"] for c in b["columns"]}
    observed = {b["array"]["name"]: b["array"] for b in m["observed_data"]}
    for kind, arrays in [("observed_data", list(observed.values())), ("constant_data", m["constant_data"])]:
        if arrays:
            groups["/" + kind] = xr.Dataset({a["name"]: data_array(a) for a in arrays})

    def phase_groups(prefix, draw_ids, grid):
        coords = {"chain": np.asarray(selected, dtype=np.uint64), "draw": np.asarray(draw_ids, dtype=np.uint64)}
        shape = (len(selected), len(draw_ids))

        def field(column, dtype=np.float64):
            return np.asarray([[r.values[column] for r in chain] for chain in grid], dtype=dtype)

        def named(fields):
            require(not set(fields) & {"chain", "draw"}, "variable conflicts with sampling dimension")
            return xr.Dataset(fields, coords=coords, attrs={"sample_dims": ["chain", "draw"]})

        posterior = {}
        for p in m["parameters"]:
            column = "parameter:" + p["name"]
            attrs = {"alea_parameter_json": json.dumps(p, ensure_ascii=False)}
            if column in m["units"]:
                attrs["alea_units"] = m["units"][column]
            posterior[p["name"]] = xr.DataArray(field(column), dims=("chain", "draw"), attrs=attrs)
        groups["/" + prefix + "posterior"] = named(posterior)
        if m["save_unconstrained"]:
            values = np.stack([field(f"unconstrained:{i}") for i in range(m["unconstrained_dimension"])], axis=-1) if m["unconstrained_dimension"] else np.empty((*shape, 0))
            # Generic q coordinates have no inferred physical-variable counterpart.
            groups["/" + prefix + "alea_unconstrained"] = xr.Dataset({"q": (("chain", "draw", "free_coordinate"), values)},
                coords={**coords, "free_coordinate": np.arange(m["unconstrained_dimension"])})
        stats, availability = {}, {}
        for source, target in STAT_NAMES.items():
            column = "stat:" + source
            present = np.asarray([[r.values[column] is not None for r in chain] for chain in grid])
            availability[target] = (("chain", "draw"), present)
            require(present.all() or not present.any(), "mixed statistic availability; inspect original records explicitly")
            if not present.any():
                continue
            first = grid[0][0].values[column]
            dtype = np.bool_ if type(first) is bool else np.uint64 if type(first) is int else np.float64
            attrs = {"alea_source_column": column}
            if source == "energy":
                attrs["description"] = "initial Hamiltonian after momentum refresh; not accepted-proposal energy"
            if source == "integration_steps":
                attrs["description"] = "complete integrator macrosteps; not assumed to be leapfrog steps"
            stats[target] = xr.DataArray(field(column, dtype), dims=("chain", "draw"), attrs=attrs)
        groups["/" + prefix + "sample_stats"] = named(stats)
        groups["/" + prefix + "sample_stats_available"] = named(availability)
        if m["predictive_units"]:
            values = np.stack([field("log_likelihood:" + u["name"]) for u in m["predictive_units"]], axis=-1)
            groups["/" + prefix + "log_likelihood"] = xr.Dataset({"likelihood": (("chain", "draw", "predictive_unit"), values)},
                coords={**coords, "predictive_unit": [u["name"] for u in m["predictive_units"]]},
                attrs={"alea_predictive_units_json": json.dumps(m["predictive_units"], ensure_ascii=False)})
        quantities = {p: (("chain", "draw"), field("prediction:" + p)) for p in m["predictions"] if p not in bound_columns}
        if quantities:
            groups["/" + prefix + "generated_quantities"] = named(quantities)
        if m["posterior_predictive"]:
            variables = {}
            for b in m["posterior_predictive"]:
                a = observed[b["variable"]]
                dims = ("chain", "draw", *(axis["name"] for axis in a["axes"]))
                values = np.stack([field("prediction:" + c) for c in b["columns"]], axis=-1) if b["columns"] else np.empty((*shape, 0))
                values = values.reshape((*shape, *(len(d["coordinates"]) for d in a["axes"]))).astype(DTYPES[a["dtype"]])
                attr = {"alea_prediction_columns_json": json.dumps(b["columns"], ensure_ascii=False)}
                if a["unit"] is not None:
                    attr["alea_units"] = a["unit"]
                variables[b["variable"]] = xr.DataArray(values, dims=dims,
                    coords={d["name"]: d["coordinates"] for d in a["axes"]}, attrs=attr)
            groups["/" + prefix + "posterior_predictive"] = named(variables)

    phase_groups("", indices, rows)
    warmup = any(c["warmup"]["rows"] for c in m["chains"] if c["id"] in selected)
    if include_warmup and warmup:
        _, draw_ids, grid = artifact.rectangular(chains=selected, phase="warmup")
        phase_groups("warmup_", draw_ids, grid)
    failures = [r for r in artifact.records if r.error is not None]
    if failures:
        groups["/analysis_failures"] = xr.Dataset({
            "chain_id": ("failure", np.asarray([r.chain for r in failures], dtype=np.uint64)),
            "draw_id": ("failure", np.asarray([r.draw for r in failures], dtype=np.uint64)),
            "phase": ("failure", [r.phase for r in failures]),
            "message": ("failure", [r.error for r in failures]),
        }, coords={"failure": np.arange(len(failures))})
    return xr.DataTree.from_dict(groups)


def write_netcdf(tree, path):
    """Delegate encoding to xarray/h5netcdf; refuse an existing output file."""
    require(not Path(path).exists(), "NetCDF output already exists")
    with Path(path).open("x+b") as handle:
        tree.to_netcdf(handle, engine="h5netcdf")


def loo_report(tree, reff=None):
    """Run pinned ArviZ PSIS-LOO and retain pointwise ELPD, Pareto k and warnings.

    This evaluates the declared units; it does not establish that their likelihoods
    or the source posterior are statistically valid. No bad Pareto k is filtered.
    """
    import warnings
    import arviz_stats as azs
    import arviz_base
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = azs.loo(tree, var_name="likelihood", pointwise=True, reff=reff)
    messages = [str(w.message) for w in caught]
    for warning in caught:
        warnings.warn(str(warning.message), warning.category, stacklevel=2)
    report = xr.Dataset({"elpd_i": result.elpd_i, "pareto_k": result.pareto_k}, attrs={
        "elpd": float(result.elpd), "se": float(result.se), "p_loo": float(result.p),
        "n_samples": int(result.n_samples), "n_data_points": int(result.n_data_points),
        "warning": int(result.warning), "good_k": float(result.good_k),
        "warning_messages_json": json.dumps(messages, ensure_ascii=False),
        "arviz_stats_version": azs.__version__, "arviz_base_version": arviz_base.__version__,
        "xarray_version": xr.__version__, "scale": "log",
        "alea_manifest_json": tree.attrs["alea_manifest_json"],
        "alea_selection_json": tree.attrs["alea_selection_json"],
    })
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("csv")
    parser.add_argument("output")
    parser.add_argument("--chains", nargs="+", type=int)
    parser.add_argument("--omit-warmup", action="store_true")
    parser.add_argument("--loo-output")
    args = parser.parse_args()
    require(not Path(args.output).exists(), "NetCDF output already exists")
    if args.loo_output:
        require(not Path(args.loo_output).exists() and Path(args.output).resolve() != Path(args.loo_output).resolve(), "LOO output exists or collides with DataTree output")
    artifact = load(args.manifest, args.csv)
    tree = to_datatree(artifact, chains=args.chains, include_warmup=not args.omit_warmup)
    report = loo_report(tree) if args.loo_output else None
    write_netcdf(tree, args.output)
    if report is not None:
        write_netcdf(report, args.loo_output)
        print(report)
