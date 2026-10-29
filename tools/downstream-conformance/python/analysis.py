"""Strict reader for Alea's sampling CSV/JSON contract. Uses only the standard library.

load() retains all records after validation. rectangular() requires an explicit
completed, failure-free common grid; it never pads, truncates or drops chains.
"""
from __future__ import annotations
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

STATS = ["stat:accepted", "stat:acceptance_probability", "stat:energy", "stat:divergent",
         "stat:depth_saturated", "stat:integration_steps", "stat:tree_depth"]
OPTIONAL = set(STATS[2:])
BOOLS = {"stat:accepted", "stat:divergent", "stat:depth_saturated"}
INTS = {"stat:integration_steps", "stat:tree_depth"}
UINT = re.compile(r"(?:0|[1-9][0-9]*)\Z")
NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")
U64_MAX = 2**64 - 1


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def keys(value, expected):
    require(type(value) is dict and set(value) == set(expected), "unknown or missing schema fields")


def uint(value):
    require(type(value) is int and 0 <= value <= U64_MAX, "invalid unsigned integer")
    return value


def names(values, nonempty=False):
    require(type(values) is list and (not nonempty or values), "invalid name list")
    require(all(type(s) is str and s for s in values), "empty or non-string name")
    require(len(set(values)) == len(values), "duplicate names")


def unique_object(pairs):
    result = {}
    for k, v in pairs:
        require(k not in result, "duplicate JSON key")
        result[k] = v
    return result


def reject_constant(value):
    raise ValueError(f"nonfinite JSON constant {value}")


def validate_manifest(m):
    keys(m, ["format", "schema_version", "kind", "csv", "provenance", "parameters",
             "unconstrained_dimension", "save_unconstrained", "predictions", "units",
             "predictive_units", "chains", "observed_data", "constant_data", "posterior_predictive"])
    require(m["format"] == "alea-analysis" and type(m["schema_version"]) is int
            and m["schema_version"] == 2 and m["kind"] == "sampling", "unsupported analysis format")
    keys(m["csv"], ["columns", "sha256"])
    require(type(m["csv"]["sha256"]) is str and re.fullmatch("[0-9a-f]{64}", m["csv"]["sha256"]), "invalid CSV digest")
    p = m["provenance"]
    keys(p, ["model", "data", "source", "method", "settings"])
    require(all(type(p[k]) is str and p[k] for k in ("model", "data", "source", "method")), "empty provenance")
    require(type(p["settings"]) is dict and all(type(k) is str and k and type(v) is str for k, v in p["settings"].items()), "invalid settings")
    require(type(m["parameters"]) is list and m["parameters"], "empty parameter schema")
    for parameter in m["parameters"]:
        keys(parameter, ["name", "dimension", "coordinate", "block"])
        require(all(parameter[k] is None or type(parameter[k]) is str for k in ("dimension", "coordinate", "block")), "invalid coordinate metadata")
    names([p["name"] for p in m["parameters"]], True)
    dimension = uint(m["unconstrained_dimension"])
    require(type(m["save_unconstrained"]) is bool, "invalid free-coordinate flag")
    names(m["predictions"])
    require(type(m["predictive_units"]) is list, "invalid predictive units")
    factors, observations = set(), set()
    for unit in m["predictive_units"]:
        keys(unit, ["name", "factors", "observations", "conditioned_on", "integrated_out"])
        for field in ("factors", "observations", "conditioned_on", "integrated_out"):
            names(unit[field], field in ("factors", "observations"))
        require(not set(unit["conditioned_on"]) & set(unit["integrated_out"]), "contradictory conditioning")
        require(not factors.intersection(unit["factors"]), "overlapping factors")
        require(not observations.intersection(unit["observations"]), "overlapping observations")
        factors.update(unit["factors"])
        observations.update(unit["observations"])
    names([u["name"] for u in m["predictive_units"]])
    columns = ["chain", "phase", "draw", "kind", "error"]
    columns += ["parameter:" + p["name"] for p in m["parameters"]]
    if m["save_unconstrained"]:
        columns += [f"unconstrained:{i}" for i in range(dimension)]
    columns += STATS
    columns += ["log_likelihood:" + u["name"] for u in m["predictive_units"]]
    columns += ["prediction:" + p for p in m["predictions"]]
    require(m["csv"]["columns"] == columns, "columns differ from named schema")
    require(type(m["units"]) is dict and all(k in columns[5:] and type(v) is str and v for k, v in m["units"].items()), "invalid physical units")
    require(type(m["chains"]) is list and m["chains"], "empty chain plan")
    ids = set()
    for chain in m["chains"]:
        keys(chain, ["id", "status", "reason", "warmup", "sampling"])
        ident = uint(chain["id"])
        require(ident not in ids, "duplicate chain identity")
        ids.add(ident)
        require(chain["status"] in ("completed", "failed", "interrupted"), "invalid chain status")
        require((chain["status"] == "completed" and chain["reason"] is None)
                or (chain["status"] != "completed" and type(chain["reason"]) is str and chain["reason"]), "missing or unexpected stopping reason")
        for phase in ("warmup", "sampling"):
            count = chain[phase]
            keys(count, ["first", "last", "rows", "failures"])
            n, failures = uint(count["rows"]), uint(count["failures"])
            require(failures <= n, "invalid failure count")
            if n == 0:
                require(count["first"] is None and count["last"] is None, "nonempty empty-chain range")
            else:
                first, last = uint(count["first"]), uint(count["last"])
                require(last >= first and last - first + 1 == n, "inconsistent declared draw range")
    validate_data_bindings(m)
    return columns


def validate_data_bindings(m):
    """Reject ambiguous axes, lost integer types, and incomplete replicated arrays."""
    require(type(m["observed_data"]) is list and type(m["constant_data"]) is list
            and type(m["posterior_predictive"]) is list, "invalid data bindings")
    axes = {}
    observed = {}
    observation_ids = set()

    def array(a):
        keys(a, ["name", "axes", "dtype", "values", "unit"])
        require(type(a["name"]) is str and a["name"], "invalid array name")
        require(a["unit"] is None or (type(a["unit"]) is str and a["unit"]), "invalid array unit")
        require(type(a["axes"]) is list and type(a["values"]) is list, "invalid array layout")
        names([d.get("name") for d in a["axes"] if type(d) is dict])
        size = 1
        local = set()
        for d in a["axes"]:
            keys(d, ["name", "coordinates"])
            name = d["name"]
            require(type(name) is str and name and name not in ("chain", "draw", "sample", "pred_id")
                    and name != a["name"] and name not in local, "invalid or reserved data axis")
            local.add(name)
            names(d["coordinates"])
            require(name not in axes or axes[name] == d["coordinates"], "inconsistent shared data axis")
            axes[name] = d["coordinates"]
            size *= len(d["coordinates"])
        require(size == len(a["values"]), "bound array shape mismatch")
        dtype = a["dtype"]
        require(dtype in ("float64", "int64", "uint64", "bool", "string"), "unsupported data dtype")
        for value in a["values"]:
            if dtype == "float64":
                require(type(value) in (int, float) and math.isfinite(float(value)), "nonfinite real data")
            elif dtype == "int64":
                require(type(value) is int and -(2**63) <= value < 2**63, "invalid int64 data")
            elif dtype == "uint64":
                uint(value)
            elif dtype == "bool":
                require(type(value) is bool, "invalid Boolean data")
            else:
                require(type(value) is str, "invalid text data")

    for bound in m["observed_data"]:
        keys(bound, ["array", "observation_dimension"])
        a = bound["array"]
        array(a)
        require(a["name"] not in observed, "duplicate observed variable")
        observed[a["name"]] = a
        dim = bound["observation_dimension"]
        matches = [d for d in a["axes"] if d["name"] == dim]
        require(type(dim) is str and len(matches) == 1, "missing observation axis")
        observation_ids.update(matches[0]["coordinates"])
    constant_names = set()
    for a in m["constant_data"]:
        array(a)
        require(a["name"] not in constant_names, "duplicate constant variable")
        constant_names.add(a["name"])
    for arrays in (list(observed.values()), m["constant_data"]):
        dimensions = {d["name"] for a in arrays for d in a["axes"]}
        require(not {a["name"] for a in arrays} & dimensions, "variable conflicts with data dimension")
    if m["observed_data"] and m["predictive_units"]:
        expected = {o for unit in m["predictive_units"] for o in unit["observations"]}
        require(observation_ids == expected, "observed IDs differ from predictive units")
    variables, columns = set(), set()
    for bound in m["posterior_predictive"]:
        keys(bound, ["variable", "columns"])
        name = bound["variable"]
        require(type(name) is str and name in observed and name not in variables, "invalid predictive counterpart")
        variables.add(name)
        names(bound["columns"])
        require(len(bound["columns"]) == len(observed[name]["values"])
                and observed[name]["dtype"] != "string", "incomplete or nonnumeric predictive binding")
        require(not columns.intersection(bound["columns"]) and set(bound["columns"]) <= set(m["predictions"]), "unknown or repeated prediction column")
        columns.update(bound["columns"])
        for column in bound["columns"]:
            unit = m["units"].get("prediction:" + column)
            require(unit is None or observed[name]["unit"] is None or unit == observed[name]["unit"], "predictive physical units differ")


@dataclass(frozen=True)
class Record:
    chain: int
    phase: str
    draw: int
    error: str | None
    values: dict


@dataclass(frozen=True)
class Artifact:
    manifest: dict
    records: tuple[Record, ...]

    def rectangular(self, chains=None, phase="sampling"):
        """Return (chain IDs, absolute draw IDs, rows) with no implicit filtering.

        Explicitly selecting chains is permitted; selected chains must be complete
        with no failed rows and identical nonempty draw-index coordinates.
        """
        require(phase in ("warmup", "sampling"), "invalid phase")
        plan = {c["id"]: c for c in self.manifest["chains"]}
        selected = list(plan) if chains is None else list(chains)
        require(selected and all(type(c) is int and c in plan for c in selected)
                and len(set(selected)) == len(selected), "invalid chain selection")
        rows, indices = [], None
        for chain in selected:
            require(plan[chain]["status"] == "completed", "selected chain is incomplete")
            current = [r for r in self.records if r.chain == chain and r.phase == phase]
            require(current and all(r.error is None for r in current), "selected grid is empty or contains failures")
            coords = [r.draw for r in current]
            require(indices is None or coords == indices, "unequal draw coordinates")
            indices = coords
            rows.append(current)
        return selected, indices, rows


def integer_field(value):
    require(bool(UINT.fullmatch(value)), "invalid integer field")
    return uint(int(value))


def numeric_field(name, text):
    if text == "" and name in OPTIONAL:
        return None
    if name in BOOLS:
        require(text in ("0", "1"), "invalid Boolean statistic")
        return text == "1"
    if name in INTS:
        return integer_field(text)
    require(bool(NUMBER.fullmatch(text)), "invalid numeric field")
    value = float(text)
    require(math.isfinite(value), "nonfinite numeric field")
    if name == "stat:acceptance_probability":
        require(0 <= value <= 1, "invalid acceptance probability")
    return value


def load(manifest_path, csv_path):
    """Validate and materialize all records. No dependencies or silent repairs."""
    with Path(manifest_path).open(encoding="utf-8") as handle:
        manifest = json.load(handle, object_pairs_hook=unique_object, parse_constant=reject_constant)
    columns = validate_manifest(manifest)
    digest = hashlib.sha256()
    # Hash and parse the same open file. Exact bytes include UTF-8 and CSV CRLF.
    with Path(csv_path).open("rb") as raw:
        for block in iter(lambda: raw.read(65536), b""):
            digest.update(block)
        require(digest.hexdigest() == manifest["csv"]["sha256"], "CSV digest mismatch")
        raw.seek(0)
        import io
        text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
        reader = csv.reader(text, strict=True)
        require(next(reader, None) == columns, "CSV header mismatch")
        plan = {c["id"]: c for c in manifest["chains"]}
        seen = {(c, phase): [] for c in plan for phase in ("warmup", "sampling")}
        records = []
        observed = {b["array"]["name"]: b["array"] for b in manifest["observed_data"]}
        predictive_types = {"prediction:" + c: observed[b["variable"]]["dtype"]
                            for b in manifest["posterior_predictive"] for c in b["columns"]}
        for row in reader:
            require(len(row) == len(columns), "CSV row width mismatch")
            chain, phase, draw, kind, error = row[:5]
            chain, draw = integer_field(chain), integer_field(draw)
            require(chain in plan and phase in ("warmup", "sampling"), "unknown chain or phase")
            require(not (phase == "warmup" and seen[chain, "sampling"]), "warmup after sampling")
            previous = seen[chain, phase]
            require(not previous or previous[-1][0] + 1 == draw, "nonconsecutive or duplicate draw")
            require(kind in ("draw", "failure"), "unknown record kind")
            if kind == "failure":
                require(error and all(v == "" for v in row[5:]), "invalid failure payload")
                values = dict.fromkeys(columns[5:])
            else:
                require(error == "", "successful row has an error")
                values = {name: numeric_field(name, value) for name, value in zip(columns[5:], row[5:])}
                for column, dtype in predictive_types.items():
                    value = values[column]
                    if dtype == "bool":
                        require(value in (0, 1), "predictive value outside Boolean domain")
                    elif dtype == "int64":
                        require(value.is_integer() and -(2**63) <= value < 2**63, "predictive value outside int64 domain")
                    elif dtype == "uint64":
                        require(value.is_integer() and 0 <= value < 2**64, "predictive value outside uint64 domain")
            seen[chain, phase].append((draw, kind == "failure"))
            records.append(Record(chain, phase, draw, error or None, values))
        for (chain, phase), rows in seen.items():
            expected = plan[chain][phase]
            require(len(rows) == expected["rows"] and sum(r[1] for r in rows) == expected["failures"], "record counts differ from manifest")
            require((rows[0][0] if rows else None) == expected["first"] and (rows[-1][0] if rows else None) == expected["last"], "record range differs from manifest")
    return Artifact(manifest, tuple(records))


def r_payload(artifact):
    """JSON bridge for R: integer identities/count values are exact strings.

    R consumes the shared strict reader instead of duplicating a second CSV and
    schema validator. This bridge is not another native artifact schema.
    """
    manifest = json.loads(json.dumps(artifact.manifest, allow_nan=False))
    for chain in manifest["chains"]:
        chain["id"] = str(chain["id"])
        for phase in ("warmup", "sampling"):
            for key in ("first", "last"):
                if chain[phase][key] is not None:
                    chain[phase][key] = str(chain[phase][key])
    arrays = [b["array"] for b in manifest["observed_data"]] + manifest["constant_data"]
    for array in arrays:
        if array["dtype"] in ("int64", "uint64"):
            array["values"] = [str(v) for v in array["values"]]
    records = [{"chain": str(r.chain), "phase": r.phase, "draw": str(r.draw), "error": r.error,
                "values": {k: str(v) if type(v) is int else v for k, v in r.values.items()}}
               for r in artifact.records]
    return {"bridge_format": "alea-analysis-r", "bridge_version": 1,
            "manifest": manifest, "records": records}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("csv")
    parser.add_argument("--rectangular", action="store_true")
    parser.add_argument("--r-json", help="validated record bridge for the R consumer")
    args = parser.parse_args()
    result = load(args.manifest, args.csv)
    if args.rectangular:
        result.rectangular()
    if args.r_json:
        with Path(args.r_json).open("x", encoding="utf-8") as handle:
            json.dump(r_payload(result), handle, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"records": len(result.records), "chains": [c["id"] for c in result.manifest["chains"]]}))
