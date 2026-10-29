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
             "predictive_units", "chains"])
    require(m["format"] == "alea-analysis" and type(m["schema_version"]) is int
            and m["schema_version"] == 1 and m["kind"] == "sampling", "unsupported analysis format")
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
    return columns


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
            seen[chain, phase].append((draw, kind == "failure"))
            records.append(Record(chain, phase, draw, error or None, values))
        for (chain, phase), rows in seen.items():
            expected = plan[chain][phase]
            require(len(rows) == expected["rows"] and sum(r[1] for r in rows) == expected["failures"], "record counts differ from manifest")
            require((rows[0][0] if rows else None) == expected["first"] and (rows[-1][0] if rows else None) == expected["last"], "record range differs from manifest")
    return Artifact(manifest, tuple(records))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("csv")
    parser.add_argument("--rectangular", action="store_true")
    args = parser.parse_args()
    result = load(args.manifest, args.csv)
    if args.rectangular:
        result.rectangular()
    print(json.dumps({"records": len(result.records), "chains": [c["id"] for c in result.manifest["chains"]]}))
