//! Cold metadata validation and explicit version-one wire representation.
use super::{
    AnalysisError, AnalysisSpec, ChainCount, ChainId, ChainStatus, PhaseCount, SCHEMA_VERSION,
};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashSet};

pub(super) const STATS: [&str; 7] = [
    "stat:accepted",
    "stat:acceptance_probability",
    "stat:energy",
    "stat:divergent",
    "stat:depth_saturated",
    "stat:integration_steps",
    "stat:tree_depth",
];
pub(super) fn validate(spec: &AnalysisSpec) -> Result<Vec<String>, AnalysisError> {
    let p = &spec.provenance;
    if [&p.source, &p.model, &p.data, &p.method]
        .iter()
        .any(|s| s.is_empty())
        || p.settings.keys().any(String::is_empty)
    {
        return Err(AnalysisError::Invalid(
            "empty provenance identity or setting key",
        ));
    }
    let mut chains = HashSet::new();
    if spec.chains.is_empty() || spec.chains.iter().any(|id| !chains.insert(id.0)) {
        return Err(AnalysisError::Invalid("empty or duplicate planned chains"));
    }
    if let Some(likelihood) = &spec.likelihood
        && (likelihood.model().layout() != spec.parameters.layout()
            || likelihood.model().coordinates().as_ref() != spec.parameters.coordinates())
    {
        return Err(AnalysisError::Invalid(
            "likelihood and parameter schemas differ",
        ));
    }
    let mut names = HashSet::new();
    if spec
        .predictions
        .iter()
        .any(|name| name.is_empty() || !names.insert(name))
    {
        return Err(AnalysisError::Invalid("empty or duplicate prediction name"));
    }
    let mut columns: Vec<String> = ["chain", "phase", "draw", "kind", "error"]
        .map(String::from)
        .into();
    columns.extend(
        spec.parameters
            .coordinates()
            .iter()
            .map(|p| format!("parameter:{}", p.name)),
    );
    if spec.save_unconstrained {
        columns.extend(
            (0..spec.parameters.layout().unconstrained_dimension())
                .map(|i| format!("unconstrained:{i}")),
        );
    }
    columns.extend(STATS.map(String::from));
    if let Some(likelihood) = &spec.likelihood {
        columns.extend(
            likelihood
                .units()
                .iter()
                .map(|u| format!("log_likelihood:{}", u.name)),
        );
    }
    columns.extend(
        spec.predictions
            .iter()
            .map(|name| format!("prediction:{name}")),
    );
    if spec
        .units
        .iter()
        .any(|(key, unit)| !columns[5..].contains(key) || unit.is_empty())
    {
        return Err(AnalysisError::Invalid(
            "unknown column or empty physical unit",
        ));
    }
    Ok(columns)
}
pub(super) fn statuses<'a>(
    spec: &AnalysisSpec,
    statuses: &'a [(ChainId, ChainStatus)],
) -> Result<BTreeMap<u64, &'a ChainStatus>, AnalysisError> {
    let mut result = BTreeMap::new();
    for (id, status) in statuses {
        if !spec.chains.contains(id) || result.insert(id.0, status).is_some() {
            return Err(AnalysisError::Invalid(
                "unplanned or duplicate chain outcome",
            ));
        }
        if matches!(status, ChainStatus::Failed(s) | ChainStatus::Interrupted(s) if s.is_empty()) {
            return Err(AnalysisError::Invalid("empty chain stopping reason"));
        }
    }
    if result.len() != spec.chains.len() {
        return Err(AnalysisError::Invalid("missing chain outcome"));
    }
    Ok(result)
}
fn count(value: &PhaseCount) -> Value {
    json!({"first": value.first, "last": value.last, "rows": value.rows, "failures": value.failures})
}
pub(super) fn manifest(
    spec: &AnalysisSpec,
    columns: &[String],
    counts: &BTreeMap<u64, ChainCount>,
    statuses: &BTreeMap<u64, &ChainStatus>,
    digest: &str,
) -> Value {
    let p = &spec.provenance;
    let parameters: Vec<_> = spec.parameters.coordinates().iter().map(|p| json!({"name": p.name, "dimension": p.dimension, "coordinate": p.coordinate, "block": p.block})).collect();
    let predictive_units: Vec<_> = spec.likelihood.as_ref().map(|p| p.units()).unwrap_or(&[]).iter().map(|u| json!({"name": u.name, "factors": u.factors, "observations": u.observations, "conditioned_on": u.conditioned_on, "integrated_out": u.integrated_out})).collect();
    let chains: Vec<_> = spec.chains.iter().map(|id| {
        let (status, reason) = match statuses[&id.0] { ChainStatus::Completed => ("completed", None), ChainStatus::Failed(s) => ("failed", Some(s)), ChainStatus::Interrupted(s) => ("interrupted", Some(s)) };
        let c = &counts[&id.0];
        json!({"id": id.0, "status": status, "reason": reason, "warmup": count(&c.warmup), "sampling": count(&c.sampling)})
    }).collect();
    json!({"format": "alea-analysis", "schema_version": SCHEMA_VERSION, "kind": "sampling",
        "csv": {"sha256": digest, "columns": columns},
        "provenance": {"model": p.model, "data": p.data, "source": p.source, "method": p.method, "settings": p.settings},
        "parameters": parameters,
        "unconstrained_dimension": spec.parameters.layout().unconstrained_dimension(),
        "save_unconstrained": spec.save_unconstrained,
        "predictions": spec.predictions, "units": spec.units, "predictive_units": predictive_units, "chains": chains})
}
