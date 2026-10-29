//! Named row-major arrays and explicit posterior-predictive bindings.
use super::{AnalysisError, AnalysisSpec};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};

/// A named array axis with unique, ordered coordinate identifiers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataAxis {
    /// Axis name; sampling dimension names are reserved.
    pub name: String,
    /// Ordered labels. Repeated observed values belong in the data, not IDs.
    pub coordinates: Vec<String>,
}
/// Typed, flat row-major array contents. Integer and Boolean data retain their type.
#[derive(Debug, Clone, PartialEq)]
pub enum ArrayValues {
    /// Finite binary64 real values, including signed zero.
    Real(Vec<f64>),
    /// Signed integer values, without conversion through floating point.
    Integer(Vec<i64>),
    /// Unsigned integer values, without conversion through floating point.
    Unsigned(Vec<u64>),
    /// Boolean observations or constants.
    Boolean(Vec<bool>),
    /// Text observations or constants, without implicit category encoding.
    Text(Vec<String>),
}
impl ArrayValues {
    fn len(&self) -> usize {
        match self {
            Self::Real(v) => v.len(),
            Self::Integer(v) => v.len(),
            Self::Unsigned(v) => v.len(),
            Self::Boolean(v) => v.len(),
            Self::Text(v) => v.len(),
        }
    }
    fn json(&self) -> (&'static str, Value) {
        match self {
            Self::Real(v) => ("float64", json!(v)),
            Self::Integer(v) => ("int64", json!(v)),
            Self::Unsigned(v) => ("uint64", json!(v)),
            Self::Boolean(v) => ("bool", json!(v)),
            Self::Text(v) => ("string", json!(v)),
        }
    }
}
/// A named array with explicit axes, row-major contents and optional physical unit.
/// An array with no axes is a scalar and has exactly one value.
#[derive(Debug, Clone, PartialEq)]
pub struct DataArray {
    /// Variable name within its observed/constant group.
    pub name: String,
    /// Axes in row-major order; the last index varies fastest.
    pub axes: Vec<DataAxis>,
    /// Contents whose length equals the product of axis lengths.
    pub values: ArrayValues,
    /// Physical unit, if declared.
    pub unit: Option<String>,
}
/// Observed random-variable data and the axis identifying predictive observations.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservedData {
    /// Observations, possibly multivariate or sharing IDs with another response.
    pub array: DataArray,
    /// Axis whose coordinate IDs match the predictive-unit observation IDs.
    pub observation_dimension: String,
}
/// Declares scalar generated columns to be replicated outcomes for one observed
/// variable. This is a caller contract, not proof of a correct simulation law.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PosteriorPredictive {
    /// Name of the matching observed variable, with the same axes/coordinates.
    pub variable: String,
    /// Prediction-column names in the observed array's row-major order.
    /// Must cover the entire array; omitted cells are never padded or invented.
    pub columns: Vec<String>,
}

fn validate_array<'a>(
    array: &'a DataArray,
    axes: &mut HashMap<&'a str, &'a [String]>,
) -> Result<(), AnalysisError> {
    if array.name.is_empty() || array.unit.as_ref().is_some_and(String::is_empty) {
        return Err(AnalysisError::Invalid("empty bound variable name or unit"));
    }
    let mut local = HashSet::new();
    let mut size = 1usize;
    for axis in &array.axes {
        let mut coordinates = HashSet::new();
        if axis.name.is_empty()
            || ["chain", "draw", "sample", "pred_id"].contains(&axis.name.as_str())
            || !local.insert(&axis.name)
            || axis.name == array.name
        {
            return Err(AnalysisError::Invalid(
                "invalid or reserved bound array axis",
            ));
        }
        if axis
            .coordinates
            .iter()
            .any(|c| c.is_empty() || !coordinates.insert(c))
        {
            return Err(AnalysisError::Invalid("empty or duplicate data coordinate"));
        }
        if axes
            .insert(&axis.name, &axis.coordinates)
            .is_some_and(|old| old != axis.coordinates)
        {
            return Err(AnalysisError::Invalid(
                "inconsistent coordinates for a shared axis",
            ));
        }
        size = size
            .checked_mul(axis.coordinates.len())
            .ok_or(AnalysisError::Invalid("bound array shape overflow"))?;
    }
    if size != array.values.len()
        || matches!(&array.values, ArrayValues::Real(v) if v.iter().any(|x| !x.is_finite()))
    {
        return Err(AnalysisError::Invalid(
            "bound array shape or nonfinite value",
        ));
    }
    Ok(())
}
pub(super) fn validate(spec: &AnalysisSpec) -> Result<(), AnalysisError> {
    let mut axes = HashMap::new();
    let mut observed = HashMap::new();
    let mut observation_ids = HashSet::new();
    for bound in &spec.observed_data {
        validate_array(&bound.array, &mut axes)?;
        if observed.insert(bound.array.name.as_str(), bound).is_some() {
            return Err(AnalysisError::Invalid("duplicate observed variable"));
        }
        let axis = bound
            .array
            .axes
            .iter()
            .find(|a| a.name == bound.observation_dimension)
            .ok_or(AnalysisError::Invalid("missing observation axis"))?;
        observation_ids.extend(axis.coordinates.iter().map(String::as_str));
    }
    let mut constants = HashSet::new();
    for array in &spec.constant_data {
        validate_array(array, &mut axes)?;
        if !constants.insert(array.name.as_str()) {
            return Err(AnalysisError::Invalid("duplicate constant variable"));
        }
    }
    for arrays in [
        spec.observed_data
            .iter()
            .map(|a| &a.array)
            .collect::<Vec<_>>(),
        spec.constant_data.iter().collect(),
    ] {
        let axis_names: HashSet<_> = arrays
            .iter()
            .flat_map(|a| a.axes.iter().map(|d| &d.name))
            .collect();
        if arrays.iter().any(|a| axis_names.contains(&a.name)) {
            return Err(AnalysisError::Invalid(
                "variable name conflicts with a data axis",
            ));
        }
    }
    if let Some(likelihood) = &spec.likelihood
        && !spec.observed_data.is_empty()
    {
        let expected: HashSet<_> = likelihood
            .units()
            .iter()
            .flat_map(|u| u.observations.iter().map(String::as_str))
            .collect();
        if expected != observation_ids {
            return Err(AnalysisError::Invalid(
                "observed IDs differ from predictive-unit observations",
            ));
        }
    }
    let mut variables = HashSet::new();
    let mut columns = HashSet::new();
    for binding in &spec.posterior_predictive {
        let array = &observed
            .get(binding.variable.as_str())
            .ok_or(AnalysisError::Invalid(
                "predictive variable has no observed counterpart",
            ))?
            .array;
        if !variables.insert(&binding.variable)
            || binding.columns.len() != array.values.len()
            || matches!(array.values, ArrayValues::Text(_))
        {
            return Err(AnalysisError::Invalid(
                "duplicate, incomplete or nonnumeric predictive binding",
            ));
        }
        for column in &binding.columns {
            if !spec.predictions.contains(column) || !columns.insert(column) {
                return Err(AnalysisError::Invalid(
                    "unknown or repeated predictive column",
                ));
            }
            if let Some(unit) = &array.unit
                && spec
                    .units
                    .get(&format!("prediction:{column}"))
                    .is_some_and(|u| u != unit)
            {
                return Err(AnalysisError::Invalid(
                    "predictive and observed units differ",
                ));
            }
        }
    }
    Ok(())
}
pub(super) fn array_json(array: &DataArray) -> Value {
    let (dtype, values) = array.values.json();
    let axes: Vec<_> = array
        .axes
        .iter()
        .map(|a| json!({"name": a.name, "coordinates": a.coordinates}))
        .collect();
    json!({"name": array.name, "axes": axes, "dtype": dtype, "values": values, "unit": array.unit})
}

#[derive(Clone, Copy)]
pub(super) enum PredictionDomain {
    Real,
    Integer,
    Unsigned,
    Boolean,
}
impl PredictionDomain {
    pub(super) fn contains(self, x: f64) -> bool {
        match self {
            Self::Real => x.is_finite(),
            Self::Integer => {
                x.fract() == 0.
                    && (-9_223_372_036_854_775_808. ..9_223_372_036_854_775_808.).contains(&x)
            }
            Self::Unsigned => x.fract() == 0. && (0. ..18_446_744_073_709_551_616.).contains(&x),
            Self::Boolean => x == 0. || x == 1.,
        }
    }
}
pub(super) fn prediction_domains(spec: &AnalysisSpec) -> Vec<PredictionDomain> {
    let columns: HashMap<_, _> = spec
        .predictions
        .iter()
        .enumerate()
        .map(|(i, c)| (c, i))
        .collect();
    let observed: HashMap<_, _> = spec
        .observed_data
        .iter()
        .map(|o| (&o.array.name, &o.array))
        .collect();
    let mut domains = vec![PredictionDomain::Real; spec.predictions.len()];
    for binding in &spec.posterior_predictive {
        let domain = match observed[&binding.variable].values {
            ArrayValues::Real(_) => PredictionDomain::Real,
            ArrayValues::Integer(_) => PredictionDomain::Integer,
            ArrayValues::Unsigned(_) => PredictionDomain::Unsigned,
            ArrayValues::Boolean(_) => PredictionDomain::Boolean,
            ArrayValues::Text(_) => unreachable!("text bindings rejected during validation"),
        };
        for column in &binding.columns {
            domains[columns[column]] = domain;
        }
    }
    domains
}
