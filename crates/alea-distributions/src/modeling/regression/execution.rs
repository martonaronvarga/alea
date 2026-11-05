//! Fixed native CPU dispatch, separate from mathematical primitive signatures.
use super::{E, kernels::TILE};

/// Accumulation policy for native regression row kernels, independent of CPU
/// dispatch. This does not enable compiler fast-math or change scalar laws,
/// parameter transforms, sufficient-statistics preparation, or sampler arithmetic.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RegressionArithmetic {
    /// Compensated sums and predictor product residuals (the default).
    #[default]
    Compensated,
    /// Conventional f64 sums/products, including SIMD lane reductions. Faster
    /// execution is workload-dependent; cancellation can lose meaningful digits.
    /// Shape/support/nonfinite checks remain enabled. No bitwise replay or
    /// accuracy equivalence with the compensated policy is promised.
    Standard,
}

/// Native regression dispatch policy. This does not select a different density,
/// precision policy, generated-source backend or number of observation threads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum RegressionExecution {
    /// Cache Pulp's available CPU implementation, using scalar execution when
    /// neither a full observation tile nor a coefficient SIMD packet is present.
    /// This is structural eligibility, not timing autotuning or a fitted cutoff.
    #[default]
    Auto,
    /// Execute row kernels using Pulp's scalar implementation. Gaussian fixed-data
    /// compression is independent of this CPU policy and uses scalar arithmetic.
    /// Useful for diagnostics; not a promise of row-wise or cross-ISA bit equality.
    Scalar,
}

/// Immutable, inspectable plan for one native regression binding.
///
/// The plan uses explicitly selected arithmetic and one observation worker. Packing
/// and CPU selection happen at construction, never per observation. The dispatch
/// branch remains once per evaluation. Plans are not serialized replay identities
/// or cost-model predictions, and do not describe separately generated Rust source.
/// For Gaussian regression this describes the row-kernel fallback, not eligibility
/// or storage of its separate square-root sufficient-statistics preparation.
#[derive(Debug, Clone, Copy)]
pub struct RegressionPlan {
    pub(super) arithmetic: RegressionArithmetic,
    execution: RegressionExecution,
    arch: pulp::Arch,
    lanes: usize,
    rows: usize,
    columns: usize,
    packed_design_bytes: usize,
    broadcast: bool,
    scalar_for_shape: bool,
}

impl RegressionPlan {
    pub(super) fn new(
        execution: RegressionExecution,
        rows: usize,
        columns: usize,
        packed_design_bytes: usize,
        broadcast: bool,
    ) -> Self {
        let mut arch = match execution {
            RegressionExecution::Auto => pulp::Arch::new(),
            RegressionExecution::Scalar => pulp::Arch::Scalar,
        };
        let lanes = arch.dispatch(Lanes);
        let scalar_for_shape =
            execution == RegressionExecution::Auto && no_vector_work(rows, columns, lanes);
        if scalar_for_shape {
            arch = pulp::Arch::Scalar;
        }
        Self {
            arithmetic: RegressionArithmetic::Compensated,
            execution,
            arch,
            lanes: if scalar_for_shape { 1 } else { lanes },
            rows,
            columns,
            packed_design_bytes,
            broadcast,
            scalar_for_shape,
        }
    }
    pub(super) fn dispatch<Op: pulp::WithSimd<Output = Result<f64, E>>>(
        &self,
        op: Op,
    ) -> Result<f64, E> {
        self.arch.dispatch(op)
    }
    /// Requested policy, preserved when a descriptor is rebound to new data.
    pub fn execution(&self) -> RegressionExecution {
        self.execution
    }
    /// Requested row-kernel arithmetic; independent of CPU provider selection.
    pub fn arithmetic(&self) -> RegressionArithmetic {
        self.arithmetic
    }
    /// Number of observations in this binding.
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Number of coefficient columns, excluding any separate scale parameter.
    pub fn columns(&self) -> usize {
        self.columns
    }
    /// Selected implementation's f64 lane count, not a SIMD-utilization measure.
    /// Kernels retain scalar tails and representation-limit fallbacks. Widths
    /// greater than the four-row tile can use scalar predictor fallback work.
    pub fn lanes(&self) -> usize {
        self.lanes
    }
    /// Whether automatic selection bypassed SIMD because no full observation
    /// tile or coefficient packet exists. Explicit scalar requests return false.
    pub fn is_scalar_for_shape(&self) -> bool {
        self.scalar_for_shape
    }
    /// Fixed observation tile size, independent of selected CPU lane count.
    pub fn tile_rows(&self) -> usize {
        TILE
    }
    /// Additional packed-design element storage, excluding allocator overhead,
    /// original data, family constants and execution-local score buffers.
    pub fn packed_design_bytes(&self) -> usize {
        self.packed_design_bytes
    }
    /// Whether the design is a single unit column with all-zero offsets.
    /// The normal kernel can additionally reduce its shared mean score by tile.
    pub fn is_broadcast(&self) -> bool {
        self.broadcast
    }
    /// Observation workers per evaluation. Chain scheduling is separate.
    pub fn workers(&self) -> usize {
        1
    }
}

fn no_vector_work(rows: usize, columns: usize, lanes: usize) -> bool {
    // Do not change wider-than-tile normal fallbacks: those have a separately
    // qualified divided path. This rule covers only tail-only kernels with no
    // vector coefficient scores, preserving their existing scalar arithmetic.
    lanes > 1 && lanes <= TILE && rows < TILE && columns < lanes
}

struct Lanes;
impl pulp::WithSimd for Lanes {
    type Output = usize;
    fn with_simd<S: pulp::Simd>(self, _: S) -> usize {
        size_of::<S::f64s>() / size_of::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eligibility_uses_both_independent_axes_and_excludes_wider_fallbacks() {
        for lanes in [1, 2, 4, 8, 16] {
            for rows in 1..=8 {
                for columns in 0..=16 {
                    assert_eq!(
                        no_vector_work(rows, columns, lanes),
                        matches!(lanes, 2 | 4) && rows < TILE && columns < lanes
                    );
                }
            }
        }
    }

    #[test]
    fn reported_selection_matches_available_packets_and_explicit_scalar_is_distinct() {
        let lanes = pulp::Arch::new().dispatch(Lanes);
        for rows in 1..=8 {
            for columns in 0..=16 {
                let plan = RegressionPlan::new(RegressionExecution::Auto, rows, columns, 0, false);
                let eligible = no_vector_work(rows, columns, lanes);
                assert_eq!(plan.is_scalar_for_shape(), eligible);
                assert_eq!(plan.execution(), RegressionExecution::Auto);
                assert_eq!(plan.lanes(), if eligible { 1 } else { lanes });
                let scalar =
                    RegressionPlan::new(RegressionExecution::Scalar, rows, columns, 0, false);
                assert!(!scalar.is_scalar_for_shape());
                assert_eq!(scalar.lanes(), 1);
            }
        }
    }
}
