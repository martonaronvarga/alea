//! Validated, statically dispatched symmetric kick/drift macrosteps.
#![deny(missing_docs)]
mod private {
    pub trait Sealed {}
}
/// A complete palindromic, consistent separable-Hamiltonian map. Internal stages
/// are never trajectory candidates. Geometry and coefficients stay fixed during
/// a trajectory and after warmup. This is not WALNUTS's local-step controller.
pub trait Integrator: private::Sealed {
    /// Number of drifts/new fused evaluations on a successful macrostep.
    fn stages(&self) -> usize;
    /// Kick coefficient at a palindromic stage index.
    fn kick(&self, index: usize) -> f64;
    /// Drift coefficient at a palindromic stage index.
    fn drift(&self, index: usize) -> f64;
}
#[derive(Debug, Clone, Copy, Default)]
/// One-stage symmetric velocity-Verlet integrator.
pub struct Leapfrog;
impl private::Sealed for Leapfrog {}
impl Integrator for Leapfrog {
    fn stages(&self) -> usize {
        1
    }
    fn kick(&self, _: usize) -> f64 {
        0.5
    }
    fn drift(&self, _: usize) -> f64 {
        1.0
    }
}
/// `B(lambda h) A(h/2) B((1-2lambda)h) A(h/2) B(lambda h)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoStage {
    lambda: f64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("two-stage coefficient must be finite and between zero and one half")]
/// A two-stage coefficient was nonfinite or outside the supported interval.
pub struct InvalidCoefficient;
impl TwoStage {
    /// Minimum-norm two-stage coefficient from Omelyan, Mryglod and Folk.
    pub const OMF: Self = Self {
        lambda: 0.193_183_327_503_783_6,
    };
    /// BCSS two-stage coefficient, optimized over a finite oscillator interval.
    pub const BCSS: Self = Self { lambda: 0.211_781 };
    /// # Errors
    /// Rejects non-finite values or coefficients outside [0, 1/2].
    pub fn new(lambda: f64) -> Result<Self, InvalidCoefficient> {
        if !lambda.is_finite() || !(0.0..=0.5).contains(&lambda) {
            return Err(InvalidCoefficient);
        }
        Ok(Self { lambda })
    }
    /// Validated outer kick coefficient.
    pub fn lambda(self) -> f64 {
        self.lambda
    }
}
impl private::Sealed for TwoStage {}
impl Integrator for TwoStage {
    fn stages(&self) -> usize {
        2
    }
    fn kick(&self, index: usize) -> f64 {
        if index == 1 {
            1.0 - 2.0 * self.lambda
        } else {
            self.lambda
        }
    }
    fn drift(&self, _: usize) -> f64 {
        0.5
    }
}

/// A conservative, user-auditable conversion from an observed HVP spectral
/// estimate to an upper curvature bound. Power iteration alone is a lower
/// estimate, so an explicit inflation greater than one is mandatory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurvatureBound {
    upper: f64,
    inflation: f64,
}

/// Invalid curvature/stability or equal-budget configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AdvancedIntegratorError {
    #[error("curvature estimate must be finite and positive, with inflation greater than one")]
    /// Curvature estimate must be finite and positive, with inflation greater than one.
    Curvature,
    #[error("dimensionless two-stage interval must be finite and strictly between zero and four")]
    /// Dimensionless two-stage interval must be finite and strictly between zero and four.
    StabilityInterval,
    #[error("force budget must be nonzero and divisible by the integrator stage count")]
    /// Force budget must be nonzero and divisible by the integrator stage count.
    ForceBudget,
}

impl CurvatureBound {
    /// Inflate an observed absolute-curvature estimate. The result is a policy
    /// bound, not a mathematically certified global Hessian bound.
    pub fn new(estimate: f64, inflation: f64) -> Result<Self, AdvancedIntegratorError> {
        let upper = estimate * inflation;
        if !estimate.is_finite()
            || estimate <= 0.0
            || !inflation.is_finite()
            || inflation <= 1.0
            || !upper.is_finite()
        {
            return Err(AdvancedIntegratorError::Curvature);
        }
        Ok(Self { upper, inflation })
    }

    /// Inflated upper curvature estimate used by the policy.
    pub fn upper(self) -> f64 {
        self.upper
    }

    /// User-selected inflation factor applied to the observed curvature estimate.
    pub fn inflation(self) -> f64 {
        self.inflation
    }

    /// `h_bar = step * sqrt(curvature_upper)` for the oscillator model.
    pub fn dimensionless_step(self, step: f64) -> Result<f64, AdvancedIntegratorError> {
        let value = step * self.upper.sqrt();
        if value.is_finite() && value > 0.0 && value < 4.0 {
            Ok(value)
        } else {
            Err(AdvancedIntegratorError::StabilityInterval)
        }
    }
}

/// Cold-path result of the two-stage s-AIA experiment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SaiaTwoStageSelection {
    /// Selected two-stage integrator.
    pub integrator: TwoStage,
    /// Upper endpoint of the oscillator interval used for selection.
    pub dimensionless_limit: f64,
    /// Largest energy-error bound on the deterministic selection grid.
    pub worst_bound: f64,
}

/// Published Gaussian expected-energy-error bound for the two-stage family.
/// Returns infinity at or beyond a singularity/stability boundary.
pub fn two_stage_energy_bound(h: f64, b: f64) -> f64 {
    if !h.is_finite() || h < 0.0 || !b.is_finite() || !(0.0..0.5).contains(&b) {
        return f64::INFINITY;
    }
    let h2 = h * h;
    let b2 = b * b;
    let complement = 0.5 - b;
    let term = 2.0 * b2 * complement * h2 + 4.0 * b2 - 6.0 * b + 1.0;
    let denominator = 8.0 * (2.0 - b * h2) * (2.0 - complement * h2) * (1.0 - b * complement * h2);
    let numerator = h2 * h2 * term * term;
    if denominator > 0.0 && numerator.is_finite() {
        numerator / denominator
    } else {
        f64::INFINITY
    }
}

fn worst_two_stage_bound(limit: f64, b: f64) -> f64 {
    // Cold warmup work. A fixed grid makes the selection deterministic and
    // includes the requested endpoint, where stability loss is most dangerous.
    const GRID: usize = 512;
    (1..=GRID)
        .map(|index| two_stage_energy_bound(limit * index as f64 / GRID as f64, b))
        .fold(0.0, f64::max)
}

/// Select a two-stage coefficient by minimizing the published s-AIA minimax
/// bound over `[0, dimensionless_limit]`. The search is deterministic, bounded,
/// allocation-free and restricted to the ME2--VV2 fail-safe interval.
pub fn select_saia_two_stage(
    dimensionless_limit: f64,
) -> Result<SaiaTwoStageSelection, AdvancedIntegratorError> {
    if !dimensionless_limit.is_finite() || dimensionless_limit <= 0.0 || dimensionless_limit >= 4.0
    {
        return Err(AdvancedIntegratorError::StabilityInterval);
    }
    let mut left = TwoStage::OMF.lambda();
    let mut right = 0.25;
    // Golden-section minimization is cold, deterministic and requires a fixed
    // amount of work. The objective itself uses the bounded h grid above.
    const INV_PHI: f64 = 0.618_033_988_749_894_9;
    let mut c = right - INV_PHI * (right - left);
    let mut d = left + INV_PHI * (right - left);
    let mut fc = worst_two_stage_bound(dimensionless_limit, c);
    let mut fd = worst_two_stage_bound(dimensionless_limit, d);
    for _ in 0..48 {
        if fc <= fd {
            right = d;
            d = c;
            fd = fc;
            c = right - INV_PHI * (right - left);
            fc = worst_two_stage_bound(dimensionless_limit, c);
        } else {
            left = c;
            c = d;
            fc = fd;
            d = left + INV_PHI * (right - left);
            fd = worst_two_stage_bound(dimensionless_limit, d);
        }
    }
    let lambda = 0.5 * (left + right);
    Ok(SaiaTwoStageSelection {
        integrator: TwoStage::new(lambda).expect("search interval is a valid coefficient"),
        dimensionless_limit,
        worst_bound: worst_two_stage_bound(dimensionless_limit, lambda),
    })
}

/// Equal-force-budget trajectory plan. A k-stage method takes `budget/k`
/// macrosteps of length `k * reference_step`, preserving both force work and
/// nominal integration time relative to leapfrog.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForceBudgetPlan {
    /// Number of new fused target evaluations in a successful trajectory.
    pub force_budget: usize,
    /// Number of complete integrator macrosteps.
    pub macrosteps: usize,
    /// Step size of one complete macrostep.
    pub macrostep_size: f64,
}

impl ForceBudgetPlan {
    /// Builds an equal-force trajectory plan.
    ///
    /// # Errors
    /// Rejects zero or indivisible budgets and nonfinite or nonpositive step sizes.
    pub fn new<I: Integrator>(
        force_budget: usize,
        reference_step: f64,
        integrator: &I,
    ) -> Result<Self, AdvancedIntegratorError> {
        let stages = integrator.stages();
        if force_budget == 0
            || !force_budget.is_multiple_of(stages)
            || !reference_step.is_finite()
            || reference_step <= 0.0
        {
            return Err(AdvancedIntegratorError::ForceBudget);
        }
        let macrostep_size = reference_step * stages as f64;
        if !macrostep_size.is_finite() {
            return Err(AdvancedIntegratorError::ForceBudget);
        }
        Ok(Self {
            force_budget,
            macrosteps: force_budget / stages,
            macrostep_size,
        })
    }
}
