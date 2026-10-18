//! Streaming and independent-chain orchestration. Numerical algorithms live in `alea-mcmc`.
#![forbid(unsafe_code)]
pub mod chees;
pub mod diagnostics;
pub mod draws;
pub mod ensemble;
pub mod monitor;
pub mod output;
use alea_mcmc::MarkovChain;
use std::ops::ControlFlow;

/// Advances a bound chain, emitting each retained point and its typed transition.
/// The observer runs outside the numerical transition and may stop the run early.
/// No draws are buffered and this runner allocates nothing. Rejections are emitted.
///
/// # Errors
/// Propagates the original chain error without invoking the observer for a failed step.
///
/// # Panics
/// Panics from the observer propagate; already completed transitions remain committed.
pub fn run_chain<C, R>(
    chain: &mut C,
    steps: usize,
    rng: &mut R,
    mut observe: impl FnMut(&[f64], &C::Transition) -> ControlFlow<()>,
) -> Result<usize, C::Error>
where
    C: MarkovChain,
    R: rand::Rng + ?Sized,
{
    for i in 0..steps {
        let transition = chain.step(rng)?;
        if observe(chain.position(), &transition).is_break() {
            return Ok(i + 1);
        }
    }
    Ok(steps)
}

/// Draw collection and diagnostics for Alea's current reference inference path.
///
/// Session persistence, checkpointing, monitoring policy, and parallel orchestration
/// remain separate components until their M15 release contracts are complete.
pub mod reference {
    pub use crate::diagnostics::modern::{
        Error as ModernDiagnosticError, Summary as ModernSummary, ebfmi,
        summarize as summarize_modern,
    };
    pub use crate::diagnostics::{
        DiagnosticError, acceptance_rate, autocorrelation_ess, classical_split_rhat,
    };
    pub use crate::draws::{DrawError, Draws, collect_draws};
    pub use crate::run_chain;
}
