//! Synchronous streaming orchestration. Numerical algorithms live in `alea-mcmc`.
#![forbid(unsafe_code)]
pub mod diagnostics;
pub mod draws;
use alea_mcmc::MarkovChain;
use std::ops::ControlFlow;

/// Advances a bound chain, emitting each retained point and its typed transition.
/// The observer runs outside the numerical transition and may stop the run early.
/// No draws are buffered and this runner allocates nothing. Rejections are emitted.
///
/// # Errors
/// Propagates the original chain error without invoking the observer for a failed step.
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
