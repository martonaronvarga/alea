//! Benchmark observers, separate from Alea's production evaluation paths.
use alea::target::{LogDensityGradient, TargetSpace};
use std::{
    cell::Cell,
    time::{Duration, Instant},
};

/// Borrowed target observer. `TIMED=false` counts calls without reading a clock.
/// Returned errors and numerical outputs are forwarded unchanged. Elapsed time
/// covers returned backend calls, including errors, but excludes observer bookkeeping
/// and unwinding calls. This is wall time, including scheduling delays, not CPU time.
pub struct EvaluationObserver<'a, T: ?Sized, const TIMED: bool> {
    target: &'a T,
    calls: Cell<usize>,
    elapsed: Cell<Duration>,
}
impl<'a, T: ?Sized, const TIMED: bool> EvaluationObserver<'a, T, TIMED> {
    /// Starts an independent observation interval without evaluating the target.
    pub fn new(target: &'a T) -> Self {
        Self {
            target,
            calls: Cell::new(0),
            elapsed: Cell::new(Duration::ZERO),
        }
    }
    /// Number of calls that returned, including typed errors.
    pub fn calls(&self) -> usize {
        self.calls.get()
    }
    /// Sum of returned-call wall durations; absent for count-only observers.
    pub fn elapsed(&self) -> Option<Duration> {
        TIMED.then(|| self.elapsed.get())
    }
}
impl<T: TargetSpace + ?Sized, const TIMED: bool> TargetSpace for EvaluationObserver<'_, T, TIMED> {
    type Error = T::Error;
    fn dimension(&self) -> usize {
        self.target.dimension()
    }
}
impl<T: LogDensityGradient + ?Sized, const TIMED: bool> LogDensityGradient
    for EvaluationObserver<'_, T, TIMED>
{
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        let started = TIMED.then(Instant::now);
        let result = self.target.logp_grad(q, g);
        if let Some(started) = started {
            self.elapsed.set(self.elapsed.get() + started.elapsed());
        }
        self.calls.set(self.calls.get() + 1);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Target;
    impl TargetSpace for Target {
        type Error = std::io::Error;
        fn dimension(&self) -> usize {
            1
        }
    }
    impl LogDensityGradient for Target {
        fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
            g[0] = -q[0];
            if q[0] < 0. {
                return Err(std::io::Error::other("probe"));
            }
            Ok(-0.5 * q[0] * q[0])
        }
    }
    fn check<const TIMED: bool>() {
        let observer = EvaluationObserver::<_, TIMED>::new(&Target);
        assert_eq!(observer.dimension(), 1);
        assert_eq!(observer.calls(), 0);
        let mut g = [f64::NAN];
        assert_eq!(
            observer.logp_grad(&[2.], &mut g).unwrap().to_bits(),
            (-2_f64).to_bits()
        );
        assert_eq!(g, [-2.]);
        assert_eq!(
            observer.logp_grad(&[-1.], &mut g).unwrap_err().to_string(),
            "probe"
        );
        assert_eq!(g, [1.]);
        assert_eq!(observer.calls(), 2);
        assert_eq!(observer.elapsed().is_some(), TIMED);
    }
    #[test]
    fn observation_preserves_outputs_and_concrete_errors() {
        check::<false>();
        check::<true>();
    }
}
