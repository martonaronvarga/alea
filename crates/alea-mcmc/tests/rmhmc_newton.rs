use alea_core::capability::PositionMetric;
use alea_distributions::Gaussian;
use alea_math::buffer::OwnedBuffer;
use alea_mcmc::riemannian::{ExactnessPendingRmhmc, RmhmcOptions};
use rand::{SeedableRng, rngs::SmallRng};
use std::convert::Infallible;
struct Metric;
impl PositionMetric for Metric {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        2
    }
    fn metric_and_derivatives(
        &self,
        q: &[f64],
        m: &mut [f64],
        d: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), Infallible> {
        m.fill(0.);
        d.fill(0.);
        m[0] = (0.2 * q[0]).exp();
        m[3] = (0.3 * q[1]).exp();
        d[0] = 0.2 * m[0];
        d[7] = 0.3 * m[3];
        Ok(())
    }
}
#[test]
fn tight_newton_and_full_replay_charge_work_and_keep_live_cache_coherent() {
    let target = Gaussian::new(2);
    let metric = Metric;
    let mut rng = SmallRng::seed_from_u64(1781);
    let options = RmhmcOptions::new(0.1, 4, 100, 1e-12)
        .unwrap()
        .with_newton_solver()
        .with_reverse_check(1e-8)
        .unwrap();
    let mut s = ExactnessPendingRmhmc::new(&target, &metric, OwnedBuffer::new(2), options).unwrap();
    let measured = allocation_counter::measure(|| {
        for _ in 0..100 {
            let o = s.step(&mut rng).unwrap();
            assert_eq!(o.reverse_passed, Some(true));
            assert_eq!(o.costs.reverse_steps, 4);
            assert_eq!(o.costs.dynamics_steps, 8);
            assert!(o.costs.newton_linear_solves > 0);
            assert!(o.maximum_reverse_error.unwrap() < 1e-8);
            assert!(s.point().position().iter().all(|x| x.is_finite()));
        }
    });
    assert_eq!(measured.count_total, 0);
}
