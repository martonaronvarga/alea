use alea_mcmc::MarkovChain;
use alea_runtime::run_chain;
use rand::{SeedableRng, rngs::SmallRng};
use std::ops::ControlFlow;

#[derive(Debug, thiserror::Error)]
#[error("test failure")]
struct Failure;
struct Chain {
    q: [f64; 1],
    fail_at: usize,
    steps: usize,
}
impl MarkovChain for Chain {
    type Error = Failure;
    type Transition = usize;
    fn position(&self) -> &[f64] {
        &self.q
    }
    fn step<R: rand::Rng + ?Sized>(&mut self, _: &mut R) -> Result<usize, Failure> {
        if self.steps == self.fail_at {
            return Err(Failure);
        }
        self.steps += 1;
        self.q[0] = self.steps as f64;
        Ok(self.steps)
    }
}

#[test]
fn zero_steps_cancellation_and_error_observer_order() {
    let mut chain = Chain {
        q: [0.0],
        fail_at: 3,
        steps: 0,
    };
    let mut rng = SmallRng::seed_from_u64(0);
    assert_eq!(
        run_chain(&mut chain, 0, &mut rng, |_, _| panic!("no draw")).unwrap(),
        0
    );
    let mut observed = 0;
    assert_eq!(
        run_chain(&mut chain, 10, &mut rng, |q, &step| {
            observed += 1;
            assert_eq!(q, &[step as f64]);
            ControlFlow::Break(())
        })
        .unwrap(),
        1
    );
    assert!(
        run_chain(&mut chain, 10, &mut rng, |_, _| {
            observed += 1;
            ControlFlow::Continue(())
        })
        .is_err()
    );
    assert_eq!(observed, 3);
    assert_eq!(chain.position(), &[3.0]);
}

#[test]
fn observer_panic_leaves_the_completed_transition_intact() {
    let mut chain = Chain {
        q: [0.0],
        fail_at: 3,
        steps: 0,
    };
    let mut rng = SmallRng::seed_from_u64(0);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_chain(
            &mut chain,
            10,
            &mut rng,
            |_, _| panic!("observer")
        )))
        .is_err()
    );
    assert_eq!(chain.position(), &[1.0]);
    assert_eq!(
        run_chain(&mut chain, 2, &mut rng, |_, _| ControlFlow::Continue(())).unwrap(),
        2
    );
}

#[test]
#[cfg(not(miri))]
fn real_hmc_stream_includes_rejections_and_allocates_nothing() {
    use alea_distributions::Gaussian;
    use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
    use alea_mcmc::{Hmc, HmcOptions};
    let target = Gaussian::new(4);
    let mut chain = Hmc::new(
        &target,
        OwnedBuffer::new(4),
        IdentityMetric::new(4),
        HmcOptions::new(100.0, 8).unwrap(),
    )
    .unwrap();
    let mut rng = SmallRng::seed_from_u64(42);
    let mut draws = 0;
    let allocations = allocation_counter::measure(|| {
        assert_eq!(
            run_chain(&mut chain, 32, &mut rng, |q, info| {
                draws += 1;
                assert!(!info.accepted);
                assert_eq!(q, &[0.0; 4]);
                ControlFlow::Continue(())
            })
            .unwrap(),
            32
        );
    });
    assert_eq!(draws, 32);
    assert_eq!(allocations.count_total, 0);
    assert_eq!(allocations.bytes_total, 0);
}
