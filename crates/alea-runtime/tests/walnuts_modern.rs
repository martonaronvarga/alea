use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::walnuts::{Walnuts, WalnutsOptions};
use alea_runtime::diagnostics::modern;
use rand::{SeedableRng, rngs::SmallRng};

#[path = "../../alea-mcmc/tests/support/targets.rs"]
mod targets;

fn buffer(values: &[f64]) -> OwnedBuffer {
    OwnedBuffer::from_fn(values.len(), |index| values[index])
}

#[test]
fn walnuts_four_target_rank_bulk_and_tail_diagnostics_pass() {
    for target in [
        targets::Target::Correlated,
        targets::Target::Logistic,
        targets::Target::Banana,
        targets::Target::Funnel,
    ] {
        let (macro_step, tolerance) = if target == targets::Target::Funnel {
            (0.36, 0.21)
        } else {
            (0.3, 0.3)
        };
        let options = WalnutsOptions::new(macro_step, 8)
            .unwrap()
            .with_local_refinement(10, 1, tolerance)
            .unwrap();
        let mut chains: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| (0..5).map(|_| Vec::with_capacity(8192)).collect())
            .collect();
        for (chain_index, seed) in (701..705).enumerate() {
            let sign = if seed % 2 == 0 { -1.0 } else { 1.0 };
            let mut chain = Walnuts::new(
                &target,
                buffer(&[sign, -sign]),
                IdentityMetric::new(2),
                options,
            )
            .unwrap();
            let mut rng = SmallRng::seed_from_u64(seed);
            for draw in 0..(1024 + 8192) {
                let _ = chain.step(&mut rng).unwrap();
                if draw >= 1024 {
                    for (output, value) in chains[chain_index]
                        .iter_mut()
                        .zip(target.observables(chain.point().position()))
                    {
                        output.push(value);
                    }
                }
            }
        }
        for observable in 0..5 {
            let views: Vec<_> = chains
                .iter()
                .map(|chain| chain[observable].as_slice())
                .collect();
            let summary = modern::summarize(&views).unwrap();
            assert!(
                summary.rhat <= 1.01,
                "{target:?} observable {observable}: R-hat {}",
                summary.rhat
            );
            assert!(
                summary.bulk_ess >= 400.0,
                "{target:?} observable {observable}: bulk ESS {}",
                summary.bulk_ess
            );
            assert!(
                summary.tail_ess >= 400.0,
                "{target:?} observable {observable}: tail ESS {}",
                summary.tail_ess
            );
        }
    }
}
