//! Fixed-work observation; timing is descriptive, without a performance gate.
use alea_core::target::LogDensityGradient;
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{Nuts, NutsOptions};
use alea_runtime::{ensemble::chain_rng, output::ChainId};
use std::{cell::Cell, convert::Infallible, time::Instant};

struct Normal {
    dimension: usize,
    calls: Cell<usize>,
}
impl LogDensityGradient for Normal {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn logp_grad(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, Infallible> {
        self.calls.set(self.calls.get() + 1);
        let mut sum = 0.0;
        for (&q, g) in q.iter().zip(gradient) {
            *g = -q;
            sum += q * q;
        }
        Ok(-0.5 * sum)
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    for dimension in [2, 16, 64] {
        let target = Normal {
            dimension,
            calls: Cell::new(0),
        };
        let options = NutsOptions::new(0.25, 8)?
            .with_position_turning(true)
            .with_step_size_jitter(0.1)?;
        let mut chain = Nuts::new(
            &target,
            OwnedBuffer::new(dimension),
            IdentityMetric::new(dimension),
            options,
        )?;
        let mut rng = chain_rng([43; 32], ChainId(0));
        for _ in 0..33 {
            let _ = chain.step(&mut rng)?;
        }
        let calls = target.calls.get();
        let mut hash = 0xcbf29ce484222325_u64;
        let start = Instant::now();
        for _ in 0..5000 {
            let _ = chain.step(&mut rng)?;
            for &value in chain.point().position() {
                hash = (hash ^ value.to_bits()).wrapping_mul(0x100000001b3);
            }
        }
        println!(
            "{{\"dimension\":{dimension},\"draws\":5000,\"elapsed_ns\":{},\"gradient_calls\":{},\"draw_checksum\":\"{hash:016x}\"}}",
            start.elapsed().as_nanos(),
            target.calls.get() - calls
        );
    }
    Ok(())
}
