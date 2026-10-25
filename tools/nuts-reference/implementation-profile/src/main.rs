//! Bounded dense-fit timings and descriptive native adaptation ablation.
use alea_core::target::LogDensityGradient;
use alea_math::{buffer::OwnedBuffer, metric::EuclideanMetric};
use alea_mcmc::{
    Nuts, NutsOptions,
    adapt::{MetricKind, OnlineCovariance},
};
use rand::{SeedableRng, rngs::SmallRng};
use std::{cell::Cell, error::Error, hint::black_box, time::Instant};

#[cfg(feature = "faer")]
mod adaptation;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("fit") => {
            let d = args.get(2).ok_or("dimension required")?.parse()?;
            let repeats = args.get(3).ok_or("repeat count required")?.parse()?;
            fit(d, repeats)
        }
        Some("flat") => {
            let d = args.get(2).ok_or("dimension required")?.parse()?;
            let draws = args.get(3).ok_or("draw count required")?.parse()?;
            flat(d, draws)
        }
        #[cfg(feature = "faer")]
        Some("adaptation") => adaptation::run(args.get(2).ok_or("new output directory required")?),
        _ => Err(
            "use fit DIM REPEATS, flat DIM DRAWS, or adaptation NEW_DIRECTORY (requires faer)"
                .into(),
        ),
    }
}

fn fit(d: usize, repeats: usize) -> Result<(), Box<dyn Error>> {
    let mut covariance = OnlineCovariance::new(d, MetricKind::Dense)?;
    for sample in 0..3 * d {
        let row: Vec<_> = (0..d)
            .map(|i| (((sample + 1) * (i + 1)) as f64 * 1.618).sin())
            .collect();
        covariance.update(&row)?;
    }
    let start = Instant::now();
    let mut checksum = 0.;
    for _ in 0..repeats {
        checksum += black_box(covariance.metric()?).log_det();
    }
    println!(
        "{{\"mode\":\"fit\",\"dimension\":{d},\"repeats\":{repeats},\"nanoseconds\":{},\"checksum\":{checksum}}}",
        start.elapsed().as_nanos()
    );
    Ok(())
}

struct Flat(usize, Cell<usize>);
impl LogDensityGradient for Flat {
    type Error = std::convert::Infallible;
    fn dimension(&self) -> usize {
        self.0
    }
    fn logp_grad(&self, _: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        self.1.set(self.1.get() + 1);
        g.fill(0.);
        Ok(0.)
    }
}
fn flat(d: usize, draws: usize) -> Result<(), Box<dyn Error>> {
    let target = Flat(d, Cell::new(0));
    let mut chain = Nuts::new(
        &target,
        OwnedBuffer::new(d),
        alea_math::metric::IdentityMetric::new(d),
        NutsOptions::new(0.1, 6)?,
    )?;
    let mut rng = SmallRng::seed_from_u64(20261022);
    // Warm construction/cache paths outside timing; every transition has63edges.
    for _ in 0..32 {
        let _ = chain.step(&mut rng)?;
    }
    target.1.set(0);
    let start = Instant::now();
    let mut edges = 0;
    let mut checksum = 0.;
    for _ in 0..draws {
        let info = black_box(chain.step(&mut rng)?);
        assert_eq!(info.integration_steps, 63);
        edges += info.integration_steps;
        checksum += chain.point().position()[0];
    }
    let nanoseconds = start.elapsed().as_nanos();
    assert_eq!(target.1.get(), edges);
    println!(
        "{{\"mode\":\"flat\",\"dimension\":{d},\"draws\":{draws},\"edges\":{edges},\"calls\":{},\"nanoseconds\":{nanoseconds},\"checksum\":{checksum}}}",
        target.1.get()
    );
    Ok(())
}
