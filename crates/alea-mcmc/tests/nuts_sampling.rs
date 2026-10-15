use alea_distributions::Gaussian;
use alea_math::{
    buffer::OwnedBuffer,
    metric::{
        CholeskyFactor, DenseMetric, DiagonalMetric, EuclideanMetric, IdentityMetric,
        LowRankDiagonalMetric,
    },
};
use alea_mcmc::{
    Nuts, NutsOptions,
    integrator::{Integrator, Leapfrog, TwoStage},
    nuts::NutsTermination,
};
use rand::{SeedableRng, rngs::SmallRng};
fn buffer(values: &[f64]) -> OwnedBuffer {
    OwnedBuffer::from_fn(values.len(), |i| values[i])
}
fn check<M: EuclideanMetric, I: Integrator + Copy>(metric: impl Fn() -> M, integrator: I) {
    let target = Gaussian::new(2);
    let mut batches = Vec::new();
    for seed in 601..605 {
        let mut chain = Nuts::new_with_integrator(
            &target,
            buffer(&[1.0, -1.0]),
            metric(),
            NutsOptions::new(0.2, 8).unwrap(),
            integrator,
        )
        .unwrap();
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut batch = [0.0; 5];
        for draw in 0..(512 + 4096) {
            let info = chain.step(&mut rng).unwrap();
            assert!(!matches!(info.termination, NutsTermination::Divergence(_)));
            let q = chain.point().position();
            for (g, q) in chain.point().gradient().iter().zip(q) {
                assert_eq!(*g, -q);
            }
            if draw >= 512 {
                for (sum, v) in
                    batch
                        .iter_mut()
                        .zip([q[0], q[1], q[0] * q[0], q[1] * q[1], q[0] * q[1]])
                {
                    *sum += v / 128.0;
                }
                if (draw - 512 + 1) % 128 == 0 {
                    batches.push(batch);
                    batch = [0.0; 5];
                }
            }
        }
    }
    for (o, expected) in [0.0, 0.0, 1.0, 1.0, 0.0].into_iter().enumerate() {
        let n = batches.len() as f64;
        let mean = batches.iter().map(|b| b[o]).sum::<f64>() / n;
        let se =
            (batches.iter().map(|b| (b[o] - mean).powi(2)).sum::<f64>() / ((n - 1.0) * n)).sqrt();
        assert!(
            (mean - expected).abs() < 6.0 * se + 0.03,
            "obs{o} mean{mean} se{se}"
        );
    }
}
#[test]
#[cfg_attr(miri, ignore = "long stationary sampling gate")]
fn stationary_gaussian_moments_for_every_geometry_and_integrator() {
    check(|| IdentityMetric::new(2), Leapfrog);
    check(
        || DiagonalMetric::new(buffer(&[0.7, 2.3])).unwrap(),
        Leapfrog,
    );
    check(
        || DenseMetric::new(CholeskyFactor::new_lower(2, buffer(&[1.2, 0.3, 0.0, 0.8])).unwrap()),
        Leapfrog,
    );
    check(
        || {
            LowRankDiagonalMetric::new(buffer(&[1.0, 0.7]), buffer(&[0.6, 0.8]), buffer(&[2.0]))
                .unwrap()
        },
        Leapfrog,
    );
    check(|| IdentityMetric::new(2), TwoStage::OMF);
    check(|| IdentityMetric::new(2), TwoStage::BCSS);
}

#[path = "support/targets.rs"]
mod targets;

#[test]
#[cfg_attr(miri, ignore = "executed upstream retained-draw gate")]
fn four_targets_match_executed_stan_blackjax_and_nuts_rs() {
    let fixture: Vec<Vec<&str>> = include_str!("fixtures/nuts-sampling.csv")
        .lines()
        .skip(1)
        .map(|line| line.split(',').collect())
        .collect();
    assert_eq!(fixture.len(), 80);
    for (t, target) in [
        targets::Target::Correlated,
        targets::Target::Banana,
        targets::Target::Logistic,
        targets::Target::Funnel,
    ]
    .into_iter()
    .enumerate()
    {
        let mut draws = Vec::with_capacity(4 * 8192);
        for seed in 501..505 {
            let sign = if seed % 2 == 0 { -1.0 } else { 1.0 };
            let mut chain = Nuts::new(
                &target,
                buffer(&[sign, -sign]),
                IdentityMetric::new(2),
                NutsOptions::new(if t == 3 { 0.06 } else { 0.15 }, 8).unwrap(),
            )
            .unwrap();
            let mut rng = SmallRng::seed_from_u64(seed);
            for draw in 0..9216 {
                let _ = chain.step(&mut rng).unwrap();
                if draw >= 1024 {
                    draws.push(target.observables(chain.point().position()));
                }
            }
        }
        for o in 0..5 {
            let mean = draws.iter().map(|r| r[o]).sum::<f64>() / draws.len() as f64;
            let mut se = 0.0_f64;
            for size in [256, 512, 8192] {
                let batch: Vec<_> = draws
                    .chunks_exact(size)
                    .map(|c| c.iter().map(|r| r[o]).sum::<f64>() / size as f64)
                    .collect();
                let n = batch.len() as f64;
                se = se.max(
                    (batch.iter().map(|b| (b - mean).powi(2)).sum::<f64>() / (n * (n - 1.0)))
                        .sqrt(),
                );
            }
            for backend in ["stan", "blackjax", "nuts-rs"] {
                let rows: Vec<_> = fixture
                    .iter()
                    .filter(|r| {
                        r[0] == backend
                            && r[1].parse::<usize>().unwrap() == t
                            && r[2].parse::<usize>().unwrap() == o
                    })
                    .collect();
                assert_eq!(rows.len(), 1);
                let row = rows[0];
                let expected = row[3].parse::<f64>().unwrap();
                let ref_se = row[4].parse::<f64>().unwrap();
                assert_eq!(row[8], "True");
                assert!(
                    (mean - expected).abs() <= 6.0 * se.hypot(ref_se) + 0.03,
                    "{backend} target{t} obs{o}: {mean} versus{expected}"
                );
            }
        }
    }
}
