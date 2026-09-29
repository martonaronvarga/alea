//! Independent BlackJAX chains, compared with estimated Monte Carlo uncertainty.
#![cfg(not(miri))]
#[path = "support/targets.rs"]
mod targets;
use alea_math::buffer::OwnedBuffer;
use alea_math::metric::{
    CholeskyFactor, DenseMetric, DiagonalMetric, EuclideanMetric, IdentityMetric,
};
use alea_mcmc::{Hmc, HmcOptions};
use rand::{SeedableRng, rngs::SmallRng};
use std::collections::BTreeSet;
use targets::Target;

const REFERENCE: &str = include_str!("fixtures/blackjax-sampling.csv");
const HEADER: &str =
    "target,mass,l00,l10,l11,step_size,steps,chains,warmup,draws,observable,mean,mcse,divergences";
fn buffer(x: &[f64]) -> OwnedBuffer {
    OwnedBuffer::from_fn(x.len(), |i| x[i])
}

struct Case {
    target: Target,
    mass: String,
    factor: [f64; 3],
    eps: f64,
    steps: usize,
    mean: [f64; 6],
    mcse: [f64; 6],
}

fn cases() -> Vec<Case> {
    let mut lines = REFERENCE.lines().filter(|line| !line.starts_with('#'));
    assert_eq!(lines.next(), Some(HEADER));
    let rows = lines.collect::<Vec<_>>();
    assert_eq!(rows.len(), 78);
    let mut seen = BTreeSet::new();
    let cases = rows
        .chunks_exact(6)
        .map(|group| {
            let first = group[0].split(',').collect::<Vec<_>>();
            let mut case = Case {
                target: Target::parse(first[0]),
                mass: first[1].to_string(),
                factor: [
                    first[2].parse().unwrap(),
                    first[3].parse().unwrap(),
                    first[4].parse().unwrap(),
                ],
                eps: first[5].parse().unwrap(),
                steps: first[6].parse().unwrap(),
                mean: [0.0; 6],
                mcse: [0.0; 6],
            };
            assert!(seen.insert((case.target, case.mass.clone())));
            assert_eq!(
                (case.eps, case.steps),
                if case.target == Target::Funnel {
                    (0.06, 18)
                } else {
                    (0.15, 9)
                }
            );
            for (i, row) in group.iter().enumerate() {
                let fields = row.split(',').collect::<Vec<_>>();
                assert_eq!(fields.len(), 14);
                assert_eq!(&fields[..10], &first[..10]);
                assert_eq!(&fields[7..10], &["4", "1024", "8192"]);
                assert_eq!(fields[10].parse::<usize>().unwrap(), i);
                assert_eq!(fields[13], "0");
                case.mean[i] = fields[11].parse().unwrap();
                case.mcse[i] = fields[12].parse().unwrap();
                assert!(
                    case.mean[i].is_finite()
                        && case.mcse[i].is_finite()
                        && case.mcse[i] > 0.0
                        && case.mcse[i] < 0.2
                );
            }
            case
        })
        .collect();
    let mut expected = BTreeSet::new();
    for target in [
        Target::Correlated,
        Target::Banana,
        Target::Logistic,
        Target::Funnel,
    ] {
        for mass in ["identity", "diagonal", "dense"] {
            expected.insert((target, mass.to_string()));
        }
    }
    expected.insert((Target::Rotated, "matched".to_string()));
    assert_eq!(seen, expected, "sampling coverage must not silently change");
    cases
}

struct Borrowed<'a, M>(&'a M);
impl<M: EuclideanMetric> EuclideanMetric for Borrowed<'_, M> {
    fn dimension(&self) -> usize {
        self.0.dimension()
    }
    fn sample_momentum(
        &self,
        s: &[f64],
        d: &mut [f64],
    ) -> Result<(), alea_math::metric::MetricError> {
        self.0.sample_momentum(s, d)
    }
    fn velocity(&self, s: &[f64], d: &mut [f64]) -> Result<(), alea_math::metric::MetricError> {
        self.0.velocity(s, d)
    }
    fn log_det(&self) -> f64 {
        self.0.log_det()
    }
}

fn check<M: EuclideanMetric>(case: &Case, metric: M) {
    let mut blocks = Vec::<[f64; 6]>::with_capacity(128);
    for seed in 901..905 {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut chain = Hmc::new(
            &case.target,
            OwnedBuffer::new(2),
            Borrowed(&metric),
            HmcOptions::new(case.eps, case.steps).unwrap(),
        )
        .unwrap();
        let mut sum = [0.0; 6];
        for iteration in 0..9216 {
            let info = chain.step(&mut rng).unwrap();
            assert!(
                info.divergence.is_none(),
                "{:?}/{} seed {seed}: {info:?}",
                case.target,
                case.mass
            );
            if iteration >= 1024 {
                for (s, v) in sum
                    .iter_mut()
                    .zip(case.target.observables(chain.point().position()))
                {
                    *s += v;
                }
                sum[5] += f64::from(info.accepted);
                if (iteration + 1 - 1024) % 256 == 0 {
                    blocks.push(sum.map(|v| v / 256.0));
                    sum.fill(0.0);
                }
            }
        }
    }
    assert_eq!(blocks.len(), 128);
    for i in 0..6 {
        let mean = blocks.iter().map(|b| b[i]).sum::<f64>() / 128.0;
        let se256 =
            (blocks.iter().map(|b| (b[i] - mean).powi(2)).sum::<f64>() / (127.0 * 128.0)).sqrt();
        let se512 = (blocks
            .chunks_exact(2)
            .map(|b| ((b[0][i] + b[1][i]) / 2.0 - mean).powi(2))
            .sum::<f64>()
            / (63.0 * 64.0))
            .sqrt();
        let se = se256.max(se512);
        assert!(se.is_finite() && se > 0.0 && se < 0.2);
        let bound = 6.0 * se.hypot(case.mcse[i]);
        assert!(
            (mean - case.mean[i]).abs() <= bound,
            "{:?}/{} observable {i}: Rust {mean} ± {se}, BlackJAX {} ± {}, bound {bound}",
            case.target,
            case.mass,
            case.mean[i],
            case.mcse[i]
        );
        // Known whitening checks are independent of BOTH implementations.
        if i < 5 && case.target != Target::Logistic {
            let expected = [0.0, 0.0, 1.0, 1.0, 0.0][i];
            assert!(
                (mean - expected).abs() <= 6.0 * se,
                "{:?}/{} observable {i}: {mean} ± {se}",
                case.target,
                case.mass
            );
            assert!((case.mean[i] - expected).abs() <= 6.0 * case.mcse[i]);
        }
        eprintln!(
            "{:?}/{},observable={i},mean={mean:.8},mcse={se:.8}",
            case.target, case.mass
        );
    }
}

#[test]
fn independent_blackjax_sampling_agrees_with_batch_means_uncertainty() {
    for case in cases() {
        let [a, b, c] = case.factor;
        match case.mass.as_str() {
            "identity" => {
                assert_eq!(case.factor, [1.0, 0.0, 1.0]);
                check(&case, IdentityMetric::new(2));
            }
            "diagonal" => {
                assert_eq!(b, 0.0);
                check(&case, DiagonalMetric::new(buffer(&[a * a, c * c])).unwrap());
            }
            "dense" | "matched" => check(
                &case,
                DenseMetric::new(CholeskyFactor::new_lower(2, buffer(&[a, b, 0.0, c])).unwrap()),
            ),
            _ => panic!("unknown mass"),
        }
    }
}
