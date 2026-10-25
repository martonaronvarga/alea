//! Cold construction excluded; each iteration evaluates log probability and parameter score.
use alea_core::distribution::ParameterScore;
use alea_distributions::{mixture::FiniteMixture, multivariate::*, scalar::*};
use std::{hint::black_box, time::Instant};
fn profile<D: ParameterScore>(name: &str, law: D, event: &D::Event) {
    let mut gradient = vec![0.0; law.parameter_count()];
    let count = 200_000;
    let start = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..count {
        checksum += black_box(law.log_prob(black_box(event)).unwrap());
        law.parameter_score(black_box(event), black_box(&mut gradient))
            .unwrap();
    }
    println!(
        "{name},{count},{:.3},{checksum}",
        start.elapsed().as_nanos() as f64 / count as f64
    );
}
fn main() {
    println!("law,iterations,nanoseconds_per_value_and_score,checksum");
    profile("normal", Normal::new(0.3, 1.2).unwrap(), &0.7);
    profile("student_t", StudentT::new(7.0, 0.3, 1.2).unwrap(), &0.7);
    profile("lognormal", LogNormal::new(0.3, 1.2).unwrap(), &0.7);
    profile("exponential", Exponential::new(1.2).unwrap(), &0.7);
    profile("gamma", Gamma::new(2.3, 1.2).unwrap(), &0.7);
    profile("beta", Beta::new(2.3, 1.2).unwrap(), &0.7);
    profile("bernoulli", Bernoulli::new(0.4).unwrap(), &true);
    profile("binomial", Binomial::new(12, 0.4).unwrap(), &3);
    profile("poisson", Poisson::new(1.2).unwrap(), &3);
    profile(
        "mvn2",
        MultivariateNormal::new(vec![0.0; 2], vec![1.0, 0.0, 0.5, 1.0]).unwrap(),
        &[0.2, 0.4],
    );
    profile(
        "dirichlet3",
        Dirichlet::new(vec![2.0, 3.0, 1.0]).unwrap(),
        &[0.2, 0.3, 0.5],
    );
    profile(
        "categorical",
        Categorical::new(vec![0.2, 0.3, 0.5]).unwrap(),
        &1,
    );
    profile(
        "multinomial",
        Multinomial::new(5, vec![0.2, 0.3, 0.5]).unwrap(),
        &[1, 2, 2],
    );
    profile(
        "lkj2",
        LkjCholesky::new(2, 2.0).unwrap(),
        &[1.0, 0.0, 0.3, 0.91f64.sqrt()],
    );
    profile(
        "mixture",
        FiniteMixture::new(
            vec![0.3, 0.7],
            vec![
                Normal::new(-2.0, 1.0).unwrap(),
                Normal::new(2.0, 1.0).unwrap(),
            ],
        )
        .unwrap(),
        &0.4,
    );
}
