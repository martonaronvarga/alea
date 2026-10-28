//! Gates and seeds are declared in docs/m14-validation-protocol.md.
use alea_core::distribution::{
    Distribution, ParameterScore, Sample, ScalarEventScore, log_prob_batch,
};
use alea_distributions::{mixture::FiniteMixture, multivariate::*, scalar::*};
use rand::{SeedableRng, rngs::SmallRng};
fn close(a: f64, b: f64, t: f64) {
    assert!((a - b).abs() <= t * (1.0 + b.abs()), "{a} != {b}");
}
fn integral<D: Distribution<Event = f64>>(d: &D, a: f64, b: f64, n: usize) -> f64 {
    let h = (b - a) / n as f64;
    (0..n)
        .map(|i| d.log_prob(&(a + (i as f64 + 0.5) * h)).unwrap().exp())
        .sum::<f64>()
        * h
}
fn derivative<D: ScalarEventScore + ParameterScore>(d: &D, x: f64) {
    let h = 1e-5;
    let fd = (d.log_prob(&(x + h)).unwrap() - d.log_prob(&(x - h)).unwrap()) / (2.0 * h);
    close(d.event_score(x).unwrap(), fd, 2e-5);
}
fn parameter<D: ParameterScore>(
    d: D,
    x: &D::Event,
    construct: impl Fn(usize, f64) -> D,
    params: &[f64],
) {
    let mut score = vec![0.0; params.len()];
    d.parameter_score(x, &mut score).unwrap();
    for (i, p) in params.iter().enumerate() {
        let h = 1e-5 * (1.0 + p.abs());
        let fd = (construct(i, p + h).log_prob(x).unwrap()
            - construct(i, p - h).log_prob(x).unwrap())
            / (2.0 * h);
        close(score[i], fd, 2e-5);
    }
}
#[test]
fn normalized_values_boundaries_tails_and_shapes() {
    close(
        Normal::new(0.0, 1.0).unwrap().log_prob(&0.0).unwrap(),
        -0.5 * (2.0 * std::f64::consts::PI).ln(),
        1e-14,
    );
    close(
        StudentT::new(1.0, 0.0, 1.0)
            .unwrap()
            .log_prob(&1.0)
            .unwrap(),
        -(2.0 * std::f64::consts::PI).ln(),
        1e-14,
    );
    close(
        Gamma::new(3.0, 2.0).unwrap().log_prob(&1.0).unwrap(),
        4.0f64.ln() - 2.0,
        1e-14,
    );
    close(
        Beta::new(2.0, 3.0).unwrap().log_prob(&0.25).unwrap(),
        1.6875f64.ln(),
        1e-14,
    );
    close(
        Binomial::new(4, 0.5).unwrap().log_prob(&2).unwrap(),
        (6.0f64 / 16.0).ln(),
        1e-14,
    );
    close(
        Poisson::new(2.0).unwrap().log_prob(&2).unwrap(),
        2.0f64.ln() - 2.0,
        1e-14,
    );
    close(
        Gamma::new(1.0, 2.0).unwrap().log_prob(&0.0).unwrap(),
        2.0f64.ln(),
        1e-14,
    );
    close(
        Beta::new(1.0, 1.0).unwrap().log_prob(&0.0).unwrap(),
        0.0,
        1e-14,
    );
    assert_eq!(
        Beta::new(0.5, 2.0).unwrap().log_prob(&0.0).unwrap(),
        f64::INFINITY
    );
    assert_eq!(
        Exponential::new(2.0).unwrap().log_prob(&-1.0).unwrap(),
        f64::NEG_INFINITY
    );
    assert_eq!(
        Bernoulli::new(0.0).unwrap().log_prob(&true).unwrap(),
        f64::NEG_INFINITY
    );
    assert_eq!(Binomial::new(0, 0.0).unwrap().log_prob(&0).unwrap(), 0.0);
    assert_eq!(
        Binomial::new(2, 0.4).unwrap().log_prob(&3).unwrap(),
        f64::NEG_INFINITY
    );
    assert!(Normal::new(0.0, 0.0).is_err());
    assert!(StudentT::new(-1.0, 0.0, 1.0).is_err());
    assert!(Gamma::new(f64::NAN, 1.0).is_err());
    assert!(Bernoulli::new(1.1).is_err());
    let t = StudentT::new(4.0, 0.0, 1.0).unwrap();
    assert!(t.log_prob(&1e300).unwrap().is_finite());
    assert!(t.log_prob(&f64::NAN).is_err());
    let mut out = [23.0; 2];
    assert!(log_prob_batch(&t, &[0.1], &mut out).is_err());
    assert_eq!(out, [23.0; 2]);
    for d in [
        Normal::new(0.0, 1.0)
            .unwrap()
            .log_prob(&f64::INFINITY)
            .unwrap(),
        Gamma::new(2.0, 1.0)
            .unwrap()
            .log_prob(&f64::INFINITY)
            .unwrap(),
        LogNormal::new(0.0, 1.0)
            .unwrap()
            .log_prob(&f64::INFINITY)
            .unwrap(),
    ] {
        assert_eq!(d, f64::NEG_INFINITY);
    }
}
#[test]
fn normalization_integrals_and_discrete_sums() {
    close(
        integral(&Normal::new(0.3, 1.2).unwrap(), -12.0, 12.0, 100_000),
        1.0,
        3e-4,
    );
    close(
        integral(&StudentT::new(7.0, 0.0, 1.0).unwrap(), -40.0, 40.0, 100_000),
        1.0,
        3e-4,
    );
    close(
        integral(&LogNormal::new(0.0, 0.5).unwrap(), 0.0, 30.0, 100_000),
        1.0,
        3e-4,
    );
    close(
        integral(&Exponential::new(1.2).unwrap(), 0.0, 30.0, 100_000),
        1.0,
        3e-4,
    );
    close(
        integral(&Gamma::new(2.5, 1.2).unwrap(), 0.0, 50.0, 100_000),
        1.0,
        3e-4,
    );
    close(
        integral(&Beta::new(2.0, 3.0).unwrap(), 0.0, 1.0, 100_000),
        1.0,
        3e-4,
    );
    let b = Binomial::new(17, 0.3).unwrap();
    close(
        (0..=17).map(|k| b.log_prob(&k).unwrap().exp()).sum(),
        1.0,
        1e-10,
    );
    let p = Poisson::new(8.0).unwrap();
    close(
        (0..100).map(|k| p.log_prob(&k).unwrap().exp()).sum(),
        1.0,
        1e-10,
    );
}
#[test]
fn all_scalar_event_and_parameter_scores_match_finite_differences() {
    derivative(&Normal::new(0.3, 1.2).unwrap(), 0.7);
    derivative(&StudentT::new(7.0, 0.3, 1.2).unwrap(), 0.7);
    derivative(&LogNormal::new(0.3, 1.2).unwrap(), 0.7);
    derivative(&Exponential::new(1.2).unwrap(), 0.7);
    derivative(&Gamma::new(2.3, 1.2).unwrap(), 0.7);
    derivative(&Beta::new(2.3, 1.2).unwrap(), 0.7);
    parameter(
        Normal::new(0.3, 1.2).unwrap(),
        &0.7,
        |i, v| Normal::new(if i == 0 { v } else { 0.3 }, if i == 1 { v } else { 1.2 }).unwrap(),
        &[0.3, 1.2],
    );
    parameter(
        LogNormal::new(0.3, 1.2).unwrap(),
        &0.7,
        |i, v| LogNormal::new(if i == 0 { v } else { 0.3 }, if i == 1 { v } else { 1.2 }).unwrap(),
        &[0.3, 1.2],
    );
    parameter(
        StudentT::new(7.0, 0.3, 1.2).unwrap(),
        &0.7,
        |i, v| {
            StudentT::new(
                if i == 0 { v } else { 7.0 },
                if i == 1 { v } else { 0.3 },
                if i == 2 { v } else { 1.2 },
            )
            .unwrap()
        },
        &[7.0, 0.3, 1.2],
    );
    parameter(
        Exponential::new(1.2).unwrap(),
        &0.7,
        |_, v| Exponential::new(v).unwrap(),
        &[1.2],
    );
    parameter(
        Gamma::new(2.3, 1.2).unwrap(),
        &0.7,
        |i, v| Gamma::new(if i == 0 { v } else { 2.3 }, if i == 1 { v } else { 1.2 }).unwrap(),
        &[2.3, 1.2],
    );
    parameter(
        Beta::new(2.3, 1.2).unwrap(),
        &0.7,
        |i, v| Beta::new(if i == 0 { v } else { 2.3 }, if i == 1 { v } else { 1.2 }).unwrap(),
        &[2.3, 1.2],
    );
    parameter(
        Bernoulli::new(0.4).unwrap(),
        &true,
        |_, v| Bernoulli::new(v).unwrap(),
        &[0.4],
    );
    parameter(
        Binomial::new(12, 0.4).unwrap(),
        &3,
        |_, v| Binomial::new(12, v).unwrap(),
        &[0.4],
    );
    parameter(
        Poisson::new(1.2).unwrap(),
        &3,
        |_, v| Poisson::new(v).unwrap(),
        &[1.2],
    );
}
fn rng_mean<D: Sample>(d: D, mean: f64, variance: f64, to_f64: impl Fn(D::Draw) -> f64) {
    let mut rng = SmallRng::seed_from_u64(14001);
    let n = 40_000;
    let actual = (0..n)
        .map(|_| to_f64(d.sample(&mut rng).unwrap()))
        .sum::<f64>()
        / n as f64;
    assert!(
        (actual - mean).abs() < 6.0 * (variance / n as f64).sqrt(),
        "{actual} vs {mean}"
    );
}
#[test]
fn scalar_rng_moments() {
    rng_mean(Normal::new(-0.3, 1.7).unwrap(), -0.3, 1.7f64.powi(2), |x| x);
    rng_mean(StudentT::new(7.0, 0.0, 1.0).unwrap(), 0.0, 7.0 / 5.0, |x| x);
    rng_mean(
        LogNormal::new(0.0, 0.5).unwrap(),
        0.125f64.exp(),
        0.25f64.exp() * 0.25f64.exp_m1(),
        |x| x,
    );
    rng_mean(Exponential::new(2.0).unwrap(), 0.5, 0.25, |x| x);
    rng_mean(Gamma::new(2.0, 3.0).unwrap(), 2.0 / 3.0, 2.0 / 9.0, |x| x);
    rng_mean(Beta::new(2.0, 3.0).unwrap(), 0.4, 0.04, |x| x);
    rng_mean(Bernoulli::new(0.4).unwrap(), 0.4, 0.24, f64::from);
    rng_mean(Binomial::new(12, 0.4).unwrap(), 4.8, 2.88, |x| x as f64);
    rng_mean(Poisson::new(50.0).unwrap(), 50.0, 50.0, |x| x as f64);
}
#[test]
fn multivariate_values_derivatives_normalization_and_rng() {
    let d = MultivariateNormal::new(vec![0.2, -0.1], vec![2.0, 0.0, 0.5, 1.0]).unwrap();
    let x = [1.2, 0.65]; // whitened .5,.5
    close(
        d.log_prob(&x).unwrap(),
        -(2.0 * std::f64::consts::PI).ln() - 2.0f64.ln() - 0.25,
        1e-14,
    );
    let mut g = [0.0; 2];
    d.event_gradient(&x, &mut g).unwrap();
    close(g[0], -0.125, 1e-14);
    close(g[1], -0.5, 1e-14);
    parameter(
        d,
        &x[..],
        |i, v| {
            let mut p = [0.2, -0.1, 2.0, 0.5, 1.0];
            p[i] = v;
            MultivariateNormal::new(p[..2].to_vec(), vec![p[2], 0.0, p[3], p[4]]).unwrap()
        },
        &[0.2, -0.1, 2.0, 0.5, 1.0],
    );
    let dir = Dirichlet::new(vec![2.0, 3.0, 1.0]).unwrap();
    close(
        dir.log_prob(&[0.2, 0.3, 0.5]).unwrap(),
        (60.0f64 * 0.2 * 0.3 * 0.3).ln(),
        1e-13,
    );
    parameter(
        dir,
        &[0.2, 0.3, 0.5][..],
        |i, v| {
            let mut a = vec![2.0, 3.0, 1.0];
            a[i] = v;
            Dirichlet::new(a).unwrap()
        },
        &[2.0, 3.0, 1.0],
    );
    let multi = Multinomial::new(4, vec![0.2, 0.3, 0.5]).unwrap();
    let mut total = 0.0;
    for a in 0..=4 {
        for b in 0..=4 - a {
            total += multi.log_prob(&[a, b, 4 - a - b]).unwrap().exp();
        }
    }
    close(total, 1.0, 1e-12);
    rng_mean(
        Categorical::new(vec![0.25, 0.75]).unwrap(),
        0.75,
        0.1875,
        |x| x as f64,
    );
    rng_mean(
        Dirichlet::new(vec![2.0, 3.0, 1.0]).unwrap(),
        1.0 / 3.0,
        2.0 * 4.0 / (36.0 * 7.0),
        |x| x[0],
    );
    rng_mean(
        Multinomial::new(10, vec![0.2, 0.3, 0.5]).unwrap(),
        2.0,
        1.6,
        |x| x[0] as f64,
    );
    rng_mean(
        MultivariateNormal::new(vec![0.2, -0.1], vec![2.0, 0.0, 0.5, 1.0]).unwrap(),
        0.2,
        4.0,
        |x| x[0],
    );
}
#[test]
fn lkj_measure_normalizer_eta_score_and_sampling() {
    for eta in [1.0, 2.0, 4.0] {
        let law = LkjCholesky::new(2, eta).unwrap();
        let n = 40_000;
        let h = 2.0 / n as f64;
        let integral = (0..n)
            .map(|i| {
                let r = -1.0 + (i as f64 + 0.5) * h;
                law.log_prob(&[1.0, 0.0, r, (1.0 - r * r).sqrt()])
                    .unwrap()
                    .exp()
            })
            .sum::<f64>()
            * h;
        close(integral, 1.0, 3e-4);
        let l = [1.0, 0.0, 0.3, 0.91f64.sqrt()];
        parameter(law, &l[..], |_, v| LkjCholesky::new(2, v).unwrap(), &[eta]);
        rng_mean(law, 1.0 / (2.0 * eta + 1.0), 1.0, |x| x[2] * x[2]);
    }
    let law = LkjCholesky::new(4, 2.0).unwrap();
    let mut rng = SmallRng::seed_from_u64(14003);
    for _ in 0..1000 {
        let l = law.sample(&mut rng).unwrap();
        assert!(law.log_prob(&l).unwrap().is_finite());
        let mut g = [0.0; 16];
        law.event_gradient(&l, &mut g).unwrap();
    }
}
#[test]
fn finite_mixture_normalization_extreme_offsets_and_event_score() {
    let d = FiniteMixture::new(
        vec![0.3, 0.7],
        vec![
            Normal::new(-2.0, 0.7).unwrap(),
            Normal::new(2.0, 1.2).unwrap(),
        ],
    )
    .unwrap();
    close(integral(&d, -20.0, 20.0, 100_000), 1.0, 3e-4);
    let h = 1e-5;
    let x = 0.4;
    close(
        d.event_score(x).unwrap(),
        (d.log_prob(&(x + h)).unwrap() - d.log_prob(&(x - h)).unwrap()) / (2.0 * h),
        2e-5,
    );
    assert!(d.log_prob(&1000.0).unwrap().is_finite());
    let d = FiniteMixture::new(
        vec![0.0, 1.0],
        vec![
            Exponential::new(1.0).unwrap(),
            Exponential::new(2.0).unwrap(),
        ],
    )
    .unwrap();
    assert_eq!(d.log_prob(&-1.0).unwrap(), f64::NEG_INFINITY);
}
#[test]
fn evaluation_and_scores_allocate_nothing() {
    let normal = Normal::new(0.3, 1.2).unwrap();
    let t = StudentT::new(7.0, 0.3, 1.2).unwrap();
    let log = LogNormal::new(0.3, 1.2).unwrap();
    let exp = Exponential::new(1.2).unwrap();
    let gamma = Gamma::new(2.3, 1.2).unwrap();
    let beta = Beta::new(2.3, 1.2).unwrap();
    let mv = MultivariateNormal::new(vec![0.0; 2], vec![1.0, 0.0, 0.5, 1.0]).unwrap();
    let dir = Dirichlet::new(vec![2.0, 3.0, 1.0]).unwrap();
    let lkj = LkjCholesky::new(2, 2.0).unwrap();
    let categorical = Categorical::new(vec![0.2, 0.3, 0.5]).unwrap();
    let multinomial = Multinomial::new(5, vec![0.2, 0.3, 0.5]).unwrap();
    let mixture = FiniteMixture::new(
        vec![0.3, 0.7],
        vec![
            Normal::new(-2.0, 1.0).unwrap(),
            Normal::new(2.0, 1.0).unwrap(),
        ],
    )
    .unwrap();
    let allocation = allocation_counter::measure(|| {
        for _ in 0..128 {
            for d in [
                &normal
                    as &dyn ScalarEventScore<Error = alea_core::distribution::DistributionError>,
                &t,
                &log,
                &exp,
                &gamma,
                &beta,
            ] {
                std::hint::black_box(d.log_prob(&0.7).unwrap());
                std::hint::black_box(d.event_score(0.7).unwrap());
            }
            normal.parameter_score(&0.7, &mut [0.0; 2]).unwrap();
            log.parameter_score(&0.7, &mut [0.0; 2]).unwrap();
            exp.parameter_score(&0.7, &mut [0.0; 1]).unwrap();
            Bernoulli::new(0.4)
                .unwrap()
                .parameter_score(&true, &mut [0.0])
                .unwrap();
            std::hint::black_box(Bernoulli::new(0.4).unwrap().log_prob(&true).unwrap());
            Binomial::new(12, 0.4)
                .unwrap()
                .parameter_score(&3, &mut [0.0])
                .unwrap();
            std::hint::black_box(Binomial::new(12, 0.4).unwrap().log_prob(&3).unwrap());
            Poisson::new(1.2)
                .unwrap()
                .parameter_score(&3, &mut [0.0])
                .unwrap();
            std::hint::black_box(Poisson::new(1.2).unwrap().log_prob(&3).unwrap());
            t.parameter_score(&0.7, &mut [0.0; 3]).unwrap();
            gamma.parameter_score(&0.7, &mut [0.0; 2]).unwrap();
            beta.parameter_score(&0.7, &mut [0.0; 2]).unwrap();
            std::hint::black_box(mv.log_prob(&[0.2, 0.4]).unwrap());
            mv.parameter_score(&[0.2, 0.4], &mut [0.0; 5]).unwrap();
            dir.parameter_score(&[0.2, 0.3, 0.5], &mut [0.0; 3])
                .unwrap();
            std::hint::black_box(lkj.log_prob(&[1.0, 0.0, 0.3, 0.91f64.sqrt()]).unwrap());
            lkj.parameter_score(&[1.0, 0.0, 0.3, 0.91f64.sqrt()], &mut [0.0])
                .unwrap();
            categorical.parameter_score(&1, &mut [0.0; 3]).unwrap();
            std::hint::black_box(categorical.log_prob(&1).unwrap());
            multinomial
                .parameter_score(&[1, 2, 2], &mut [0.0; 3])
                .unwrap();
            std::hint::black_box(multinomial.log_prob(&[1, 2, 2]).unwrap());
            mixture.parameter_score(&0.4, &mut [0.0; 6]).unwrap();
            std::hint::black_box(mixture.log_prob(&0.4).unwrap());
        }
    });
    assert_eq!(allocation.count_total, 0);
}

fn rng_variance<D: Sample>(law: D, expected: f64, convert: impl Fn(D::Draw) -> f64) {
    let mut rng = SmallRng::seed_from_u64(14002);
    let n = 40_000;
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for i in 1..=n {
        let x = convert(law.sample(&mut rng).unwrap());
        let delta = x - mean;
        mean += delta / i as f64;
        m2 += delta * (x - mean);
    }
    let variance = m2 / (n - 1) as f64;
    assert!(
        (variance - expected).abs() < 0.08 * expected,
        "variance {variance} vs {expected}"
    );
}
#[test]
fn scalar_rng_spreads_match_analytic_variances() {
    rng_variance(Normal::new(-0.3, 1.7).unwrap(), 1.7f64.powi(2), |x| x);
    rng_variance(StudentT::new(7.0, 0.0, 1.0).unwrap(), 7.0 / 5.0, |x| x);
    rng_variance(
        LogNormal::new(0.0, 0.5).unwrap(),
        0.25f64.exp() * 0.25f64.exp_m1(),
        |x| x,
    );
    rng_variance(Exponential::new(2.0).unwrap(), 0.25, |x| x);
    rng_variance(Gamma::new(2.0, 3.0).unwrap(), 2.0 / 9.0, |x| x);
    rng_variance(Beta::new(2.0, 3.0).unwrap(), 0.04, |x| x);
    rng_variance(Bernoulli::new(0.4).unwrap(), 0.24, f64::from);
    rng_variance(Binomial::new(12, 0.4).unwrap(), 2.88, |x| x as f64);
    rng_variance(Poisson::new(50.0).unwrap(), 50.0, |x| x as f64);
}

#[test]
fn categorical_multinomial_and_mixture_parameter_scores_follow_simplex_tangents() {
    let category = Categorical::new(vec![0.2, 0.3, 0.5]).unwrap();
    let mut g = [0.0; 3];
    category.parameter_score(&1, &mut g).unwrap();
    let h = 1e-5;
    let plus = Categorical::new(vec![0.2 + h, 0.3 - h, 0.5]).unwrap();
    let minus = Categorical::new(vec![0.2 - h, 0.3 + h, 0.5]).unwrap();
    close(
        g[0] - g[1],
        (plus.log_prob(&1).unwrap() - minus.log_prob(&1).unwrap()) / (2.0 * h),
        2e-5,
    );
    let multinomial = Multinomial::new(5, vec![0.2, 0.3, 0.5]).unwrap();
    multinomial.parameter_score(&[1, 2, 2], &mut g).unwrap();
    let plus = Multinomial::new(5, vec![0.2 + h, 0.3 - h, 0.5]).unwrap();
    let minus = Multinomial::new(5, vec![0.2 - h, 0.3 + h, 0.5]).unwrap();
    close(
        g[0] - g[1],
        (plus.log_prob(&[1, 2, 2]).unwrap() - minus.log_prob(&[1, 2, 2]).unwrap()) / (2.0 * h),
        2e-5,
    );
    let make = |w, mu| {
        FiniteMixture::new(
            vec![w, 1.0 - w],
            vec![
                Normal::new(mu, 1.0).unwrap(),
                Normal::new(2.0, 1.0).unwrap(),
            ],
        )
        .unwrap()
    };
    let mixture = make(0.3, -2.0);
    let mut score = [0.0; 6];
    mixture.parameter_score(&0.4, &mut score).unwrap();
    close(
        score[0] - score[1],
        (make(0.3 + h, -2.0).log_prob(&0.4).unwrap() - make(0.3 - h, -2.0).log_prob(&0.4).unwrap())
            / (2.0 * h),
        2e-5,
    );
    close(
        score[2],
        (make(0.3, -2.0 + h).log_prob(&0.4).unwrap() - make(0.3, -2.0 - h).log_prob(&0.4).unwrap())
            / (2.0 * h),
        2e-5,
    );
}

#[test]
fn invalid_component_outputs_are_errors_instead_of_support_zeros() {
    use alea_core::distribution::{DistributionError, EventShape, Support};
    struct Faulty(bool);
    impl Distribution for Faulty {
        type Error = alea_core::distribution::DistributionError;
        type Event = f64;
        fn event_shape(&self) -> EventShape {
            EventShape::Scalar
        }
        fn reference_measure(&self) -> alea_core::distribution::ReferenceMeasure {
            alea_core::distribution::ReferenceMeasure::Lebesgue { dimension: 1 }
        }
        fn support(&self) -> Support {
            Support::Real
        }
        fn log_prob(&self, x: &f64) -> Result<f64, DistributionError> {
            if self.0 {
                Ok(f64::NAN)
            } else {
                Normal::new(0.0, 1.0)?.log_prob(x)
            }
        }
    }
    let mixture = FiniteMixture::new(vec![0.5, 0.5], vec![Faulty(false), Faulty(true)]).unwrap();
    assert!(mixture.log_prob(&0.3).is_err());
    let zero_weight =
        FiniteMixture::new(vec![1.0, 0.0], vec![Faulty(false), Faulty(true)]).unwrap();
    assert!(zero_weight.log_prob(&0.3).unwrap().is_finite());
    assert!(log_prob_batch(&Faulty(true), &[0.3], &mut [0.0]).is_err());
}

#[test]
fn gamma_rate_is_not_misvalidated_as_a_second_shape() {
    let rate = 1e306;
    let law = Gamma::new(1.0, rate).unwrap();
    close(law.log_prob(&1e-306).unwrap(), rate.ln() - 1.0, 1e-14);
}

#[test]
fn mixture_compatibility_uses_measure_not_domain() {
    use alea_core::distribution::{DistributionError, EventShape, ReferenceMeasure, Support};
    struct Component {
        counting: bool,
        positive: bool,
    }
    impl Distribution for Component {
        type Error = DistributionError;
        type Event = f64;
        fn event_shape(&self) -> EventShape {
            EventShape::Scalar
        }
        fn support(&self) -> Support {
            if self.positive {
                Support::Positive
            } else {
                Support::Real
            }
        }
        fn reference_measure(&self) -> ReferenceMeasure {
            if self.counting {
                ReferenceMeasure::Counting
            } else {
                ReferenceMeasure::Lebesgue { dimension: 1 }
            }
        }
        fn log_prob(&self, x: &f64) -> Result<f64, Self::Error> {
            if self.counting {
                Ok(if *x == 0. { 0. } else { f64::NEG_INFINITY })
            } else if self.positive {
                Exponential::new(1.)?.log_prob(x)
            } else {
                Normal::new(0., 1.)?.log_prob(x)
            }
        }
    }
    let mixture = FiniteMixture::new(
        vec![0.5, 0.5],
        vec![
            Component {
                counting: false,
                positive: false,
            },
            Component {
                counting: false,
                positive: true,
            },
        ],
    )
    .unwrap();
    assert_eq!(mixture.support(), Support::Union);
    close(
        mixture.log_prob(&-1.).unwrap(),
        0.5f64.ln() + Normal::new(0., 1.).unwrap().log_prob(&-1.).unwrap(),
        1e-14,
    );
    assert!(
        FiniteMixture::new(
            vec![0.5, 0.5],
            vec![
                Component {
                    counting: false,
                    positive: false
                },
                Component {
                    counting: true,
                    positive: false
                }
            ]
        )
        .is_err()
    );
    assert_eq!(
        ReferenceMeasure::SimplexCoordinates { components: 4 }.dimension(),
        Some(3)
    );
    assert_eq!(
        ReferenceMeasure::SphereSurface {
            ambient_dimension: 4
        }
        .dimension(),
        Some(3)
    );
    assert_eq!(
        ReferenceMeasure::SymmetricMatrixCoordinates { order: 4 }.dimension(),
        Some(10)
    );
    assert_eq!(
        ReferenceMeasure::CorrelationCoordinates { order: 4 }.dimension(),
        Some(6)
    );
    assert_eq!(
        ReferenceMeasure::SymmetricMatrixCoordinates { order: usize::MAX }.dimension(),
        None
    );
}

#[test]
fn bound_wiener_laws_preserve_kernels_scores_support_and_errors() {
    use alea_distributions::wiener::*;
    let obs = WienerObservation {
        rt: 0.8,
        boundary: Boundary::Upper,
    };
    let options = WienerOptions::default();
    let p4 = Wiener4Params {
        alpha: 1.2,
        tau: 0.2,
        beta: 0.4,
        delta: 0.7,
    };
    let p5 = Wiener5Params::from_array([1.2, 0.2, 0.4, 0.7, 0.3]);
    let p7 = Wiener7Params::from_array([1.2, 0.2, 0.4, 0.7, 0.3, 0.1, 0.05]);
    macro_rules! check {
        ($law:ident,$kernel:ident,$params:expr,$n:literal) => {{
            let law = $law::new($params, options).unwrap();
            assert_eq!(
                law.log_prob(&obs).unwrap(),
                $kernel
                    .try_log_prob(&obs, &$params, options)
                    .unwrap()
                    .log_prob
            );
            let mut score = [0.; $n];
            law.parameter_score(&obs, &mut score).unwrap();
            assert_eq!(
                score,
                $kernel
                    .try_fused(&obs, &$params, options)
                    .unwrap()
                    .grad_array()
            );
            assert!(law.parameter_score(&obs, &mut []).is_err());
            assert_eq!(
                law.log_prob(&WienerObservation { rt: 0.1, ..obs }).unwrap(),
                f64::NEG_INFINITY
            );
            assert!(
                law.log_prob(&WienerObservation {
                    rt: f64::NAN,
                    ..obs
                })
                .is_err()
            );
            let mut rng = SmallRng::seed_from_u64(123);
            let draw = law.sample(&mut rng).unwrap();
            assert!(law.log_prob(&draw).unwrap().is_finite());
        }};
    }
    check!(Wiener4Law, Wiener4, p4, 4);
    let law = Wiener4Law::new(p4, options).unwrap();
    let mixture = FiniteMixture::new(vec![0.4, 0.6], vec![law, law]).unwrap();
    close(
        mixture.log_prob(&obs).unwrap(),
        law.log_prob(&obs).unwrap(),
        1e-14,
    );
    let mut rng = SmallRng::seed_from_u64(904);
    let draw = mixture.sample(&mut rng).unwrap();
    assert!(mixture.log_prob(&draw).unwrap().is_finite());

    check!(Wiener5Law, Wiener5, p5, 5);
    check!(Wiener7Law, Wiener7, p7, 7);
    assert!(Wiener4Law::new(Wiener4Params { beta: 1., ..p4 }, options).is_err());
    assert!(
        Wiener7Law::new(
            Wiener7Params::from_array([1.2, 0.2, 0.4, 0.7, 0.3, 1., 0.05]),
            options
        )
        .is_err()
    );
}
