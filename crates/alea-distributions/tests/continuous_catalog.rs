#![allow(clippy::excessive_precision)] // Independent high-precision fixture decimals.
use alea_core::distribution::{ContinuousDistribution, Distribution, ParameterScore, Sample};
use alea_distributions::scalar::*;
use rand::{SeedableRng, rngs::SmallRng};
use statrs::distribution::{self as reference, Continuous};
use std::f64::consts::PI;
fn close(a: f64, b: f64, t: f64) {
    assert!((a - b).abs() <= t * (1.0 + b.abs()), "{a} != {b}");
}
fn scores<D: ContinuousDistribution + ParameterScore>(
    law: D,
    p: &[f64],
    x: f64,
    make: impl Fn(&[f64]) -> D,
) {
    let mut s = vec![0.0; p.len()];
    law.parameter_score(&x, &mut s).unwrap();
    for i in 0..p.len() {
        let h = 2e-5 * (1.0 + p[i].abs());
        let mut plus = p.to_vec();
        let mut minus = plus.clone();
        plus[i] += h;
        minus[i] -= h;
        close(
            s[i],
            (make(&plus).log_prob(&x).unwrap() - make(&minus).log_prob(&x).unwrap()) / (2.0 * h),
            3e-7,
        );
    }
    let h = 1e-5;
    close(
        law.event_score(x).unwrap(),
        (law.log_prob(&(x + h)).unwrap() - law.log_prob(&(x - h)).unwrap()) / (2.0 * h),
        3e-7,
    );
    assert!(law.parameter_score(&x, &mut []).is_err());
    assert!(law.log_prob(&f64::NAN).is_err());
}
#[test]
fn every_continuous_score_matches_independent_differences() {
    macro_rules! check {($law:ident,[$($p:expr),+],$x:expr)=>{scores($law::new($($p),+).unwrap(),&[$($p),+],$x,|p|{let mut it=p.iter();$law::new($( {let _=$p;*it.next().unwrap()} ),+).unwrap()});};}
    check!(Uniform, [-1.0, 3.0], 0.6);
    check!(Cauchy, [0.2, 1.3], 0.6);
    check!(Laplace, [0.2, 1.3], 0.6);
    check!(Logistic, [0.2, 1.3], 0.6);
    check!(Gumbel, [0.2, 1.3], 0.6);
    check!(SkewNormal, [0.2, 1.3, -2.0], 0.6);
    check!(ExpModNormal, [0.2, 1.3, 2.0], 0.6);
    check!(ExpModNormal, [0.2, 1.3, 20.0], -2.0);
    check!(Weibull, [1.7, 1.3], 0.6);
    check!(Frechet, [-0.2, 1.3, 1.7], 0.6);
    check!(Pareto, [0.2, 1.7], 0.6);
    check!(ParetoType2, [-0.2, 1.3, 1.7], 0.6);
    check!(InverseGamma, [2.1, 1.7], 0.6);
    check!(ChiSquared, [3.2], 0.6);
    check!(InverseChiSquared, [3.2], 0.6);
    check!(ScaledInverseChiSquared, [3.2, 1.3], 0.6);
    check!(FisherF, [3.2, 4.7], 0.6);
    check!(Rayleigh, [1.3], 0.6);
    check!(InverseGaussian, [1.3, 2.1], 0.6);
    check!(Levy, [-0.2, 1.3], 0.6);
    check!(HalfNormal, [1.3], 0.6);
    check!(HalfCauchy, [1.3], 0.6);
    check!(LogLogistic, [1.7, 1.3], 0.6);
    check!(Triangular, [-1.0, 3.0, 0.2], 0.6);
    check!(Pert, [-1.0, 3.0, 0.2, 4.0], 0.6);
    check!(VonMises, [0.2, 2.1], 0.6);
    check!(VonMises, [0.2, 80.0], 0.6);
}
#[test]
fn independent_reference_values_and_boundaries() {
    macro_rules! compare {($law:ident,$ref:ident,[$($p:expr),+])=>{for x in [0.01,0.3,1.0,3.0,100.0]{close($law::new($($p),+).unwrap().log_prob(&x).unwrap(),reference::$ref::new($($p),+).unwrap().ln_pdf(x),2e-12);}};}
    compare!(Uniform, Uniform, [-1.0, 300.0]);
    compare!(Cauchy, Cauchy, [0.2, 1.3]);
    compare!(Laplace, Laplace, [0.2, 1.3]);
    compare!(Gumbel, Gumbel, [0.2, 1.3]);
    compare!(Weibull, Weibull, [1.7, 1.3]);
    compare!(Pareto, Pareto, [0.001, 1.7]);
    compare!(InverseGamma, InverseGamma, [2.1, 1.7]);
    compare!(ChiSquared, ChiSquared, [3.2]);
    compare!(FisherF, FisherSnedecor, [3.2, 4.7]);
    for x in [-0.8, 0.0, 0.3, 2.9] {
        close(
            Triangular::new(-1.0, 3.0, 0.2)
                .unwrap()
                .log_prob(&x)
                .unwrap(),
            reference::Triangular::new(-1.0, 3.0, 0.2)
                .unwrap()
                .ln_pdf(x),
            1e-13,
        );
    }
    close(
        ChiSquared::new(2.0).unwrap().log_prob(&0.0).unwrap(),
        -2.0_f64.ln(),
        1e-14,
    );
    close(
        Weibull::new(1.0, 2.0).unwrap().log_prob(&0.0).unwrap(),
        -2.0_f64.ln(),
        1e-14,
    );
    assert!(
        Weibull::new(0.5, 2.0)
            .unwrap()
            .log_prob(&0.0)
            .unwrap()
            .is_infinite()
    );
    assert_eq!(
        HalfNormal::new(1.0).unwrap().log_prob(&-1.0).unwrap(),
        f64::NEG_INFINITY
    );
    assert!(Uniform::new(1.0, 1.0).is_err());
    assert!(Triangular::new(0.0, 1.0, 2.0).is_err());
    assert!(VonMises::new(0.0, -1.0).is_err());
    for x in [-100.0, -20.0, 0.0, 20.0] {
        let s = SkewNormal::new(0.0, 1.0, 0.0).unwrap();
        close(
            s.log_prob(&x).unwrap(),
            Normal::new(0.0, 1.0).unwrap().log_prob(&x).unwrap(),
            1e-13,
        );
    }
}
fn integral<D: Distribution<Event = f64> + ?Sized>(d: &D, lo: f64, hi: f64, n: usize) -> f64 {
    let h = (hi - lo) / n as f64;
    (0..n)
        .map(|i| d.log_prob(&(lo + (i as f64 + 0.5) * h)).unwrap().exp())
        .sum::<f64>()
        * h
}
fn generated_mean<D: Sample<Draw = f64> + Distribution<Event = f64>>(
    d: D,
    mean: f64,
    tolerance: f64,
) {
    let mut rng = SmallRng::seed_from_u64(5789);
    let n = 30000;
    let mut sum = 0.0;
    for _ in 0..n {
        let x = d.sample(&mut rng).unwrap();
        assert!(d.log_prob(&x).unwrap().is_finite());
        sum += x;
    }
    close(sum / n as f64, mean, tolerance);
}
#[test]
fn normalization_and_generative_checks() {
    let laws: Vec<(Box<dyn Distribution<Event = f64>>, f64, f64)> = vec![
        (Box::new(Logistic::new(0.2, 1.3).unwrap()), -40.0, 40.0),
        (
            Box::new(SkewNormal::new(0.2, 1.3, -2.0).unwrap()),
            -20.0,
            20.0,
        ),
        (
            Box::new(ExpModNormal::new(0.2, 1.3, 2.0).unwrap()),
            -20.0,
            40.0,
        ),
        (Box::new(InverseGaussian::new(1.3, 2.1).unwrap()), 0.0, 60.0),
        (Box::new(LogLogistic::new(3.0, 1.3).unwrap()), 0.0, 100.0),
        (Box::new(Pert::new(-1.0, 3.0, 0.2, 4.0).unwrap()), -1.0, 3.0),
        (Box::new(VonMises::new(0.2, 0.0).unwrap()), -PI, PI),
        (Box::new(VonMises::new(0.2, 2.1).unwrap()), -PI, PI),
        (Box::new(VonMises::new(0.2, 80.0).unwrap()), -PI, PI),
    ];
    for (d, a, b) in laws {
        close(integral(&*d, a, b, 100000), 1.0, 3e-6);
    }
    generated_mean(Weibull::new(2.0, 2.0).unwrap(), PI.sqrt(), 0.012);
    generated_mean(InverseGaussian::new(1.3, 2.1).unwrap(), 1.3, 0.015);
    generated_mean(ExpModNormal::new(0.2, 1.3, 2.0).unwrap(), 0.7, 0.015);
    generated_mean(
        Pert::new(-1.0, 3.0, 0.2, 4.0).unwrap(),
        (-1.0 + 3.0 + 4.0 * 0.2) / 6.0,
        0.015,
    );
    let law = VonMises::new(0.0, 2.1).unwrap();
    let mut rng = SmallRng::seed_from_u64(908);
    let n = 30000;
    let c = (0..n)
        .map(|_| law.sample(&mut rng).unwrap().cos())
        .sum::<f64>()
        / n as f64;
    let mut s = [0.0; 2];
    law.parameter_score(&0.0, &mut s).unwrap();
    close(c, 1.0 - s[1], 0.012);
}

#[test]
fn von_mises_normalizer_matches_80_digit_bessel_reference() {
    // mpmath 1.4.1, 80 decimal digits: k - log(2*pi*I0(k)), 1-I1(k)/I0(k).
    for (k, lp, concentration_score) in [
        (0.0, -1.83787706640934548, 1.0),
        (0.01, -1.82790206625309722, 0.995000062498958351),
        (2.0, -0.661870607892301766, 0.302225342035992018),
        (49.9, 1.03354131678728308, 0.0100712795925539805),
        (50.0, 1.03454743171884993, 0.0100510326215022474),
        (80.0, 1.27050238880505535, 0.00626978028713216346),
        (1000.0, 2.53481404372118984, 0.000500125125195719801),
        (1e8, 8.29140183752150999, 5.00000001250000013e-9),
    ] {
        let law = VonMises::new(0.0, k).unwrap();
        close(law.log_prob(&0.0).unwrap(), lp, 1e-13);
        let mut s = [0.0; 2];
        law.parameter_score(&0.0, &mut s).unwrap();
        close(s[1], concentration_score, 1e-14);
    }
}

#[test]
fn normal_inverse_gaussian_density_scores_and_mixture_sampling() {
    scores(
        NormalInverseGaussian::new(2.0, 0.4, 0.2, 1.3).unwrap(),
        &[2.0, 0.4, 0.2, 1.3],
        0.7,
        |p| NormalInverseGaussian::new(p[0], p[1], p[2], p[3]).unwrap(),
    );
    // Independent mpmath 80-digit Bessel K evaluations, including scaled-function tails.
    for (p, x, expected) in [
        ([2.0, 0.4, 0.2, 1.3], -30.0, -75.40546476222145155337),
        ([2.0, 0.4, 0.2, 1.3], 0.7, -0.7294127950436785371776),
        ([2.0, 0.4, 0.2, 1.3], 30.0, -50.58617029415230835982),
        ([1000.0, 1.0, 0.0, 1.0], 0.0, 2.534813918825247654304),
        ([1e-8, 0.0, 0.0, 1.0], 0.0, -1.144729875849401125974),
    ] {
        close(
            NormalInverseGaussian::new(p[0], p[1], p[2], p[3])
                .unwrap()
                .log_prob(&x)
                .unwrap(),
            expected,
            2e-13,
        );
    }
    let law = NormalInverseGaussian::new(2.0, 0.4, 0.2, 1.3).unwrap();
    close(integral(&law, -30.0, 30.0, 100000), 1.0, 1e-10);
    generated_mean(law, 0.2 + 1.3 * 0.4 / (4.0_f64 - 0.16).sqrt(), 0.012);
    assert!(NormalInverseGaussian::new(1.0, 1.0, 0.0, 1.0).is_err());
}

#[test]
fn beta_proportion_and_extreme_value_limits() {
    scores(
        BetaProportion::new(0.3, 4.2).unwrap(),
        &[0.3, 4.2],
        0.6,
        |p| BetaProportion::new(p[0], p[1]).unwrap(),
    );
    scores(
        GeneralizedExtremeValue::new(0.2, 1.3, 0.0).unwrap(),
        &[0.2, 1.3, 0.0],
        0.6,
        |p| GeneralizedExtremeValue::new(p[0], p[1], p[2]).unwrap(),
    );
    for shape in [-0.3, 0.2] {
        scores(
            GeneralizedExtremeValue::new(0.2, 1.3, shape).unwrap(),
            &[0.2, 1.3, shape],
            0.6,
            |p| GeneralizedExtremeValue::new(p[0], p[1], p[2]).unwrap(),
        );
    }
    for x in [0.01, 0.3, 0.8, 0.99] {
        close(
            BetaProportion::new(0.3, 4.2).unwrap().log_prob(&x).unwrap(),
            Beta::new(0.3 * 4.2, 0.7 * 4.2)
                .unwrap()
                .log_prob(&x)
                .unwrap(),
            1e-13,
        );
    }
    for x in [-20.0, 0.0, 2.0, 100.0] {
        close(
            GeneralizedExtremeValue::new(0.2, 1.3, 0.0)
                .unwrap()
                .log_prob(&x)
                .unwrap(),
            Gumbel::new(0.2, 1.3).unwrap().log_prob(&x).unwrap(),
            1e-13,
        );
    }
    close(
        GeneralizedExtremeValue::new(0.0, 2.0, -1.0)
            .unwrap()
            .log_prob(&2.0)
            .unwrap(),
        -2.0_f64.ln(),
        1e-13,
    );
    assert_eq!(
        GeneralizedExtremeValue::new(0.0, 1.0, -0.2)
            .unwrap()
            .log_prob(&6.0)
            .unwrap(),
        f64::NEG_INFINITY
    );
    close(
        integral(
            &GeneralizedExtremeValue::new(0.2, 1.3, -0.2).unwrap(),
            -20.0,
            6.7,
            100000,
        ),
        1.0,
        1e-8,
    );
    generated_mean(
        GeneralizedExtremeValue::new(0.2, 1.3, 0.0).unwrap(),
        0.2 + 1.3 * 0.5772156649015329,
        0.018,
    );
    generated_mean(BetaProportion::new(0.3, 4.2).unwrap(), 0.3, 0.008);
}

#[test]
fn heavy_tailed_densities_do_not_overflow_intermediate_ratios() {
    let x = 1e308_f64;
    let scale = 1e-308_f64;
    close(
        Cauchy::new(0.0, scale).unwrap().log_prob(&x).unwrap(),
        -std::f64::consts::PI.ln() + scale.ln() - 2.0 * x.ln(),
        1e-13,
    );
    close(
        Pareto::new(scale, 2.0).unwrap().log_prob(&x).unwrap(),
        2.0_f64.ln() - scale.ln() - 3.0 * (x.ln() - scale.ln()),
        1e-13,
    );
    close(
        Levy::new(-x, x).unwrap().log_prob(&x).unwrap(),
        0.5 * (x.ln() - (2.0 * std::f64::consts::PI).ln()) - 1.5 * (x.ln() + 2.0_f64.ln()) - 0.25,
        1e-13,
    );
    assert!(
        Frechet::new(-x, 1.0, 2.0)
            .unwrap()
            .log_prob(&x)
            .unwrap()
            .is_finite()
    );
    assert!(
        LogLogistic::new(2.0, scale)
            .unwrap()
            .log_prob(&x)
            .unwrap()
            .is_finite()
    );
}
