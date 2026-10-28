#![allow(clippy::excessive_precision)] // Independent zeta derivative fixture.
use alea_core::distribution::{Distribution, ParameterScore, Sample};
use alea_distributions::scalar::*;
use rand::{SeedableRng, rngs::SmallRng};
use statrs::distribution::{self as reference, Discrete};
fn close(a: f64, b: f64, t: f64) {
    assert!((a - b).abs() <= t * (1.0 + b.abs()), "{a} != {b}");
}
fn sum<D: Distribution<Event = u64>>(d: &D, n: u64) -> f64 {
    (0..=n).map(|k| d.log_prob(&k).unwrap().exp()).sum()
}
#[test]
fn normalized_masses_independent_references_and_support() {
    let g = Geometric::new(0.3).unwrap();
    let nb = NegativeBinomial::new(2.3, 0.4).unwrap();
    let hg = Hypergeometric::new(25, 9, 11).unwrap();
    for k in 0..50 {
        close(
            g.log_prob(&k).unwrap(),
            reference::Geometric::new(0.3).unwrap().ln_pmf(k + 1),
            1e-12,
        );
        close(
            nb.log_prob(&k).unwrap(),
            reference::NegativeBinomial::new(2.3, 0.4)
                .unwrap()
                .ln_pmf(k),
            1e-12,
        );
    }
    for k in 2..=9 {
        close(
            hg.log_prob(&k).unwrap(),
            reference::Hypergeometric::new(25, 9, 11).unwrap().ln_pmf(k),
            1e-12,
        );
    }
    close(sum(&g, 1000), 1.0, 1e-13);
    close(sum(&nb, 1000), 1.0, 1e-12);
    close(sum(&hg, 11), 1.0, 1e-12);
    close(
        sum(&BetaBinomial::new(12, 0.3, 2.1).unwrap(), 12),
        1.0,
        1e-12,
    );
    close(sum(&Zipf::new(300, 0.0).unwrap(), 300), 1.0, 1e-12);
    close(sum(&Zipf::new(300, 1.3).unwrap(), 300), 1.0, 1e-12);
    for r in [0.2, 2.3, 50.0] {
        for mu in [0.1, 3.0, 100.0] {
            let a = NegativeBinomialMean::new(mu, r).unwrap();
            let b = NegativeBinomial::new(r, r / (r + mu)).unwrap();
            for k in [0, 1, 10, 100] {
                close(a.log_prob(&k).unwrap(), b.log_prob(&k).unwrap(), 2e-12);
            }
        }
    }
    let d = PoissonBinomial::new(vec![0.0, 0.2, 0.7, 1.0]).unwrap();
    for (k, p) in [0.0, 0.24, 0.62, 0.14, 0.0].into_iter().enumerate() {
        close(d.log_prob(&(k as u64)).unwrap().exp(), p, 1e-14);
    }
    let d = DiscreteUniform::new(-3, 4).unwrap();
    for k in -3..=4 {
        close(d.log_prob(&k).unwrap(), -8.0_f64.ln(), 1e-14);
    }
    assert_eq!(d.log_prob(&-4).unwrap(), f64::NEG_INFINITY);
    assert_eq!(Geometric::new(1.0).unwrap().log_prob(&0).unwrap(), 0.0);
    assert!(Geometric::new(0.0).is_err());
    assert!(Hypergeometric::new(3, 4, 1).is_err());
    assert!(Zeta::new(1.0).is_err());
    assert!(Zipf::new(0, 2.0).is_err());
    assert!(PoissonBinomial::new(vec![f64::NAN]).is_err());
}
#[test]
fn scores_and_high_precision_zeta() {
    let law = BetaBinomial::new(13, 1.3, 2.7).unwrap();
    let mut s = [0.0; 2];
    law.parameter_score(&4, &mut s).unwrap();
    let h = 1e-5;
    close(
        s[0],
        (BetaBinomial::new(13, 1.3 + h, 2.7)
            .unwrap()
            .log_prob(&4)
            .unwrap()
            - BetaBinomial::new(13, 1.3 - h, 2.7)
                .unwrap()
                .log_prob(&4)
                .unwrap())
            / (2.0 * h),
        2e-8,
    );
    close(
        s[1],
        (BetaBinomial::new(13, 1.3, 2.7 + h)
            .unwrap()
            .log_prob(&4)
            .unwrap()
            - BetaBinomial::new(13, 1.3, 2.7 - h)
                .unwrap()
                .log_prob(&4)
                .unwrap())
            / (2.0 * h),
        2e-8,
    );
    let law = NegativeBinomialMean::new(2.1, 1.7).unwrap();
    law.parameter_score(&4, &mut s).unwrap();
    close(
        s[0],
        (NegativeBinomialMean::new(2.1 + h, 1.7)
            .unwrap()
            .log_prob(&4)
            .unwrap()
            - NegativeBinomialMean::new(2.1 - h, 1.7)
                .unwrap()
                .log_prob(&4)
                .unwrap())
            / (2.0 * h),
        2e-8,
    );
    close(
        s[1],
        (NegativeBinomialMean::new(2.1, 1.7 + h)
            .unwrap()
            .log_prob(&4)
            .unwrap()
            - NegativeBinomialMean::new(2.1, 1.7 - h)
                .unwrap()
                .log_prob(&4)
                .unwrap())
            / (2.0 * h),
        2e-8,
    );
    // zeta(2)=pi²/6 and independently computed -zeta'(2)/zeta(2).
    let z = Zeta::new(2.0).unwrap();
    close(
        z.log_prob(&1).unwrap(),
        -(std::f64::consts::PI.powi(2) / 6.0).ln(),
        1e-14,
    );
    let mut s = [0.0];
    z.parameter_score(&1, &mut s).unwrap();
    close(s[0], 0.569960993094532806, 2e-14);
    for exponent in [1.001, 1.1, 2.0, 10.0, 32.0, 33.0, 1000.0] {
        let z = Zeta::new(exponent).unwrap();
        let mut s = [0.0];
        z.parameter_score(&3, &mut s).unwrap();
        let h = 1e-6 * (exponent - 1.0).min(1.0);
        let fd = (Zeta::new(exponent + h).unwrap().log_prob(&3).unwrap()
            - Zeta::new(exponent - h).unwrap().log_prob(&3).unwrap())
            / (2.0 * h);
        close(s[0], fd, 2e-7);
    }
}
fn mean<D: Sample<Draw = u64> + Distribution<Event = u64>>(d: D, expected: f64) {
    let mut rng = SmallRng::seed_from_u64(909);
    let n = 20000;
    let mut total = 0.0;
    for _ in 0..n {
        let k = d.sample(&mut rng).unwrap();
        assert!(d.log_prob(&k).unwrap().is_finite());
        total += k as f64;
    }
    close(total / n as f64, expected, 0.025);
}
#[test]
fn generative_distributions() {
    mean(Geometric::new(0.4).unwrap(), 1.5);
    mean(NegativeBinomial::new(2.3, 0.4).unwrap(), 3.45);
    mean(NegativeBinomialMean::new(3.45, 2.3).unwrap(), 3.45);
    mean(BetaBinomial::new(12, 2.0, 3.0).unwrap(), 4.8);
    mean(Hypergeometric::new(25, 9, 11).unwrap(), 99.0 / 25.0);
    mean(PoissonBinomial::new(vec![0.0, 0.2, 0.7, 1.0]).unwrap(), 1.9);
    let mut rng = SmallRng::seed_from_u64(128);
    let zipf = Zipf::new(20, 1.3).unwrap();
    let zeta = Zeta::new(2.3).unwrap();
    let n = 50000;
    let mut p_zip = 0;
    let mut p_zeta = 0;
    for _ in 0..n {
        p_zip += usize::from(zipf.sample(&mut rng).unwrap() == 1);
        p_zeta += usize::from(zeta.sample(&mut rng).unwrap() == 1);
    }
    close(
        p_zip as f64 / n as f64,
        zipf.log_prob(&1).unwrap().exp(),
        0.008,
    );
    close(
        p_zeta as f64 / n as f64,
        zeta.log_prob(&1).unwrap().exp(),
        0.008,
    );
}

#[test]
fn ordered_models_normalize_and_have_cutpoint_scores() {
    for probit in [false, true] {
        for location in [-30.0, -2.0, 0.0, 2.0, 30.0] {
            let cuts = vec![-1.2, 0.3, 2.0];
            let law: Box<
                dyn ParameterScore<
                        Event = usize,
                        Error = alea_core::distribution::DistributionError,
                    >,
            > = if probit {
                Box::new(OrderedProbit::new(location, cuts.clone()).unwrap())
            } else {
                Box::new(OrderedLogistic::new(location, cuts.clone()).unwrap())
            };
            let mass = (0..4).map(|k| law.log_prob(&k).unwrap().exp()).sum::<f64>();
            close(mass, 1.0, 1e-13);
            for k in 0..4 {
                let mut s = [0.0; 4];
                law.parameter_score(&k, &mut s).unwrap();
                for i in 0..4 {
                    let h = 1e-4;
                    let eval = |delta: f64| {
                        let mut c = cuts.clone();
                        let loc = if i == 0 {
                            location + delta
                        } else {
                            c[i - 1] += delta;
                            location
                        };
                        if probit {
                            OrderedProbit::new(loc, c).unwrap().log_prob(&k).unwrap()
                        } else {
                            OrderedLogistic::new(loc, c).unwrap().log_prob(&k).unwrap()
                        }
                    };
                    close(s[i], (eval(h) - eval(-h)) / (2.0 * h), 2e-7);
                }
            }
        }
    }
    let b = BetaNegativeBinomial::new(2.0, 4.0, 3.0).unwrap();
    close(sum(&b, 10000), 1.0, 1e-10);
    let mut s = [0.0; 3];
    b.parameter_score(&5, &mut s).unwrap();
    for i in 0..3 {
        let mut p = [2.0, 4.0, 3.0];
        let h = 1e-5;
        p[i] += h;
        let a = BetaNegativeBinomial::new(p[0], p[1], p[2])
            .unwrap()
            .log_prob(&5)
            .unwrap();
        p[i] -= 2.0 * h;
        let c = BetaNegativeBinomial::new(p[0], p[1], p[2])
            .unwrap()
            .log_prob(&5)
            .unwrap();
        close(s[i], (a - c) / (2.0 * h), 1e-7);
    }
}

#[test]
fn degenerate_counts_and_signed_integer_sampling() {
    let p = Poisson::new(0.0).unwrap();
    let mut rng = SmallRng::seed_from_u64(99);
    assert_eq!(p.log_prob(&0).unwrap(), 0.0);
    assert_eq!(p.log_prob(&1).unwrap(), f64::NEG_INFINITY);
    assert_eq!(p.sample(&mut rng).unwrap(), 0);
    let mut s = [0.0];
    p.parameter_score(&0, &mut s).unwrap();
    assert_eq!(s, [-1.0]);
    for (n, prob, k, gradient) in [(0, 0.0, 0, 0.0), (10, 0.0, 0, -10.0), (10, 1.0, 10, 10.0)] {
        let b = Binomial::new(n, prob).unwrap();
        b.parameter_score(&k, &mut s).unwrap();
        assert_eq!(s, [gradient]);
    }
    for (lo, hi) in [
        (-5, 3),
        (i64::MIN, i64::MAX),
        (i64::MIN, i64::MIN),
        (i64::MAX - 10, i64::MAX),
    ] {
        let law = DiscreteUniform::new(lo, hi).unwrap();
        for _ in 0..100 {
            let k = law.sample(&mut rng).unwrap();
            assert!((lo..=hi).contains(&k));
        }
    }
}

#[test]
fn regression_links_remain_stable_at_extreme_predictors() {
    for eta in [-1000.0, -40.0, -1.2, 0.0, 1.3, 40.0, 1000.0] {
        let b = BernoulliLogit::new(eta).unwrap();
        let d = BinomialLogit::new(7, eta).unwrap();
        close(
            b.log_prob(&true).unwrap().exp() + b.log_prob(&false).unwrap().exp(),
            1.0,
            1e-14,
        );
        close(sum(&d, 7), 1.0, 1e-13);
        let mut s = [0.0];
        b.parameter_score(&true, &mut s).unwrap();
        if eta == 40.0 {
            assert!(s[0] > 0.0);
        }
        let h = 1e-3;
        d.parameter_score(&3, &mut s).unwrap();
        let fd = (BinomialLogit::new(7, eta + h)
            .unwrap()
            .log_prob(&3)
            .unwrap()
            - BinomialLogit::new(7, eta - h)
                .unwrap()
                .log_prob(&3)
                .unwrap())
            / (2.0 * h);
        close(s[0], fd, 1e-7);
    }
    for eta in [-20.0, -1.0, 0.0, 3.0] {
        let p = PoissonLog::new(eta).unwrap();
        for k in [0, 1, 5] {
            close(
                p.log_prob(&k).unwrap(),
                Poisson::new(eta.exp()).unwrap().log_prob(&k).unwrap(),
                1e-13,
            );
            let mut s = [0.0];
            p.parameter_score(&k, &mut s).unwrap();
            close(s[0], k as f64 - eta.exp(), 1e-14);
        }
    }
}

#[test]
fn poisson_binomial_log_tails_and_probability_vector_scores() {
    let law = PoissonBinomial::new(vec![0.5; 4096]).unwrap();
    close(law.log_prob(&0).unwrap(), -4096.0 * 2.0_f64.ln(), 2e-13);
    close(law.log_prob(&4096).unwrap(), -4096.0 * 2.0_f64.ln(), 2e-13);
    let p = [0.1, 0.3, 0.6, 0.9];
    for k in 0..=4 {
        let law = PoissonBinomial::new(p.to_vec()).unwrap();
        let mut s = [0.0; 4];
        law.parameter_score(&k, &mut s).unwrap();
        for i in 0..4 {
            let h = 1e-5;
            let mut a = p;
            let mut b = p;
            a[i] += h;
            b[i] -= h;
            let fd = (PoissonBinomial::new(a.to_vec())
                .unwrap()
                .log_prob(&k)
                .unwrap()
                - PoissonBinomial::new(b.to_vec())
                    .unwrap()
                    .log_prob(&k)
                    .unwrap())
                / (2.0 * h);
            close(s[i], fd, 2e-7);
        }
    }
    let tiny = PoissonBinomial::new(vec![1e-100; 4]).unwrap();
    let mut s = [0.0; 4];
    tiny.parameter_score(&4, &mut s).unwrap();
    for g in s {
        close(g, 1e100, 1e-12);
    }
    let deterministic = PoissonBinomial::new(vec![0.0, 1.0]).unwrap();
    let mut s = [0.0; 2];
    deterministic.parameter_score(&1, &mut s).unwrap();
    assert_eq!(s, [-1.0, 1.0]);
}
