use alea_core::distribution::{Distribution, ParameterScore, Sample};
use alea_distributions::{multivariate::*, scalar::*};
use rand::{SeedableRng, rngs::SmallRng};
fn close(a: f64, b: f64, t: f64) {
    assert!((a - b).abs() <= t * (1.0 + b.abs()), "{a} != {b}");
}
#[test]
fn scalar_reductions_and_correlation_measure() {
    for df in [1.3, 3.0, 10.0] {
        for scale in [0.3, 1.0, 2.0] {
            for x in [0.1, 1.0, 4.0] {
                close(
                    MultivariateStudentT::new(df, vec![0.2], vec![scale])
                        .unwrap()
                        .log_prob(&[x])
                        .unwrap(),
                    StudentT::new(df, 0.2, scale).unwrap().log_prob(&x).unwrap(),
                    1e-13,
                );
                close(
                    Wishart::new(df, 1, vec![scale])
                        .unwrap()
                        .log_prob(&[x])
                        .unwrap(),
                    Gamma::new(df * 0.5, 1.0 / (2.0 * scale * scale))
                        .unwrap()
                        .log_prob(&x)
                        .unwrap(),
                    1e-13,
                );
                close(
                    InverseWishart::new(df, 1, vec![scale])
                        .unwrap()
                        .log_prob(&[x])
                        .unwrap(),
                    InverseGamma::new(df * 0.5, scale * scale * 0.5)
                        .unwrap()
                        .log_prob(&x)
                        .unwrap(),
                    1e-13,
                );
            }
        }
    }
    let law = Lkj::new(3, 1.0).unwrap();
    close(
        law.log_prob(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            .unwrap(),
        (2.0 / std::f64::consts::PI.powi(2)).ln(),
        1e-13,
    );
    for eta in [0.5, 1.0, 3.0] {
        for r in [-0.9, -0.3, 0.0, 0.8] {
            let lp = Lkj::new(2, eta)
                .unwrap()
                .log_prob(&[1.0, r, r, 1.0])
                .unwrap();
            let beta = Beta::new(eta, eta)
                .unwrap()
                .log_prob(&((r + 1.0) * 0.5))
                .unwrap()
                - 2.0_f64.ln();
            close(lp, beta, 1e-13);
        }
    }
    let dm = DirichletMultinomial::new(7, vec![1.3, 2.7]).unwrap();
    let bb = BetaBinomial::new(7, 1.3, 2.7).unwrap();
    for k in 0..=7 {
        close(
            dm.log_prob(&[k, 7 - k]).unwrap(),
            bb.log_prob(&k).unwrap(),
            1e-13,
        );
    }
    assert!(Wishart::new(1.0, 2, vec![1.0, 0.0, 0.0, 1.0]).is_err());
    assert_eq!(
        Wishart::new(4.0, 2, vec![1.0, 0.0, 0.0, 1.0])
            .unwrap()
            .log_prob(&[1.0, 2.0, 2.0, 1.0])
            .unwrap(),
        f64::NEG_INFINITY
    );
}
fn matrix_scores(inverted: bool) {
    let p = [5.2, 1.3, 0.2, 0.9];
    let x = [2.0, 0.3, 0.3, 1.5];
    let make = |p: [f64; 4]| -> Box<dyn ParameterScore<Event = [f64]>> {
        let l = vec![p[1], 0.0, p[2], p[3]];
        if inverted {
            Box::new(InverseWishart::new(p[0], 2, l).unwrap())
        } else {
            Box::new(Wishart::new(p[0], 2, l).unwrap())
        }
    };
    let law = make(p);
    let mut s = [0.0; 4];
    law.parameter_score(&x, &mut s).unwrap();
    for i in 0..4 {
        let mut a = p;
        let mut b = p;
        let h = 1e-5;
        a[i] += h;
        b[i] -= h;
        close(
            s[i],
            (make(a).log_prob(&x).unwrap() - make(b).log_prob(&x).unwrap()) / (2.0 * h),
            3e-8,
        );
    }
    let mut g = [0.0; 4];
    if inverted {
        InverseWishart::new(p[0], 2, vec![p[1], 0.0, p[2], p[3]])
            .unwrap()
            .event_gradient(&x, &mut g)
            .unwrap();
    } else {
        Wishart::new(p[0], 2, vec![p[1], 0.0, p[2], p[3]])
            .unwrap()
            .event_gradient(&x, &mut g)
            .unwrap();
    }
    for (i, j) in [(0, 0), (1, 2), (3, 3)] {
        let h = 1e-5;
        let mut a = x;
        let mut b = x;
        a[i] += h;
        b[i] -= h;
        if i != j {
            a[j] += h;
            b[j] -= h;
        }
        let fd = (law.log_prob(&a).unwrap() - law.log_prob(&b).unwrap()) / (2.0 * h);
        close(if i == j { g[i] } else { g[i] + g[j] }, fd, 3e-8);
    }
}
#[test]
fn matrix_and_student_scores() {
    matrix_scores(false);
    matrix_scores(true);
    let p = [3.2, 0.2, -0.3, 1.3, 0.2, 0.9];
    let x = [0.7, -0.8];
    let make = |p: [f64; 6]| {
        MultivariateStudentT::new(p[0], vec![p[1], p[2]], vec![p[3], 0.0, p[4], p[5]]).unwrap()
    };
    let law = make(p);
    let mut s = [0.0; 6];
    law.parameter_score(&x, &mut s).unwrap();
    for i in 0..6 {
        let h = 1e-5;
        let mut a = p;
        let mut b = p;
        a[i] += h;
        b[i] -= h;
        close(
            s[i],
            (make(a).log_prob(&x).unwrap() - make(b).log_prob(&x).unwrap()) / (2.0 * h),
            2e-8,
        );
    }
    let mut g = [0.0; 2];
    law.event_gradient(&x, &mut g).unwrap();
    for i in 0..2 {
        let h = 1e-5;
        let mut a = x;
        let mut b = x;
        a[i] += h;
        b[i] -= h;
        close(
            g[i],
            (law.log_prob(&a).unwrap() - law.log_prob(&b).unwrap()) / (2.0 * h),
            2e-8,
        );
    }
}
#[test]
fn bartlett_sampling_and_vector_moments() {
    let w = Wishart::new(8.0, 2, vec![1.3, 0.0, 0.2, 0.9]).unwrap();
    let iw = InverseWishart::new(8.0, 2, vec![1.3, 0.0, 0.2, 0.9]).unwrap();
    let t = MultivariateStudentT::new(8.0, vec![0.2, -0.3], vec![1.3, 0.0, 0.2, 0.9]).unwrap();
    let mut rng = SmallRng::seed_from_u64(8909);
    let n = 20000;
    let mut sw = [0.0; 4];
    let mut si = [0.0; 4];
    let mut st = [0.0; 2];
    for _ in 0..n {
        let a = w.sample(&mut rng).unwrap();
        let b = iw.sample(&mut rng).unwrap();
        let c = t.sample(&mut rng).unwrap();
        assert!(w.log_prob(&a).unwrap().is_finite());
        assert!(iw.log_prob(&b).unwrap().is_finite());
        for i in 0..4 {
            sw[i] += a[i];
            si[i] += b[i];
        }
        for i in 0..2 {
            st[i] += c[i];
        }
    }
    for (i, s) in [1.69, 0.26, 0.26, 0.85].into_iter().enumerate() {
        close(sw[i] / n as f64, 8.0 * s, 0.015);
        close(si[i] / n as f64, s / 5.0, 0.015);
    }
    close(st[0] / n as f64, 0.2, 0.02);
    close(st[1] / n as f64, -0.3, 0.02);
    let law = Lkj::new(3, 1.0).unwrap();
    let mut sum = 0.0;
    for _ in 0..1000 {
        let x = law.sample(&mut rng).unwrap();
        assert!(law.log_prob(&x).unwrap().is_finite());
        sum += x[1];
    }
    close(sum / 1000.0, 0.0, 0.05);
}
