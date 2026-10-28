#![allow(clippy::excessive_precision)] // Independent 60-digit spherical integrals.
use alea_core::distribution::{Distribution, ParameterScore, Sample};
use alea_distributions::{directional::*, scalar::VonMises};
use rand::{SeedableRng, rngs::SmallRng};
use std::f64::consts::PI;
fn close(a: f64, b: f64, t: f64) {
    assert!((a - b).abs() <= t * (1.0 + b.abs()), "{a} != {b}");
}
#[test]
fn holonomic_normalizer_and_gradients_match_independent_integrals() {
    // mpmath, 60 decimal digits, direct axial integral on S²; no ODE evaluation.
    for (k, logc, m3) in [
        (0.1, 2.4648048177143116641, 0.34230571538850405412),
        (1.0, 1.9112752995959406187, 0.42923070582775095996),
        (10.0, -0.40573029439809986748, 0.89272776140925085672),
        (1000.0, -5.0693775860255836259, 0.9989994987453528091),
    ] {
        let b = Bingham::new(vec![-k, -k, 0.0]).unwrap();
        close(b.log_normalizer(), logc, 2e-10);
        close(b.mean_squares()[2], m3, 1e-10);
        close(b.mean_squares()[0], (1.0 - m3) * 0.5, 1e-10);
    }
    let b = Bingham::new(vec![-10.0, -2.0, 0.0]).unwrap();
    close(b.log_normalizer(), 0.52490724901501294572, 2e-10);
    for (a, e) in b.mean_squares().iter().zip([
        0.053258668757892489486,
        0.27033250640658288466,
        0.67640882483552462585,
    ]) {
        close(*a, e, 1e-10);
    }
    // Independent S¹ identity: the angle density is von Mises in twice the angle.
    for k in [0.0, 0.1, 1.0, 10.0, 100.0] {
        let b = Bingham::new(vec![-2.0 * k, 0.0]).unwrap();
        let v = VonMises::new(0.0, k).unwrap();
        for phi in [-1.2_f64, 0.2, 1.4] {
            let x = [phi.sin(), phi.cos()];
            close(
                b.log_prob(&x).unwrap(),
                v.log_prob(&(2.0 * phi)).unwrap(),
                3e-10,
            );
        }
    }
}
#[test]
fn repeated_concentrations_shift_invariance_and_score_derivatives() {
    let p = [-10.0, -2.0, 0.0];
    let b = Bingham::new(p.to_vec()).unwrap();
    let x = [0.6, 0.8, 0.0];
    let mut s = [0.0; 3];
    b.parameter_score(&x, &mut s).unwrap();
    close(s.iter().sum(), 0.0, 1e-10);
    for i in 0..3 {
        let h = 1e-4;
        let mut a = p;
        let mut c = p;
        a[i] += h;
        c[i] -= h;
        close(
            s[i],
            (Bingham::new(a.to_vec()).unwrap().log_prob(&x).unwrap()
                - Bingham::new(c.to_vec()).unwrap().log_prob(&x).unwrap())
                / (2.0 * h),
            1e-7,
        );
    }
    close(
        b.log_prob(&x).unwrap(),
        Bingham::new(vec![90.0, 98.0, 100.0])
            .unwrap()
            .log_prob(&x)
            .unwrap(),
        1e-12,
    );
    for d in [2, 3, 4, 8, 32, 64] {
        let b = Bingham::new(vec![8.0; d]).unwrap();
        let mut x = vec![0.0; d];
        x[0] = 1.0;
        close(
            b.log_prob(&x).unwrap(),
            UniformSphere::new(d).unwrap().log_prob(&x).unwrap(),
            1e-13,
        );
        assert_eq!(b.normalizer_steps(), 0);
    }
    for p in [
        vec![-10000.0, -10000.0, 0.0],
        vec![-1.0, -1.0, -0.5, 0.0],
        vec![-20.0; 32],
    ] {
        let b = Bingham::new(p).unwrap();
        close(b.mean_squares().iter().sum(), 1.0, 1e-10);
    }
    let mut g = [0.0; 3];
    b.tangent_score(&x, &mut g).unwrap();
    close(x.iter().zip(g).map(|(x, g)| x * g).sum(), 0.0, 1e-12);
    assert!(Bingham::new(vec![0.0, f64::NAN]).is_err());
    assert!(Bingham::new(vec![-10001.0, 0.0]).is_err());
    assert_eq!(b.log_prob(&[1.0, 1.0, 0.0]).unwrap(), f64::NEG_INFINITY);
}
#[test]
fn angular_central_gaussian_rejection_and_uniform_sampling() {
    let b = Bingham::new(vec![-10.0, -2.0, 0.0]).unwrap();
    let mut rng = SmallRng::seed_from_u64(888);
    let n = 30000;
    let mut m = [0.0; 3];
    for _ in 0..n {
        let x = b.sample(&mut rng).unwrap();
        assert!(b.log_prob(&x).unwrap().is_finite());
        for i in 0..3 {
            m[i] += x[i] * x[i];
        }
    }
    for (a, e) in m.iter().zip(b.mean_squares()) {
        close(*a / n as f64, *e, 0.008);
    }
    let s = UniformSphere::new(3).unwrap();
    let ball = UniformBall::new(3).unwrap();
    close(
        s.log_prob(&[1.0, 0.0, 0.0]).unwrap(),
        -(4.0 * PI).ln(),
        1e-13,
    );
    close(
        ball.log_prob(&[0.0, 0.0, 0.0]).unwrap(),
        -(4.0 * PI / 3.0).ln(),
        1e-13,
    );
    let mut r2 = 0.0;
    for _ in 0..n {
        let x = ball.sample(&mut rng).unwrap();
        assert!(ball.log_prob(&x).unwrap().is_finite());
        r2 += x.iter().map(|x| x * x).sum::<f64>();
        assert!(
            s.log_prob(&s.sample(&mut rng).unwrap())
                .unwrap()
                .is_finite()
        );
    }
    close(r2 / n as f64, 3.0 / 5.0, 0.008);
}
