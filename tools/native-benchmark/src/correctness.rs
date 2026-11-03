//! Independent posterior equations and finite differences, never timed.
use super::*;

fn input(n: usize) -> Input {
    let k = 8;
    let mut x = Vec::with_capacity(n * k);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let row: Vec<_> = (0..k)
            .map(|j| {
                (((i + 1) * (j + 1)) as f64 * 0.017).sin()
                    + 0.5 * ((((i + 3) * (j + 2)) as f64) * 0.019).cos()
            })
            .collect();
        let eta: f64 = row
            .iter()
            .enumerate()
            .map(|(j, x)| x * if j % 2 == 0 { 0.3 } else { -0.3 } / (1. + j as f64 / 4.))
            .sum();
        y.push(eta + 0.4 * (i as f64 * 0.71).sin());
        x.extend(row);
    }
    Input { n, k, x, y }
}
fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * (1. + expected.abs()),
        "{actual} vs {expected}"
    );
}

// Stan's positive-real chart is sigma=exp(s). Its normal prior omits the
// half-normal ln(2) constant; Alea's positive normal factor does so as well.
fn reference(name: &str, data: &Input, q: &[f64]) -> (f64, Vec<f64>) {
    let p = if name == "mean" { 1 } else { data.k };
    let mut value = -(p as f64) * (2_f64.ln() + 0.5 * std::f64::consts::TAU.ln());
    let mut gradient = vec![0.; q.len()];
    for j in 0..p {
        value -= q[j] * q[j] / 8.;
        gradient[j] = -q[j] / 4.;
    }
    if name != "logistic" {
        let s = q[p];
        let variance = (2. * s).exp();
        value += -0.5 * variance - 0.5 * std::f64::consts::TAU.ln() + s;
        gradient[p] = 1. - variance - data.n as f64;
        for i in 0..data.n {
            let eta: f64 = if name == "mean" {
                q[0]
            } else {
                (0..p).map(|j| data.x[i * p + j] * q[j]).sum()
            };
            let r = data.y[i] - eta;
            value += -s - 0.5 * std::f64::consts::TAU.ln() - 0.5 * r * r / variance;
            for (j, score) in gradient.iter_mut().take(p).enumerate() {
                *score += r / variance
                    * if name == "mean" {
                        1.
                    } else {
                        data.x[i * p + j]
                    };
            }
            gradient[p] += r * r / variance;
        }
    } else {
        for i in 0..data.n {
            let eta: f64 = (0..p).map(|j| data.x[i * p + j] * q[j]).sum();
            let signed = if data.y[i] == 1. { -eta } else { eta };
            value -= signed.max(0.) + (-signed.abs()).exp().ln_1p();
            let probability = if eta >= 0. {
                1. / (1. + (-eta).exp())
            } else {
                eta.exp() / (1. + eta.exp())
            };
            for (j, score) in gradient.iter_mut().take(p).enumerate() {
                *score += data.x[i * p + j] * (data.y[i] - probability);
            }
        }
    }
    (value, gradient)
}

#[test]
fn posterior_density_scores_and_finite_differences_match_equations() {
    for n in [37, 1000, 2500, 5000, 7500, 10000] {
        for name in ["mean", "linear", "logistic"] {
            let mut data = input(n);
            if name == "logistic" {
                data.y
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, y)| *y = f64::from(i % 3 != 0));
            }
            let model = build_model(name, &data).unwrap();
            let target = model.select_posterior().unwrap().into_target().unwrap();
            for scale in [0., 0.3, -1.5] {
                let mut q = vec![scale; target.dimension()];
                if name != "logistic" {
                    *q.last_mut().unwrap() = scale;
                }
                let mut g = vec![0.; q.len()];
                let value = target.logp_grad(&q, &mut g).unwrap();
                let (expected, score) = reference(name, &data, &q);
                close(value, expected, 2e-12);
                for (a, b) in g.iter().zip(score) {
                    close(*a, b, 3e-10);
                }
                if n == 37 {
                    for j in 0..q.len() {
                        let h = 1e-5;
                        q[j] += h;
                        let plus = target.logp_grad(&q, &mut vec![0.; g.len()]).unwrap();
                        q[j] -= 2. * h;
                        let minus = target.logp_grad(&q, &mut vec![0.; g.len()]).unwrap();
                        q[j] += h;
                        close(g[j], (plus - minus) / (2. * h), 2e-7);
                    }
                }
            }
        }
    }
}

#[test]
fn extreme_coordinates_preserve_finite_values_or_explicit_errors() {
    let data = input(37);
    let target = build_model("linear", &data).unwrap().into_target().unwrap();
    for log_sigma in [-1000., -745.2, -400., -300., 400., 710., 1000.] {
        let mut q = vec![0.; 9];
        q[8] = log_sigma;
        let mut g = vec![42.; 9];
        let error = target.logp_grad(&q, &mut g).unwrap_err();
        assert!(target.is_numerical_error(&error), "{error:?}");
        assert_eq!(g, vec![42.; 9]);
    }
    let mut q = vec![0.; 9];
    q[0] = 1e200;
    let error = target.logp_grad(&q, &mut [0.; 9]).unwrap_err();
    assert!(target.is_numerical_error(&error));
    let mut data = input(37);
    data.y
        .iter_mut()
        .enumerate()
        .for_each(|(i, y)| *y = f64::from(i % 2 == 0));
    let target = build_model("logistic", &data)
        .unwrap()
        .into_target()
        .unwrap();
    for q in [vec![1000.; 8], vec![-1000.; 8]] {
        let mut g = vec![0.; 8];
        let value = target.logp_grad(&q, &mut g).unwrap();
        let (expected, score) = reference("logistic", &data, &q);
        close(value, expected, 1e-12);
        for (a, b) in g.iter().zip(score) {
            close(*a, b, 1e-12);
        }
    }
}

#[test]
fn original_seed_large_regression_recovers_from_second_warmup_transition() {
    use alea::{
        math::buffer::OwnedBuffer,
        output::ChainId,
        sampling::adapt::{NutsWarmup, WarmupOptions},
    };
    let metrics = Arc::new(Mutex::new(Metrics::default()));
    let target = Observed {
        target: build_model("linear", &input(10000))
            .unwrap()
            .select_posterior()
            .unwrap()
            .into_target()
            .unwrap(),
        metrics: Arc::clone(&metrics),
        timed: false,
    };
    let warmup = NutsWarmup::new(
        &target,
        OwnedBuffer::new(9),
        NutsOptions::new(0.2, 10).unwrap(),
        WarmupOptions::new(2).with_search(None),
    )
    .unwrap();
    let mut rng = alea::ensemble::chain_rng([17; 32], ChainId(0));
    let (chain, report) = warmup.run(&mut rng).unwrap();
    assert_eq!(report.iterations, 2);
    assert_eq!(report.divergences, 2);
    assert_eq!(chain.point().position(), &[0.; 9]);
    let m = metrics.lock().unwrap();
    assert_eq!(m.calls, 3);
    assert_eq!(m.errors, 1);
    assert!(
        m.first_error
            .as_ref()
            .unwrap()
            .contains("PrecisionLoss { index: 0 }")
    );
}
