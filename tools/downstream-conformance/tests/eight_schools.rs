//! Independent measure/score checks for the centered eight-schools review case.
//! These are target checks, not evidence that a sampling policy converges.
use alea::{
    model::assembly::{AssemblyError, NativeModel},
    target::LogDensityGradient,
};

const Y: [f64; 8] = [28., 8., -3., 7., -1., 1., 18., 12.];
const SIGMA: [f64; 8] = [15., 10., 16., 11., 9., 11., 10., 18.];
const N: usize = 8;

// The shared module also serves the benchmark package's Enzyme build script.
#[allow(unexpected_cfgs)]
#[path = "../../native-benchmark/src/conformance.rs"]
mod conformance;

fn model(noncentered: bool) -> Result<NativeModel, AssemblyError> {
    conformance::native(
        if noncentered { "schools_nc" } else { "schools" },
        N,
        1,
        &SIGMA,
        &Y,
    )
}

fn scalar(q: &[f64; 10]) -> (f64, [f64; 10]) {
    fn gaussian(x: f64, mean: f64, sd: f64) -> f64 {
        -0.5 * ((x - mean) / sd).powi(2) - sd.ln() - 0.5 * std::f64::consts::TAU.ln()
    }
    let mu = q[8];
    let lt = q[9];
    let tau = lt.exp();
    let mut value =
        gaussian(mu, 0., 5.) - (5. * std::f64::consts::PI).ln() - (tau * tau / 25.).ln_1p() + lt;
    let mut score = [0.; 10];
    score[8] = -mu / 25.;
    score[9] = 1. - 2. * tau * tau / (25. + tau * tau);
    for i in 0..N {
        let residual = q[i] - mu;
        let precision_residual = residual / (tau * tau);
        value += gaussian(q[i], mu, tau) + gaussian(Y[i], q[i], SIGMA[i]);
        score[i] = -precision_residual + (Y[i] - q[i]) / SIGMA[i].powi(2);
        score[8] += precision_residual;
        score[9] += -1. + (residual / tau).powi(2);
    }
    (value, score)
}

#[test]
fn centered_density_and_score_include_exactly_one_scale_jacobian()
-> Result<(), Box<dyn std::error::Error>> {
    let description = model(false)?;
    assert_eq!(description.coordinates()[0].name, "theta[1]");
    assert_eq!(description.coordinates()[8].name, "mu");
    assert_eq!(description.coordinates()[9].name, "tau");
    let target = description.into_target()?;
    for lt in [-7.0_f64, -2., 0., 2., 5.] {
        for offset in [0., 0.3, -1.7] {
            let mut q = [0.; 10];
            q[8] = 1.3;
            q[9] = lt;
            for (i, theta) in q[..N].iter_mut().enumerate() {
                *theta = 1.3 + offset * lt.exp() * (i as f64 - 3.5);
            }
            let (expected, score) = scalar(&q);
            let mut actual = [0.; 10];
            let value = target.logp_grad(&q, &mut actual)?;
            assert!((value - expected).abs() < 2e-10 * (1. + expected.abs()));
            for i in 0..10 {
                assert!(
                    (actual[i] - score[i]).abs() < 2e-10 * (1. + score[i].abs()),
                    "lt={lt}, coordinate={i}"
                );
                for step in [1e-4, 1e-5, 1e-6] {
                    let h = step * if i < N { lt.exp().min(1.) } else { 1. };
                    let mut plus = q;
                    let mut minus = q;
                    plus[i] += h;
                    minus[i] -= h;
                    let fd = (scalar(&plus).0 - scalar(&minus).0) / (2. * h);
                    assert!(
                        (fd - score[i]).abs() < 2e-5 * (1. + score[i].abs()),
                        "finite difference lt={lt}, coordinate={i}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn noncentered_change_of_variables_preserves_density_and_pullback()
-> Result<(), Box<dyn std::error::Error>> {
    let centered = model(false)?.into_target()?;
    let noncentered = model(true)?.into_target()?;
    for lt in [-7.0_f64, -2., 0., 2., 5.] {
        let z = [-1.2, 0.3, 2., -0.1, 0.7, -0.5, 1.1, -2.3, 1.3, lt];
        let mut q = z;
        for i in 0..N {
            q[i] = z[8] + lt.exp() * z[i];
        }
        let mut gc = [0.; 10];
        let mut gn = [0.; 10];
        let lc = centered.logp_grad(&q, &mut gc)?;
        let ln = noncentered.logp_grad(&z, &mut gn)?;
        // det(d theta / d z) = tau^8, separate from tau's exp Jacobian.
        assert!((ln - lc - N as f64 * lt).abs() < 2e-9 * (1. + ln.abs()));
        for i in 0..N {
            assert!((gn[i] - lt.exp() * gc[i]).abs() < 2e-9 * (1. + gn[i].abs()));
        }
        let mu_score = gc[8] + gc[..N].iter().sum::<f64>();
        let scale_score = gc[9] + N as f64 + (0..N).map(|i| gc[i] * (q[i] - z[8])).sum::<f64>();
        assert!((gn[8] - mu_score).abs() < 2e-8 * (1. + gn[8].abs()));
        assert!((gn[9] - scale_score).abs() < 2e-8 * (1. + gn[9].abs()));
        for (k, &score) in gn.iter().enumerate() {
            for h in [1e-4, 1e-5, 1e-6] {
                let mut plus = z;
                let mut minus = z;
                plus[k] += h;
                minus[k] -= h;
                let fd = (noncentered.logp_grad(&plus, &mut [0.; 10])?
                    - noncentered.logp_grad(&minus, &mut [0.; 10])?)
                    / (2. * h);
                assert!((fd - score).abs() < 2e-5 * (1. + score.abs()));
            }
        }
    }
    Ok(())
}

fn gaussian(value: f64, mean: f64, log_sd: f64) -> f64 {
    -0.5 * ((value - mean) * (-log_sd).exp()).powi(2) - log_sd - 0.5 * std::f64::consts::TAU.ln()
}

#[test]
fn covariance_block_orientation_full_density_and_all_scores_match_independent_cholesky_equations() {
    let (x, y) = conformance::covariance_data();
    let model = conformance::native("covariance", 32, 2, &x, &y).unwrap();
    assert_eq!(
        model
            .coordinates()
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>(),
        ["mu0", "mu1", "s0", "s1", "loading"]
    );
    let target = model.into_target().unwrap();
    let reference = conformance::build("covariance", 32, 2, &x, &y)
        .unwrap()
        .into_target()
        .unwrap();
    for q in [
        [0.; 5],
        [0.3, -0.7, 0.2, -0.4, 0.6],
        [-3., 2., -2., 1., -1.5],
    ] {
        let [m0, m1, l0, l1, loading] = q;
        let s0 = f64::exp(l0);
        let s1 = f64::exp(l1);
        let mut value = gaussian(m0, 0., 2_f64.ln())
            + gaussian(m1, 0., 2_f64.ln())
            + gaussian(s0, 0., 0.)
            + gaussian(s1, 0., 0.)
            + gaussian(loading, 0., 0.)
            + l0
            + l1;
        let mut gradient = [-m0 / 4., -m1 / 4., 1. - s0 * s0, 1. - s1 * s1, -loading];
        for row in x.chunks_exact(2) {
            let z0 = (row[0] - m0) / s0;
            let z1 = (row[1] - m1 - loading * z0) / s1;
            value += -0.5 * (z0 * z0 + z1 * z1) - l0 - l1 - std::f64::consts::TAU.ln();
            gradient[0] += z0 / s0 - loading * z1 / (s0 * s1);
            gradient[1] += z1 / s1;
            gradient[2] += z0 * z0 - loading * z0 * z1 / s1 - 1.;
            gradient[3] += z1 * z1 - 1.;
            gradient[4] += z0 * z1 / s1;
        }
        let mut actual = [0.; 5];
        let got = target.logp_grad(&q, &mut actual).unwrap();
        assert!((got - value).abs() < 2e-10 * (1. + value.abs()));
        assert!(
            (reference.logp_grad(&q, &mut [0.; 5]).unwrap() - value).abs()
                < 2e-10 * (1. + value.abs())
        );
        for (k, (&a, &g)) in actual.iter().zip(&gradient).enumerate() {
            assert!(
                (a - g).abs() < 2e-10 * (1. + g.abs()),
                "covariance coordinate {k}: {a} != {g}"
            );
            for h in [1e-4, 1e-5, 1e-6] {
                let mut plus = q;
                let mut minus = q;
                plus[k] += h;
                minus[k] -= h;
                let fd = (target.logp_grad(&plus, &mut [0.; 5]).unwrap()
                    - target.logp_grad(&minus, &mut [0.; 5]).unwrap())
                    / (2. * h);
                assert!((fd - g).abs() < 3e-5 * (1. + g.abs()));
            }
        }
    }
}

#[test]
fn radon_block_full_density_scores_coordinates_and_conditional_reporting() {
    use alea::model::assembly::{PointwiseLikelihood, PredictiveUnit};
    // The corpus remains external to source releases. Explicit opt-in exercises
    // the pinned actual data; the default fixture preserves repeated groups.
    let data: serde_json::Value = if let Some(path) = std::env::var_os("ALEA_RADON_DATA") {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    } else {
        serde_json::json!({"J":3,"N":5,"county_idx":[3,1,3,2,1],"floor_measure":[0.,1.,1.,0.,0.],"log_radon":[0.5,1.,-0.25,0.,0.75]})
    };
    let j = usize::try_from(data["J"].as_u64().unwrap()).unwrap();
    let county: Vec<_> = data["county_idx"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| {
            usize::try_from(i.as_u64().unwrap())
                .unwrap()
                .checked_sub(1)
                .unwrap()
        })
        .collect();
    let floor: Vec<_> = data["floor_measure"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let y: Vec<_> = data["log_radon"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(
        y.len(),
        usize::try_from(data["N"].as_u64().unwrap()).unwrap()
    );
    let model = conformance::radon(j, &county, &floor, &y).unwrap();
    assert_eq!(model.coordinates()[0].name, "alpha_raw[1]");
    assert_eq!(model.coordinates()[j].name, "beta");
    assert_eq!(model.coordinates()[j + 1].name, "mu_alpha");
    assert_eq!(model.coordinates()[j + 2].name, "sigma_alpha");
    assert_eq!(model.coordinates()[j + 3].name, "sigma_y");
    assert!(matches!(
        model.rust_primal(),
        Err(AssemblyError::MissingPrimal { .. })
    ));
    let binding = model.binding().clone();
    let target = model.into_target().unwrap();
    let scalar_model = conformance::radon_scalar(j, &county, &floor, &y).unwrap();
    assert!(scalar_model.rust_primal().is_ok()); // Eligibility, not compiled AD evidence.
    let scalar_target = scalar_model.into_target().unwrap();
    for (lt, ls) in [(0., 0.), (-5., -1.), (3., 2.)] {
        let mut q: Vec<_> = (0..j).map(|i| (i % 7) as f64 / 8. - 0.375).collect();
        q.extend([0.25, -0.5, lt, ls]);
        let beta = q[j];
        let mu = q[j + 1];
        let tau = f64::exp(lt);
        let sigma = f64::exp(ls);
        let mut value: f64 = q[..j].iter().map(|&v| gaussian(v, 0., 0.)).sum();
        value += gaussian(beta, 0., 10_f64.ln())
            + gaussian(mu, 0., 10_f64.ln())
            + gaussian(tau, 0., 0.)
            + gaussian(sigma, 0., 0.)
            + lt
            + ls;
        let mut expected: Vec<_> = q[..j].iter().map(|&x| -x).collect();
        expected.extend([-beta / 100., -mu / 100., 1. - tau * tau, 1. - sigma * sigma]);
        let mut likelihood = 0.;
        for ((&group, &floor), &y) in county.iter().zip(&floor).zip(&y) {
            let mean = mu + tau * q[group] + beta * floor;
            let z = (y - mean) / sigma;
            let score = z / sigma;
            likelihood += gaussian(y, mean, ls);
            expected[group] += score * tau;
            expected[j] += score * floor;
            expected[j + 1] += score;
            expected[j + 2] += score * tau * q[group];
            expected[j + 3] += z * z - 1.;
        }
        value += likelihood;
        let mut gradient = vec![0.; j + 4];
        assert!(
            (target.logp_grad(&q, &mut gradient).unwrap() - value).abs()
                < 2e-10 * (1. + value.abs())
        );
        let mut scalar_gradient = vec![0.; j + 4];
        assert!(
            (scalar_target.logp_grad(&q, &mut scalar_gradient).unwrap() - value).abs()
                < 2e-10 * (1. + value.abs())
        );
        for (&actual, &expected) in scalar_gradient.iter().zip(&expected) {
            assert!((actual - expected).abs() < 2e-10 * (1. + expected.abs()));
        }
        for (k, (&actual, &expected)) in gradient.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() < 2e-10 * (1. + expected.abs()),
                "radon coordinate {k}"
            );
            for h in [1e-4, 1e-5, 1e-6] {
                let mut plus = q.clone();
                let mut minus = q.clone();
                plus[k] += h;
                minus[k] -= h;
                let fd = (target.logp_grad(&plus, &mut vec![0.; j + 4]).unwrap()
                    - target.logp_grad(&minus, &mut vec![0.; j + 4]).unwrap())
                    / (2. * h);
                assert!(
                    (fd - expected).abs() < 3e-5 * (1. + expected.abs()),
                    "radon finite difference {k}: {fd} != {expected}"
                );
            }
        }
        let report = PointwiseLikelihood::new(
            binding.compile().unwrap().into_native_model(),
            vec![PredictiveUnit {
                name: "conditional observations".into(),
                factors: vec!["observations".into()],
                observations: (0..y.len()).map(|i| format!("radon{i}")).collect(),
                conditioned_on: vec![
                    "alpha_raw".into(),
                    "beta".into(),
                    "mu_alpha".into(),
                    "sigma_alpha".into(),
                    "sigma_y".into(),
                ],
                integrated_out: vec![],
            }],
        )
        .unwrap();
        let mut out = [91.];
        assert!(
            (report.evaluate(&q, &mut out).unwrap() - likelihood).abs()
                < 2e-10 * (1. + likelihood.abs())
        );
        assert_eq!(out[0], report.evaluate(&q, &mut [0.]).unwrap());
        let saved = gradient.clone();
        let mut invalid = q.clone();
        invalid[j + 3] = -1000.;
        assert!(target.logp_grad(&invalid, &mut gradient).is_err());
        assert_eq!(gradient, saved);
        assert!(target.logp_grad(&q, &mut gradient).is_ok());
    }
    assert!(conformance::radon(j, &vec![j; y.len()], &floor, &y).is_err());
}

#[test]
fn joint_schools_and_covariance_reports_keep_conditional_units_and_exclude_priors_and_charts() {
    use alea::model::assembly::{PointwiseLikelihood, PredictiveUnit};
    for name in ["schools", "schools_nc", "covariance"] {
        let (n, k, x, y, q, factors) = if name == "covariance" {
            let (x, y) = conformance::covariance_data();
            (
                32,
                2,
                x,
                y,
                vec![0.3, -0.7, 0.2, -0.4, 0.6],
                vec!["first coordinate".into(), "second conditional".into()],
            )
        } else {
            (
                8,
                1,
                SIGMA.to_vec(),
                Y.to_vec(),
                vec![-1.2, 0.3, 2., -0.1, 0.7, -0.5, 1.1, -2.3, 1.3, 0.5],
                vec!["observations".into()],
            )
        };
        let model = conformance::native(name, n, k, &x, &y).unwrap();
        let expected: f64 = if name == "covariance" {
            x.chunks_exact(2)
                .map(|row| {
                    let z0 = (row[0] - q[0]) * (-q[2]).exp();
                    let z1 = (row[1] - q[1] - q[4] * z0) * (-q[3]).exp();
                    -0.5 * (z0 * z0 + z1 * z1) - q[2] - q[3] - std::f64::consts::TAU.ln()
                })
                .sum()
        } else {
            y.iter()
                .enumerate()
                .map(|(i, &y)| {
                    gaussian(
                        y,
                        if name == "schools" {
                            q[i]
                        } else {
                            q[8] + q[9].exp() * q[i]
                        },
                        x[i].ln(),
                    )
                })
                .sum()
        };
        let report = PointwiseLikelihood::new(
            model,
            vec![PredictiveUnit {
                name: "joint data".into(),
                factors,
                observations: (0..n).map(|i| format!("observation{i}")).collect(),
                conditioned_on: vec!["model parameters".into()],
                integrated_out: vec![],
            }],
        )
        .unwrap();
        let mut out = [91.];
        assert!(
            (report.evaluate(&q, &mut out).unwrap() - expected).abs()
                < 2e-10 * (1. + expected.abs())
        );
        assert!((out[0] - expected).abs() < 2e-10 * (1. + expected.abs()));
        assert!(report.units()[0].integrated_out.is_empty());
        let mut invalid = q.clone();
        invalid[if name == "covariance" { 2 } else { 9 }] = -1000.;
        let saved = out;
        assert!(report.evaluate(&invalid, &mut out).is_err());
        assert_eq!(out, saved);
        assert!(report.evaluate(&q, &mut out).is_ok());
    }
}
