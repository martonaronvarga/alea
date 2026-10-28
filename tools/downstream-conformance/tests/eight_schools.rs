//! Independent measure/score checks for the centered eight-schools review case.
//! These are target checks, not evidence that a sampling policy converges.
use alea::{
    distributions::modeling::normal,
    model::assembly::{AssemblyError, CompiledModel, ModelBuilder},
    target::LogDensityGradient,
    transform::Transform,
};

const Y: [f64; 8] = [28., 8., -3., 7., -1., 1., 18., 12.];
const SIGMA: [f64; 8] = [15., 10., 16., 11., 9., 11., 10., 18.];
const N: usize = 8;

fn model(noncentered: bool) -> Result<CompiledModel, AssemblyError> {
    let mut b = ModelBuilder::new();
    let effects = b.vector::<alea::model::assembly::Real>(
        if noncentered { "z" } else { "theta" },
        "school",
        &["1", "2", "3", "4", "5", "6", "7", "8"],
        "effect",
        Transform::identity(N)?,
    )?;
    let mu = b.real("mu")?;
    let tau = b.positive("tau")?;
    let zero = b.constant(0.)?;
    let one = b.constant(1.)?;
    let five = b.constant(5.)?;
    let prior = normal(&mut b, mu.expr(), zero, five)?;
    b.factor(prior)?;
    // Stan's Cauchy density restricted to tau > 0. No half-Cauchy
    // renormalization: it would add a harmless but detectable log(2).
    let square = b.multiply(tau.expr(), tau.expr())?;
    let inverse_scale_squared = b.constant(1. / 25.)?;
    let scaled = b.multiply(square, inverse_scale_squared)?;
    let denominator = b.add(one, scaled)?;
    let log_denominator = b.log(denominator)?;
    let minus_one = b.constant(-1.)?;
    let kernel = b.multiply(minus_one, log_denominator)?;
    let constant = b.constant(-(5. * std::f64::consts::PI).ln())?;
    let prior = b.add(kernel, constant)?;
    b.factor(prior)?;
    for (i, &effect) in effects.expressions().iter().enumerate() {
        let (theta, prior) = if noncentered {
            let deviation = b.multiply(tau.expr(), effect)?;
            (
                b.add(mu.expr(), deviation)?,
                normal(&mut b, effect, zero, one)?,
            )
        } else {
            (effect, normal(&mut b, effect, mu.expr(), tau.expr())?)
        };
        b.factor(prior)?;
        let observation = b.constant(Y[i])?;
        let sigma = b.constant(SIGMA[i])?;
        let likelihood = normal(&mut b, observation, theta, sigma)?;
        b.factor(likelihood)?;
    }
    b.compile()
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
                let h = 1e-5 * if i < N { lt.exp().min(1.) } else { 1. };
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
    }
    Ok(())
}
