//! Conditional mixed effects retain latent coordinates: no marginalization.
use alea_core::{
    model::{
        TransformedTarget,
        assembly::{ModelBuilder, NativeModel},
    },
    target::LogDensityGradient,
};
use alea_distributions::modeling::{
    normal,
    regression::{RegressionData, normal_regression},
};

fn mixed_effects(chunk: usize) -> TransformedTarget<NativeModel> {
    let mut b = ModelBuilder::new();
    let intercept = b.real("intercept").unwrap();
    let slope = b.real("slope").unwrap();
    let sigma = b.positive("sigma").unwrap();
    let tau = b.positive("tau").unwrap();
    let mut coefficients = vec![intercept.expr(), slope.expr()];
    let zero = b.constant(0.).unwrap();
    let one = b.constant(1.).unwrap();
    for j in 0..3 {
        let z = b.real(&format!("z{j}")).unwrap();
        let prior = normal(&mut b, z.expr(), zero, one).unwrap();
        b.prior(&format!("z_prior{j}"), prior).unwrap();
        coefficients.push(b.multiply(tau.expr(), z.expr()).unwrap());
    }
    for (name, parameter) in [("sigma_prior", sigma.expr()), ("tau_prior", tau.expr())] {
        let prior = normal(&mut b, parameter, zero, one).unwrap();
        b.prior(name, prior).unwrap();
    }
    // Fixed columns [1,x,group0,group1,group2], dynamic coefficients
    // [intercept,slope,tau*z0,tau*z1,tau*z2]. Sigma and tau remain estimated.
    for start in (0..512).step_by(chunk) {
        let end = (start + chunk).min(512);
        let design: Vec<_> = (start..end)
            .flat_map(|i| {
                [
                    1.,
                    (i % 17) as f64 / 8. - 1.,
                    f64::from(i % 3 == 0),
                    f64::from(i % 3 == 1),
                    f64::from(i % 3 == 2),
                ]
            })
            .collect();
        let y = (start..end)
            .map(|i| (i as f64 * 0.37).sin() + (i % 3) as f64 / 5.)
            .collect();
        let factor = normal_regression(
            &mut b,
            &coefficients,
            &sigma,
            RegressionData::new(5, design, y).unwrap(),
        )
        .unwrap();
        b.likelihood(&format!("observations{start}"), factor)
            .unwrap();
    }
    b.compile().unwrap().into_target().unwrap()
}

#[test]
fn gaussian_statistics_preserve_noncentered_effects_scale_jacobians_and_graph_scores() {
    let compressed = mixed_effects(512);
    // Chunks below preparation threshold deliberately retain the row oracle.
    let rows = mixed_effects(64);
    for q in [
        [0.2, -0.3, 0.1, -0.4, 0.1, -0.5, 0.8],
        [-0.4, 0.7, -1., 0.8, -0.2, 0.3, 0.4],
    ] {
        let mut g = [0.; 7];
        let mut h = [0.; 7];
        let value = compressed.logp_grad(&q, &mut g).unwrap();
        let reference = rows.logp_grad(&q, &mut h).unwrap();
        for (a, b) in g.iter().chain([&value]).zip(h.iter().chain([&reference])) {
            assert!((a - b).abs() < 3e-11 * (1. + b.abs()), "{a} != {b}");
        }
        for j in 0..q.len() {
            let mut plus = q;
            let mut minus = q;
            plus[j] += 1e-5;
            minus[j] -= 1e-5;
            let fd = (compressed.logp_grad(&plus, &mut h).unwrap()
                - compressed.logp_grad(&minus, &mut h).unwrap())
                / 2e-5;
            assert!((fd - g[j]).abs() < 3e-7 * (1. + g[j].abs()));
        }
    }
}
