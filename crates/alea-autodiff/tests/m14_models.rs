#![cfg_attr(feature = "std-autodiff", feature(autodiff))]
#[path = "../../../tools/downstream-conformance/src/modeling.rs"]
mod downstream;
use alea_core::target::LogDensityGradient;
use alea_distributions::wiener::{Boundary, WienerObservation};
#[cfg(feature = "std-autodiff")]
use std::autodiff::autodiff_reverse;
const C: f64 = 0.9189385332046727;
#[cfg_attr(
    feature = "std-autodiff",
    autodiff_reverse(d_regression, Duplicated, Active)
)]
fn regression(q: &[f64]) -> f64 {
    let mut p = -0.5 * q[0] * q[0] - 0.5 * q[1] * q[1] - 2.0 * C;
    for (x, y) in [(-1.0, -0.2), (0.0, 0.4), (1.0, 1.3)] {
        let r = (y - q[0] - q[1] * x) / 0.7;
        p += -0.5 * r * r - 0.7f64.ln() - C;
    }
    p
}
#[cfg_attr(
    feature = "std-autodiff",
    autodiff_reverse(d_logistic, Duplicated, Active)
)]
fn logistic(q: &[f64]) -> f64 {
    let mut p = -0.5 * (q[0] / 0.7).powi(2)
        - 0.7f64.ln()
        - C
        - 0.5 * ((q[1] + 0.7) / 0.3).powi(2)
        - 0.3f64.ln()
        - C
        - 0.5 * q[2] * q[2]
        - 0.5 * q[3] * q[3]
        - 2.0 * C;
    for (group, y) in [(0, 1.0), (1, 0.0)] {
        let z = q[0] + q[1].exp() * q[2 + group];
        let softplus = if z > 0.0 {
            z + (-z).exp().ln_1p()
        } else {
            z.exp().ln_1p()
        };
        p += y * z - softplus;
    }
    p
}
#[test]
fn typed_lowering_agrees_with_independent_elementary_primals_and_enzyme_where_enabled() {
    let models = [
        downstream::regression(&[(-1.0, -0.2), (0.0, 0.4), (1.0, 1.3)]),
        downstream::hierarchy(&[
            downstream::Observation::Logistic {
                group: 0,
                event: true,
            },
            downstream::Observation::Logistic {
                group: 1,
                event: false,
            },
        ]),
    ];
    for (kind, model) in models.into_iter().enumerate() {
        let target = model.into_target().unwrap();
        for point in [[0.2, -0.5, 0.3, -0.4], [-0.4, -1.0, 1.0, 0.2]] {
            let q = &point[..target.dimension()];
            let mut gradient = vec![0.0; q.len()];
            let value = target.logp_grad(q, &mut gradient).unwrap();
            let primal = if kind == 0 { regression } else { logistic };
            assert!((value - primal(q)).abs() < 1e-12);
            for i in 0..q.len() {
                let mut plus = q.to_vec();
                let mut minus = q.to_vec();
                plus[i] += 1e-5;
                minus[i] -= 1e-5;
                let fd = (primal(&plus) - primal(&minus)) / 2e-5;
                assert!((gradient[i] - fd).abs() < 2e-5 * (1.0 + fd.abs()));
            }
            #[cfg(feature = "std-autodiff")]
            {
                let mut generated = vec![0.0; q.len()];
                if kind == 0 {
                    d_regression(q, &mut generated, 1.0);
                } else {
                    d_logistic(q, &mut generated, 1.0);
                }
                for (a, b) in generated.iter().zip(&gradient) {
                    assert!((a - b).abs() < 1e-10 * (1.0 + b.abs()));
                }
            }
        }
    }
    // The opaque DDM vertical remains constructible without differentiating its backend IR.
    let _ = downstream::hierarchy(&[downstream::Observation::Ddm {
        group: 0,
        event: WienerObservation {
            rt: 0.6,
            boundary: Boundary::Upper,
        },
    }]);
}
