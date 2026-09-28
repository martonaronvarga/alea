#![cfg_attr(feature = "std-autodiff", feature(autodiff))]
#[path = "../examples/support/models.rs"]
mod fixtures;
use kernels::{
    gradient_check::{GradientCheckOptions, check_gradient},
    model::{AnalyticModel, ConstrainedModel, TransformedTarget},
    target::{LogDensityGradient, evaluate},
    transform::{ParameterLayout, Transform},
};
use std::{cell::Cell, convert::Infallible};

#[test]
fn all_six_models_pass_gradient_diagnostics() {
    for kind in 0..5 {
        for q in [[0.3, -0.7], [-1.2, 2.3], [0.0, 0.0]] {
            let report = check_gradient(
                &fixtures::analytic_target(kind),
                &q,
                GradientCheckOptions::default(),
            )
            .unwrap();
            assert!(report.passed(), "{report:?}");
        }
    }
    let target =
        TransformedTarget::new(fixtures::wiener_primitive(), fixtures::wiener_layout()).unwrap();
    for q in [
        [0.2, -0.5, 0.3, -0.4],
        [-0.7, 0.4, -0.5, 1.2],
        [0.8, -0.1, 0.7, -1.0],
    ] {
        let report = check_gradient(&target, &q, GradientCheckOptions::default()).unwrap();
        assert!(report.passed(), "{report:?}");
    }
}

#[test]
fn fused_model_calls_once_has_aligned_scratch_and_is_allocation_free() {
    let calls = Cell::new(0);
    let model = AnalyticModel::new(7, |q: &[f64], g: &mut [f64]| {
        assert_eq!(q.as_ptr() as usize % 64, 0);
        assert_eq!(g.as_ptr() as usize % 64, 0);
        calls.set(calls.get() + 1);
        for (g, q) in g.iter_mut().zip(q) {
            *g = -q;
        }
        Ok::<_, Infallible>(-0.5 * q.iter().map(|v| v * v).sum::<f64>())
    });
    let layout = ParameterLayout::new([
        Transform::simplex(3).unwrap(),
        Transform::covariance(2).unwrap(),
    ])
    .unwrap();
    let target = TransformedTarget::new(model, layout).unwrap();
    let mut g = [0.0; 5];
    evaluate(&target, &[0.1; 5], &mut g).unwrap();
    assert_eq!(calls.get(), 1);
    #[cfg(not(miri))]
    {
        let allocation = allocation_counter::measure(|| {
            for _ in 0..64 {
                evaluate(&target, &[0.1; 5], &mut g).unwrap();
            }
        });
        assert_eq!(allocation.count_total, 0);
    }
    let report = check_gradient(&target, &[0.1; 5], GradientCheckOptions::default()).unwrap();
    assert!(report.passed(), "{report:?}");
}

#[test]
fn errors_and_panics_do_not_publish_partial_gradients() {
    let mode = Cell::new(0);
    let model = AnalyticModel::new(2, |_: &[f64], g: &mut [f64]| -> Result<f64, Infallible> {
        g[0] = 1.0;
        match mode.get() {
            0 => Ok(1.0), // second component deliberately missing
            1 => {
                g[1] = 2.0;
                Ok(f64::INFINITY)
            }
            2 => panic!("model panic"),
            _ => {
                g[1] = 2.0;
                Ok(1.0)
            }
        }
    });
    let target = TransformedTarget::new(
        model,
        ParameterLayout::new([Transform::identity(2).unwrap()]).unwrap(),
    )
    .unwrap();
    let mut g = [23.0; 2];
    for value in [0, 1] {
        mode.set(value);
        assert!(target.logp_grad(&[0.0; 2], &mut g).is_err());
        assert_eq!(g, [23.0; 2]);
    }
    mode.set(2);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || target.logp_grad(&[0.0; 2], &mut g)
        ))
        .is_err()
    );
    assert_eq!(g, [23.0; 2]);
    mode.set(3);
    target.logp_grad(&[0.0; 2], &mut g).unwrap();
    assert_eq!(g, [1.0, 2.0]);
    assert!(target.logp_grad(&[0.0], &mut g).is_err());
}

#[test]
fn diagnostics_reject_bad_configuration_and_detect_bad_gradients() {
    let target = fixtures::analytic_target(0);
    for step in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MIN_POSITIVE] {
        assert!(
            check_gradient(
                &target,
                &[1.0, 2.0],
                GradientCheckOptions {
                    relative_step: step,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
    assert!(check_gradient(&target, &[0.0], GradientCheckOptions::default()).is_err());
    let wrong = AnalyticModel::new(1, |q: &[f64], g: &mut [f64]| {
        g[0] = 0.0;
        Ok::<_, Infallible>(q[0] * q[0])
    });
    let target = TransformedTarget::new(
        wrong,
        ParameterLayout::new([Transform::identity(1).unwrap()]).unwrap(),
    )
    .unwrap();
    assert!(
        !check_gradient(&target, &[1.0], GradientCheckOptions::default())
            .unwrap()
            .passed()
    );
}

#[test]
fn changing_model_dimensions_are_rejected_before_evaluation() {
    struct Changing(Cell<usize>);
    impl ConstrainedModel for Changing {
        type Error = Infallible;
        fn dimension(&self) -> usize {
            self.0.get()
        }
        fn logp_grad(&self, _: &[f64], _: &mut [f64]) -> Result<f64, Infallible> {
            panic!("must not execute")
        }
    }
    let target = TransformedTarget::new(
        Changing(Cell::new(1)),
        ParameterLayout::new([Transform::identity(1).unwrap()]).unwrap(),
    )
    .unwrap();
    target.model().0.set(2);
    assert!(target.logp_grad(&[0.0], &mut [0.0]).is_err());
}
