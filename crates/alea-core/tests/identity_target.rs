//! Identity charts must remove only chart work, never checked-evaluation behavior.
use alea_core::{
    model::{AnalyticModel, ComposedObjective, ModelError, TransformedTarget},
    target::{EvaluationError, LogDensityGradient},
    transform::{ParameterLayout, Transform, TransformError},
};
use std::{cell::Cell, convert::Infallible};

#[test]
fn identity_density_and_composition_preserve_value_score_bits_and_zero_dimensions() {
    for n in [0, 1, 9, 64] {
        let layout = ParameterLayout::new((0..n).map(|_| Transform::identity(1).unwrap())).unwrap();
        let model = || {
            AnalyticModel::new(n, |x: &[f64], g: &mut [f64]| {
                for (g, x) in g.iter_mut().zip(x) {
                    *g = -*x;
                }
                Ok::<_, Infallible>(-0.5 * x.iter().map(|x| x * x).sum::<f64>())
            })
        };
        let density = TransformedTarget::new(model(), layout.clone()).unwrap();
        let composition = ComposedObjective::new(model(), layout.clone()).unwrap();
        let q: Vec<_> = (0..n)
            .map(|i| if i % 2 == 0 { -0. } else { i as f64 / 7. })
            .collect();
        let mut values = vec![0.; n];
        let mut reference = vec![0.; n];
        let jac = layout.constrain(&q, &mut values, &mut []).unwrap();
        layout
            .pullback(
                &q,
                &q.iter().map(|v| -*v).collect::<Vec<_>>(),
                &mut reference,
                &mut [],
            )
            .unwrap();
        let lp = -0.5 * values.iter().map(|x| x * x).sum::<f64>();
        for (target, value) in [
            (
                &density as &dyn LogDensityGradient<Error = ModelError<Infallible>>,
                lp + jac,
            ),
            (&composition, lp),
        ] {
            let mut actual = vec![17.; n];
            assert_eq!(
                target.logp_grad(&q, &mut actual).unwrap().to_bits(),
                value.to_bits()
            );
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn identity_fast_path_retains_input_failure_staging_incomplete_scores_and_recovery() {
    let incomplete = Cell::new(false);
    let model = AnalyticModel::new(2, |q: &[f64], g: &mut [f64]| {
        g[0] = q[0];
        if !incomplete.get() {
            g[1] = q[1];
        }
        Ok::<_, Infallible>(-1.)
    });
    let target = TransformedTarget::new(
        model,
        ParameterLayout::new([Transform::identity(1).unwrap(); 2]).unwrap(),
    )
    .unwrap();
    let mut g = [17.; 2];
    assert!(matches!(
        target.logp_grad(&[0., f64::NAN], &mut g),
        Err(ModelError::Transform(TransformError::NonFiniteInput {
            index: 1
        }))
    ));
    assert_eq!(g, [17.; 2]);
    incomplete.set(true);
    assert!(matches!(
        target.logp_grad(&[1., 2.], &mut g),
        Err(ModelError::Evaluation(EvaluationError::NonFiniteGradient {
            index: 1
        }))
    ));
    assert_eq!(g, [17.; 2]);
    incomplete.set(false);
    target.logp_grad(&[1., 2.], &mut g).unwrap();
    assert_eq!(g, [1., 2.]);
    assert!(matches!(
        target.logp_grad(&[1.], &mut g),
        Err(ModelError::Dimension)
    ));
    assert_eq!(g, [1., 2.]);
}
