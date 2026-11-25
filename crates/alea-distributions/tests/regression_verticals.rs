use alea_core::{
    capability::PotentialHessianVector,
    model::assembly::{AssemblyError, ModelBuilder},
    target::LogDensityGradient,
};
use alea_distributions::modeling::regression::*;

fn close(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() <= tolerance * (1. + b.abs()), "{a} != {b}");
}
#[test]
fn native_glm_hvps_match_gradient_differences_symmetry_and_error_recovery() {
    for poisson in [false, true] {
        for n in [1, 3, 4, 7, 127, 1000] {
            let design: Vec<_> = (0..n)
                .flat_map(|i| [1., (i % 5) as f64 * 0.25 - 0.5])
                .collect();
            let mut b = ModelBuilder::new();
            let beta = [b.real("a").unwrap().expr(), b.real("b").unwrap().expr()];
            let (target, term) = if poisson {
                let data = RegressionData::new(2, design, vec![2; n]).unwrap();
                (
                    GlmLikelihood::poisson_log(data.clone()).unwrap(),
                    poisson_log_regression(&mut b, &beta, data).unwrap(),
                )
            } else {
                let data =
                    RegressionData::new(2, design, (0..n).map(|i| i % 2 == 0).collect()).unwrap();
                (
                    GlmLikelihood::bernoulli_logit(data.clone()).unwrap(),
                    bernoulli_logit_regression(&mut b, &beta, data).unwrap(),
                )
            };
            b.likelihood("data", term).unwrap();
            let original = b.compile().unwrap().into_objective().unwrap();
            let x = [0.2, -0.4];
            let v = [0.75, -0.25];
            let mut a = [0.; 2];
            let mut g = [0.; 2];
            assert_eq!(
                original.logp_grad(&x, &mut a).unwrap().to_bits(),
                target.logp_grad(&x, &mut g).unwrap().to_bits()
            );
            assert_eq!(a.map(f64::to_bits), g.map(f64::to_bits));
            let mut h = [0.; 2];
            target.potential_hvp(&x, &v, &mut h).unwrap();
            let eps = 1e-5;
            let plus: [f64; 2] = std::array::from_fn(|j| x[j] + eps * v[j]);
            let minus: [f64; 2] = std::array::from_fn(|j| x[j] - eps * v[j]);
            target.logp_grad(&plus, &mut a).unwrap();
            target.logp_grad(&minus, &mut g).unwrap();
            for j in 0..2 {
                close(h[j], -(a[j] - g[j]) / (2. * eps), 1e-8);
            }
            let mut c0 = [0.; 2];
            let mut c1 = [0.; 2];
            target.potential_hvp(&x, &[1., 0.], &mut c0).unwrap();
            target.potential_hvp(&x, &[0., 1.], &mut c1).unwrap();
            close(c0[1], c1[0], 1e-13);
            assert!(v[0] * h[0] + v[1] * h[1] >= 0.);
            assert_eq!(
                allocation_counter::measure(|| {
                    target.potential_hvp(&x, &v, &mut h).unwrap();
                })
                .count_total,
                0
            );
            h.fill(99.);
            assert!(matches!(
                target.potential_hvp(&[f64::NAN, 0.], &v, &mut h),
                Err(AssemblyError::Numerical)
            ));
            assert_eq!(h, [99.; 2]);
            assert!(target.potential_hvp(&x, &[0.], &mut h).is_err());
            assert_eq!(h, [99.; 2]);
            target.potential_hvp(&x, &[0., 0.], &mut h).unwrap();
            assert_eq!(h, [0.; 2]);
        }
    }
    let data = RegressionData::new(1, vec![1.], vec![true]).unwrap();
    let logit = GlmLikelihood::bernoulli_logit(data).unwrap();
    for eta in [-740., -100., 0., 100., 740.] {
        let mut h = [0.];
        logit.potential_hvp(&[eta], &[1.], &mut h).unwrap();
        assert!(h[0] > 0. && h[0] <= 0.25);
    }
    let poisson =
        GlmLikelihood::poisson_log(RegressionData::new(1, vec![1.], vec![2]).unwrap()).unwrap();
    let mut out = [99.];
    assert!(poisson.potential_hvp(&[1000.], &[0.], &mut out).is_err());
    assert_eq!(out, [99.]);
    poisson.potential_hvp(&[-1000.], &[1.], &mut out).unwrap();
    assert_eq!(out, [0.]);
}

#[test]
fn typed_map_sum_uses_one_existing_native_node_and_preserves_all_three_families() {
    for n in [1, 7, 10000] {
        for family in 0..3 {
            let mut b = ModelBuilder::new();
            let beta = [b.real("beta").unwrap().expr()];
            let sigma = b.positive("sigma").unwrap();
            let data = RegressionData::new(1, vec![1.; n], vec![0.5; n]).unwrap();
            let term = match family {
                0 => data
                    .linear_map(&beta)
                    .unwrap()
                    .normal(&sigma)
                    .unwrap()
                    .sum_log_prob(&mut b),
                1 => data
                    .with_observations(vec![true; n])
                    .unwrap()
                    .linear_map(&beta)
                    .unwrap()
                    .bernoulli_logit()
                    .unwrap()
                    .sum_log_prob(&mut b),
                _ => data
                    .with_observations(vec![2_u64; n])
                    .unwrap()
                    .linear_map(&beta)
                    .unwrap()
                    .poisson_log()
                    .unwrap()
                    .sum_log_prob(&mut b),
            }
            .unwrap();
            b.likelihood("data", term).unwrap();
            let model = b.compile().unwrap();
            assert_eq!(model.expression_count(), 3);
            let pointwise = model.batch_likelihood().unwrap();
            let mut rows = vec![0.; n];
            let total = pointwise.evaluate(&[0.2, 0.3], &mut rows).unwrap();
            close(total, rows.iter().sum(), 1e-11);
        }
    }
    let mut b = ModelBuilder::new();
    let x = b.real("foreign").unwrap().expr();
    let data = RegressionData::new(1, vec![1.], vec![true]).unwrap();
    assert!(data.clone().linear_map(&[]).is_err());
    let map = data.linear_map(&[x]).unwrap().bernoulli_logit().unwrap();
    assert!(matches!(
        map.sum_log_prob(&mut ModelBuilder::new()),
        Err(AssemblyError::ForeignHandle)
    ));
}

#[test]
fn broadcast_intercept_matches_declared_unit_column_for_all_laws() {
    use alea_core::model::{
        ConstrainedModel,
        assembly::{NativeModel, Value},
    };

    fn model(family: usize, broadcast: bool) -> (NativeModel, Value) {
        let mut b = ModelBuilder::new();
        let beta = b.real("beta").unwrap().expr();
        let sigma = b.positive("sigma").unwrap();
        let (columns, design) = if broadcast {
            (1, vec![0.5, -1., 2.])
        } else {
            (2, vec![0.5, 1., -1., 1., 2., 1.])
        };
        let data = RegressionData::new(columns, design, vec![-0.5, 0.75, 2.])
            .unwrap()
            .with_offsets(vec![0.25, -0.75, 0.125])
            .unwrap();
        let mut last = None;
        for name in ["rows", "replicate"] {
            // The same expression feeds the slope and intercept. The call must
            // keep the two local scores and add both graph contributions.
            let term = match (family, broadcast) {
                (0, true) => data
                    .clone()
                    .linear_map_with_intercept(&[beta], beta)
                    .unwrap()
                    .normal(&sigma)
                    .unwrap()
                    .sum_log_prob(&mut b),
                (0, false) => normal_regression(&mut b, &[beta, beta], &sigma, data.clone()),
                (1, true) => data
                    .with_observations(vec![true, false, true])
                    .unwrap()
                    .linear_map_with_intercept(&[beta], beta)
                    .unwrap()
                    .bernoulli_logit()
                    .unwrap()
                    .sum_log_prob(&mut b),
                (1, false) => bernoulli_logit_regression(
                    &mut b,
                    &[beta, beta],
                    data.with_observations(vec![true, false, true]).unwrap(),
                ),
                (_, true) => data
                    .with_observations(vec![0_u64, 3, 7])
                    .unwrap()
                    .linear_map_with_intercept(&[beta], beta)
                    .unwrap()
                    .poisson_log()
                    .unwrap()
                    .sum_log_prob(&mut b),
                (_, false) => poisson_log_regression(
                    &mut b,
                    &[beta, beta],
                    data.with_observations(vec![0_u64, 3, 7]).unwrap(),
                ),
            }
            .unwrap();
            b.likelihood(name, term).unwrap();
            last = Some(term);
        }
        (b.compile().unwrap(), last.unwrap())
    }

    for family in 0..3 {
        let (explicit, explicit_term) = model(family, false);
        let (broadcast, broadcast_term) = model(family, true);
        assert_eq!(broadcast.expression_count(), 4);
        let a = explicit.batch_likelihood().unwrap();
        let b = broadcast.batch_likelihood().unwrap();
        assert_eq!(a.units(), b.units());
        let (mut row_a, mut row_b) = ([0.; 6], [0.; 6]);
        let (mut g_a, mut g_b) = ([0.; 2], [0.; 2]);
        for q in [[0.125, 1.], [-0.75, 0.5], [1., 2.]] {
            let lp = b.model().logp_grad(&q, &mut g_b).unwrap();
            assert_eq!(
                lp.to_bits(),
                a.model().logp_grad(&q, &mut g_a).unwrap().to_bits()
            );
            assert_eq!(g_a.map(f64::to_bits), g_b.map(f64::to_bits));
            let free = [q[0], q[1].ln()];
            close(lp, b.evaluate(&free, &mut row_b).unwrap(), 3e-13);
            a.evaluate(&free, &mut row_a).unwrap();
            assert_eq!(row_a.map(f64::to_bits), row_b.map(f64::to_bits));
        }
        let rebound = |model: &NativeModel, term| {
            let binding = model.binding();
            let offsets = || vec![0.75, -1.25, 0.5];
            match family {
                0 => binding.rebind(term, |law: &NormalRegression| {
                    law.with_data(law.data().clone().with_offsets(offsets()).unwrap())
                }),
                1 => binding.rebind(term, |law: &BernoulliLogitRegression| {
                    law.with_data(law.data().clone().with_offsets(offsets()).unwrap())
                }),
                _ => binding.rebind(term, |law: &PoissonLogRegression| {
                    law.with_data(law.data().clone().with_offsets(offsets()).unwrap())
                }),
            }
            .unwrap()
            .compile()
            .unwrap()
            .into_native_model()
            .batch_likelihood()
            .unwrap()
        };
        let (rebound_a, rebound_b) = (
            rebound(a.model(), explicit_term),
            rebound(b.model(), broadcast_term),
        );
        let q = [0.125, 1.];
        let lp = rebound_b.model().logp_grad(&q, &mut g_b).unwrap();
        assert_eq!(
            lp.to_bits(),
            rebound_a.model().logp_grad(&q, &mut g_a).unwrap().to_bits()
        );
        assert_eq!(g_a.map(f64::to_bits), g_b.map(f64::to_bits));
        assert_ne!(
            lp.to_bits(),
            b.model().logp_grad(&q, &mut g_b).unwrap().to_bits()
        );
        rebound_a.evaluate(&[q[0], 0.], &mut row_a).unwrap();
        rebound_b.evaluate(&[q[0], 0.], &mut row_b).unwrap();
        assert_eq!(row_a.map(f64::to_bits), row_b.map(f64::to_bits));
        if family != 1 {
            let bad = if family == 0 { [0., -1.] } else { [1000., 1.] };
            g_b = [19.; 2];
            assert!(b.model().logp_grad(&bad, &mut g_b).is_err());
            assert_eq!(g_b, [19.; 2]);
            assert!(b.model().logp_grad(&[0.125, 1.], &mut g_b).is_ok());
        }
    }
}
