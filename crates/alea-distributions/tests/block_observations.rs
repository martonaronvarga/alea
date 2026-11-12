//! Production shared block composition; independent formulas here are test oracles.
use alea_core::{
    model::{
        assembly::{AssemblyError as E, ModelBuilder, NativeModel},
        tensor::{BlockRules, BlockType, ElementType, Shape},
    },
    target::LogDensityGradient,
};
use alea_distributions::{
    modeling::{self, observations},
    scalar::{BernoulliLogit, Normal, PoissonLog},
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
fn shape(d: &[usize]) -> Shape {
    Shape::new(d.to_vec()).unwrap()
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 3e-11 * (1. + b.abs()), "{a} != {b}");
}
fn data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<bool>) {
    (
        (0..n)
            .flat_map(|i| [1., (i as f64 - n as f64 / 2.) / (n + 1) as f64])
            .collect(),
        (0..n).map(|i| (i as f64 * 0.7).sin()).collect(),
        (0..n).map(|i| i % 3 != 1).collect(),
    )
}
fn poisson_counts(n: usize) -> Vec<u64> {
    (0..n).map(|i| (i % 5) as u64).collect()
}
fn poisson_offsets(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i % 3) as f64 * 0.07 - 0.07).collect()
}
// A small-count oracle, independent of the production log-gamma normalizer.
fn log_factorial(k: u64) -> f64 {
    (2..=k).map(|j| (j as f64).ln()).sum()
}
fn add_offsets(
    b: &mut ModelBuilder,
    value: alea_core::model::assembly::Value,
    offsets: Vec<f64>,
) -> alea_core::model::assembly::Value {
    use alea_core::model::tensor::BinaryOp;
    let shape = shape(&[offsets.len()]);
    let fixed = b.constant_block(shape.clone(), offsets).unwrap();
    let mut r = b.region();
    let x = r.input("predictor", shape.clone()).unwrap();
    let o = r.input("offsets", shape).unwrap();
    let sum = r.binary(BinaryOp::Add, x, o).unwrap();
    let binding = r.compile(sum).unwrap().bind(&[]).unwrap();
    b.call_region(&[value, fixed], binding).unwrap()
}
fn build(n: usize, family: u8) -> (NativeModel, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let (design, y, binary) = data(n);
    let mut b = ModelBuilder::new();
    let beta = b.real_vector("beta", &["a", "b"]).unwrap();
    let beta = b.vector_value(&beta, shape(&[2, 1])).unwrap();
    let rho = b.real("rho").unwrap().expr();
    let sigma = b.exp(rho).unwrap();
    let x = b.constant_block(shape(&[n, 2]), design).unwrap();
    let eta = b.matmul(x, beta).unwrap();
    let eta = b.reshape(eta, shape(&[n])).unwrap();
    let maps = Arc::new(AtomicUsize::new(0));
    let mapped = b
        .register_block(
            &[eta],
            Arc::clone(&maps),
            BlockRules {
                inputs: vec![BlockType {
                    dtype: ElementType::Real,
                    shape: shape(&[n]),
                }],
                output: shape(&[n]),
                workspace: |_| (),
                heap_bytes: 0,
                forward: |count, inputs, out, _| {
                    count.fetch_add(1, Ordering::Relaxed);
                    for (out, &x) in out.iter_mut().zip(inputs.real(0)?) {
                        *out = x + 0.1 * x * x;
                    }
                    Ok(())
                },
                vjp: Some(|_, inputs, _, seed, g, _| {
                    for ((g, &x), &seed) in g.iter_mut().zip(inputs.real(0)?).zip(seed) {
                        *g = seed * (1. + 0.2 * x);
                    }
                    Ok(())
                }),
            },
        )
        .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    if family & 1 != 0 {
        let count = Arc::clone(&calls);
        let rows = observations::<Normal, _, _>(&mut b, y, &[mapped, sigma], move |p| {
            count.fetch_add(1, Ordering::Relaxed);
            Normal::new(p[0], p[1])
        })
        .unwrap();
        let joint = b.sum_value(rows).unwrap();
        b.likelihood("normal", joint).unwrap();
    }
    if family & 2 != 0 {
        let count = Arc::clone(&calls);
        let rows = observations::<BernoulliLogit, _, _>(&mut b, binary, &[mapped], move |p| {
            count.fetch_add(1, Ordering::Relaxed);
            BernoulliLogit::new(p[0])
        })
        .unwrap();
        let joint = b.sum_value(rows).unwrap();
        b.likelihood("logit", joint).unwrap();
    }
    if family & 4 != 0 {
        let count = Arc::clone(&calls);
        let log_rate = add_offsets(&mut b, mapped, poisson_offsets(n));
        let rows =
            observations::<PoissonLog, _, _>(&mut b, poisson_counts(n), &[log_rate], move |p| {
                count.fetch_add(1, Ordering::Relaxed);
                PoissonLog::new(p[0])
            })
            .unwrap();
        let joint = b.sum_value(rows).unwrap();
        b.likelihood("poisson", joint).unwrap();
    }
    (b.compile().unwrap(), calls, maps)
}
fn oracle(n: usize, q: &[f64], family: u8) -> (f64, [f64; 3]) {
    let (x, y, binary) = data(n);
    let sigma = q[2].exp();
    let counts = poisson_counts(n);
    let offsets = poisson_offsets(n);
    let mut value = 0.;
    let mut g = [0.; 3];
    for (i, row) in x.chunks_exact(2).enumerate() {
        let eta = row[0] * q[0] + row[1] * q[1];
        let mu = eta + 0.1 * eta * eta;
        let mut score = 0.;
        if family & 1 != 0 {
            let z = (y[i] - mu) / sigma;
            value += -0.5 * z * z - q[2] - 0.5 * std::f64::consts::TAU.ln();
            score += z / sigma;
            g[2] += z * z - 1.;
        }
        if family & 2 != 0 {
            let z = if binary[i] { -mu } else { mu };
            value -= z.max(0.) + (-z.abs()).exp().ln_1p();
            let p = if mu >= 0. {
                1. / (1. + (-mu).exp())
            } else {
                let e = mu.exp();
                e / (1. + e)
            };
            score += f64::from(binary[i]) - p;
        }
        if family & 4 != 0 {
            let k = counts[i];
            let log_rate = mu + offsets[i];
            let rate = log_rate.exp();
            value += k as f64 * log_rate - rate - log_factorial(k);
            score += k as f64 - rate;
        }
        for j in 0..2 {
            g[j] += score * (1. + 0.2 * eta) * row[j];
        }
    }
    (value, g)
}
#[test]
fn three_families_share_production_predictor_and_match_independent_answers() {
    for n in [0, 1, 3, 4, 5, 17, 257] {
        for family in [1, 2, 3, 4, 5, 6, 7] {
            let (model, calls, maps) = build(n, family);
            let target = model.into_target().unwrap();
            let q = [0.3, -0.7, -0.2];
            let mut g = [99.; 3];
            let (value, expected) = oracle(n, &q, family);
            close(target.logp_grad(&q, &mut g).unwrap(), value);
            for (a, b) in g.into_iter().zip(expected) {
                close(a, b);
            }
            assert_eq!(
                calls.load(Ordering::Relaxed),
                n * family.count_ones() as usize
            );
            assert_eq!(maps.load(Ordering::Relaxed), 1); // all consumers share one forward
            for h in [1e-4, 1e-5, 1e-6] {
                for j in 0..3 {
                    let mut lo = q;
                    let mut hi = q;
                    lo[j] -= h;
                    hi[j] += h;
                    let fd = (target.logp_grad(&hi, &mut [0.; 3]).unwrap()
                        - target.logp_grad(&lo, &mut [0.; 3]).unwrap())
                        / (2. * h);
                    assert!((fd - g[j]).abs() < 2e-6 * (1. + g[j].abs()));
                }
            }
            let a = allocation_counter::measure(|| {
                for _ in 0..3 {
                    target.logp_grad(&q, &mut g).unwrap();
                }
            });
            assert_eq!(a.count_total, 0);
        }
    }
}
#[test]
fn shapes_aliases_source_rejection_and_recovery_remain_explicit() {
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    let mut other = ModelBuilder::new();
    let foreign = other.real("x").unwrap().expr();
    assert!(matches!(
        observations::<Normal, _, _>(&mut b, vec![0.], &[foreign, x], |p| Normal::new(p[0], p[1])),
        Err(E::ForeignHandle)
    ));
    let wrong = b.constant_block(shape(&[2, 1]), vec![0.; 2]).unwrap();
    assert!(matches!(
        observations::<BernoulliLogit, _, _>(&mut b, vec![true, true], &[wrong], |p| {
            BernoulliLogit::new(p[0])
        }),
        Err(E::Shape)
    ));
    let rows = observations::<Normal, _, _>(&mut b, vec![0.2, -0.3, 0.7], &[x, x], |p| {
        Normal::new(p[0], p[1])
    })
    .unwrap();
    let sum = b.sum_value(rows).unwrap();
    b.likelihood("observed", sum).unwrap();
    let model = b.compile().unwrap();
    assert!(!model.source_capabilities().supports_rust_primal());
    assert!(matches!(model.rust_primal(), Err(E::MissingPrimal { .. })));
    let target = model.into_target().unwrap();
    let mut g = [99.];
    assert!(target.logp_grad(&[-1.], &mut g).is_err());
    assert!(target.logp_grad(&[f64::NAN], &mut g).is_err());
    let q = 0.8;
    let actual = target.logp_grad(&[q], &mut g).unwrap();
    let mut value = 0.;
    let mut grad = 0.;
    for y in [0.2, -0.3, 0.7] {
        let z = (y - q) / q;
        value += -0.5 * z * z - q.ln() - 0.5 * std::f64::consts::TAU.ln();
        grad += z / q + (z * z - 1.) / q;
    }
    close(actual, value);
    close(g[0], grad);
    let mut scalar = ModelBuilder::new();
    let x = scalar.real("x").unwrap().expr();
    let term = modeling::observation::<BernoulliLogit, _, _>(&mut scalar, true, &[x], |p| {
        BernoulliLogit::new(p[0])
    })
    .unwrap();
    scalar.factor(term).unwrap();
    assert!(
        scalar
            .compile()
            .unwrap()
            .into_target()
            .unwrap()
            .logp_grad(&[0.], &mut [0.])
            .is_ok()
    );
}

#[test]
fn incomplete_scores_are_rejected_under_zero_seed_and_cached_workspace_recovers() {
    use alea_core::distribution::{
        Distribution, DistributionError, EventShape, ParameterScore, ReferenceMeasure, Support,
    };
    use alea_core::{model::ModelError, target::EvaluationError};
    struct Faulty {
        law: BernoulliLogit,
        mode: usize,
    }
    impl Distribution for Faulty {
        type Event = bool;
        type Error = DistributionError;
        fn event_shape(&self) -> EventShape {
            self.law.event_shape()
        }
        fn support(&self) -> Support {
            self.law.support()
        }
        fn reference_measure(&self) -> ReferenceMeasure {
            self.law.reference_measure()
        }
        fn log_prob(&self, x: &bool) -> Result<f64, Self::Error> {
            self.law.log_prob(x)
        }
    }
    impl ParameterScore for Faulty {
        fn parameter_count(&self) -> usize {
            if self.mode == 2 { 2 } else { 1 }
        }
        fn parameter_score(&self, x: &bool, g: &mut [f64]) -> Result<(), Self::Error> {
            self.law.parameter_score(x, g)
        }
        fn log_prob_parameter_score(&self, x: &bool, g: &mut [f64]) -> Result<f64, Self::Error> {
            if self.mode == 1 {
                *g.first_mut().unwrap() = f64::NAN;
                self.law.log_prob(x)
            } else if self.mode == 3 {
                self.law.log_prob(x)
            } else {
                self.law.log_prob_parameter_score(x, g)
            }
        }
    }
    let flag = Arc::new(AtomicUsize::new(0));
    let copy = Arc::clone(&flag);
    let mut b = ModelBuilder::new();
    let beta = b.real_vector("x", &["a", "b"]).unwrap();
    let x = b.vector_value(&beta, shape(&[2])).unwrap();
    let rows = observations::<Faulty, _, _>(&mut b, vec![true, false], &[x], move |p| {
        Ok(Faulty {
            law: BernoulliLogit::new(p[0])?,
            mode: if p[0] > 0. {
                copy.load(Ordering::Relaxed)
            } else {
                0
            },
        })
    })
    .unwrap();
    let sum = b.sum_value(rows).unwrap();
    let zero = b.constant(0.).unwrap();
    let term = b.multiply(sum, zero).unwrap();
    b.factor(term).unwrap();
    let target = b.compile().unwrap().into_target().unwrap();
    let mut g = [91.; 2];
    for mode in [0, 1, 0, 2, 0, 3, 0] {
        flag.store(mode, Ordering::Relaxed);
        g.fill(91.);
        match target.logp_grad(&[-0.2, 0.3], &mut g) {
            Ok(value) => {
                assert_eq!(mode, 0);
                assert_eq!(value, 0.);
                assert_eq!(g, [0.; 2]);
            }
            Err(ModelError::Evaluation(EvaluationError::Model(E::Factor(source)))) => {
                assert!(matches!(
                    source.downcast_ref::<DistributionError>(),
                    Some(DistributionError::Shape)
                ));
                assert_eq!(mode, 2);
                assert_eq!(g, [91.; 2]);
            }
            Err(ModelError::Evaluation(EvaluationError::Model(E::Numerical))) => {
                assert!(mode == 1 || mode == 3);
                assert_eq!(g, [91.; 2]);
            }
            result => panic!("mode {mode}: {result:?}"),
        }
    }
}

#[test]
fn block_likelihoods_preserve_selection_joint_reporting_and_one_positive_chart() {
    use alea_core::model::assembly::{PointwiseLikelihood, PredictiveUnit};
    let mut b = ModelBuilder::new();
    let mu = b.real("mu").unwrap().expr();
    let sigma = b.positive("sigma").unwrap();
    let prior = modeling::normal(&mut b, mu, 0., 2.).unwrap();
    b.prior("prior", prior).unwrap();
    let normal = observations::<Normal, _, _>(&mut b, vec![0.2, -0.3], &[mu, sigma.expr()], |p| {
        Normal::new(p[0], p[1])
    })
    .unwrap();
    let normal = b.sum_value(normal).unwrap();
    b.likelihood("normal", normal).unwrap();
    let logit = observations::<BernoulliLogit, _, _>(&mut b, vec![true, false], &[mu], |p| {
        BernoulliLogit::new(p[0])
    })
    .unwrap();
    let logit = b.sum_value(logit).unwrap();
    b.likelihood("logit", logit).unwrap();
    let mut model = b.compile().unwrap();
    let reference = |q: [f64; 2], selection: u8| {
        let sigma = q[1].exp();
        let mut lp = 0.;
        let mut g = [0.; 2];
        if selection != 1 {
            lp += -0.5 * (q[0] / 2.).powi(2) - 2_f64.ln() - 0.5 * std::f64::consts::TAU.ln();
            g[0] -= q[0] / 4.;
        }
        if selection != 0 {
            for y in [0.2, -0.3] {
                let z = (y - q[0]) / sigma;
                lp += -0.5 * z * z - q[1] - 0.5 * std::f64::consts::TAU.ln();
                g[0] += z / sigma;
                g[1] += z * z - 1.;
            }
            lp -= (1. + (-q[0]).exp()).ln() + (1. + q[0].exp()).ln();
            g[0] += 1. - 2. / (1. + (-q[0]).exp());
        }
        (lp, g)
    };
    for selection in [0, 1, 2, 1, 0, 2] {
        model = match selection {
            0 => model.select_prior().unwrap(),
            1 => model.select_likelihood().unwrap(),
            _ => model.select_posterior().unwrap(),
        };
        assert_eq!(
            model.source_capabilities().supports_rust_primal(),
            selection == 0
        );
        let objective = model.into_objective().unwrap();
        for q in [[0.3, -0.2], [-0.7, 0.1]] {
            let (v, g) = reference(q, selection);
            let mut actual = [99.; 2];
            close(objective.logp_grad(&q, &mut actual).unwrap(), v);
            for (a, b) in actual.into_iter().zip(g) {
                close(a, b);
            }
        }
        let target = objective.into_model().into_target().unwrap();
        for q in [[0.3, -0.2], [-0.7, 0.1]] {
            let (v, g) = reference(q, selection);
            let mut actual = [99.; 2];
            close(target.logp_grad(&q, &mut actual).unwrap(), v + q[1]);
            close(actual[0], g[0]);
            close(actual[1], g[1] + 1.);
        }
        model = target.into_model();
    }
    let pointwise = PointwiseLikelihood::new(
        model,
        vec![PredictiveUnit {
            name: "joint data".into(),
            factors: vec!["normal".into(), "logit".into()],
            observations: vec!["continuous rows".into(), "binary rows".into()],
            conditioned_on: vec!["mu".into(), "sigma".into()],
            integrated_out: vec![],
        }],
    )
    .unwrap();
    let q = [0.3, -0.2];
    let mut out = [99.];
    let expected = reference(q, 1).0;
    close(pointwise.evaluate(&q, &mut out).unwrap(), expected);
    close(out[0], expected);
}

#[test]
fn shared_row_outputs_accumulate_nonuniform_and_zero_seeds_without_law_replay() {
    use alea_core::model::tensor::BinaryOp;
    for family in [1, 2, 4] {
        let calls = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&calls);
        let mut b = ModelBuilder::new();
        let mu = b.real("mu").unwrap().expr();
        let one = b.constant(1.).unwrap();
        let rows = if family == 1 {
            observations::<Normal, _, _>(&mut b, vec![0.2, -0.3, 0.7], &[mu, one], move |p| {
                count.fetch_add(1, Ordering::Relaxed);
                Normal::new(p[0], p[1])
            })
            .unwrap()
        } else if family == 2 {
            observations::<BernoulliLogit, _, _>(&mut b, vec![true, false, true], &[mu], move |p| {
                count.fetch_add(1, Ordering::Relaxed);
                BernoulliLogit::new(p[0])
            })
            .unwrap()
        } else {
            observations::<PoissonLog, _, _>(&mut b, vec![0_u64, 2, 4], &[mu], move |p| {
                count.fetch_add(1, Ordering::Relaxed);
                PoissonLog::new(p[0])
            })
            .unwrap()
        };
        let mut r = b.region();
        let input = r.input("rows", shape(&[3])).unwrap();
        let weights = r.constant(shape(&[3]), vec![-1., -0.5, 2.]).unwrap();
        let weighted = r.binary(BinaryOp::Multiply, input, weights).unwrap();
        let sum = r.sum(weighted).unwrap();
        let binding = r.compile(sum).unwrap().bind(&[]).unwrap();
        let weighted = b.call_region(&[rows], binding).unwrap();
        let plain = b.sum_value(rows).unwrap();
        let term = b.add(plain, weighted).unwrap();
        b.factor(term).unwrap();
        let target = b.compile().unwrap().into_target().unwrap();
        let q = 0.3_f64;
        let mut g = [99.];
        let actual = target.logp_grad(&[q], &mut g).unwrap();
        let mut expected = 0.;
        let mut score = 0.;
        for (i, w) in [0., 0.5, 3.].into_iter().enumerate() {
            if family == 1 {
                let z = [0.2, -0.3, 0.7][i] - q;
                expected += w * (-0.5 * z * z - 0.5 * std::f64::consts::TAU.ln());
                score += w * z;
            } else if family == 2 {
                let event = [true, false, true][i];
                let z = if event { -q } else { q };
                expected -= w * (1. + z.exp()).ln();
                score += w * (f64::from(event) - 1. / (1. + (-q).exp()));
            } else {
                let k = [0_u64, 2, 4][i];
                expected += w * (k as f64 * q - q.exp() - log_factorial(k));
                score += w * (k as f64 - q.exp());
            }
        }
        close(actual, expected);
        close(g[0], score);
        assert_eq!(calls.load(Ordering::Relaxed), 3);
        assert_eq!(
            build(1, family).0.graph().count(),
            build(257, family).0.graph().count()
        );
    }
}

#[test]
fn scalar_observation_preserves_fused_value_bits_including_signed_zero() {
    use alea_core::distribution::ParameterScore;
    for q in [-1000., 1000.] {
        let expected = BernoulliLogit::new(q)
            .unwrap()
            .log_prob_parameter_score(&true, &mut [0.])
            .unwrap()
            .to_bits();
        let mut b = ModelBuilder::new();
        let x = b.real("x").unwrap().expr();
        let term = modeling::observation::<BernoulliLogit, _, _>(&mut b, true, &[x], |p| {
            BernoulliLogit::new(p[0])
        })
        .unwrap();
        let checked = b
            .custom(&[term], move |x, g| {
                assert_eq!(x[0].to_bits(), expected);
                g[0] = 0.;
                Ok(1.)
            })
            .unwrap();
        b.factor(checked).unwrap();
        let target = b.compile().unwrap().into_target().unwrap();
        assert_eq!(target.logp_grad(&[q], &mut [0.]).unwrap(), 1.);
    }
}

// Test construction only: the production adapter and law remain unchanged.
fn poisson_rows_model(counts: Vec<u64>, zero: bool) -> NativeModel {
    let mut b = ModelBuilder::new();
    let names: Vec<_> = (0..counts.len()).map(|i| format!("row{i}")).collect();
    let labels: Vec<_> = names.iter().map(String::as_str).collect();
    let eta = b.real_vector("eta", &labels).unwrap();
    let eta = b.vector_value(&eta, shape(&[counts.len()])).unwrap();
    let rows = observations::<PoissonLog, _, _>(&mut b, counts, &[eta], |p| PoissonLog::new(p[0]))
        .unwrap();
    let sum = b.sum_value(rows).unwrap();
    let root = if zero {
        let zero = b.constant(0.).unwrap();
        b.multiply(sum, zero).unwrap()
    } else {
        sum
    };
    b.likelihood("counts", root).unwrap();
    b.compile().unwrap()
}

#[test]
fn poisson_typed_count_boundaries_and_source_gaps_are_not_erased() {
    use alea_core::{
        distribution::DistributionError,
        model::{
            ModelError,
            assembly::{CurvatureRoute, DerivativeOrder, SourceMode, SourceRequest},
        },
        target::EvaluationError,
    };
    let limit = 1_u64 << 53;
    let model = poisson_rows_model(vec![limit], false);
    for mode in [SourceMode::Density, SourceMode::Objective] {
        for order in [
            DerivativeOrder::Value,
            DerivativeOrder::First,
            DerivativeOrder::Hvp,
            DerivativeOrder::Third,
        ] {
            for curvature in [
                CurvatureRoute::Unavailable,
                CurvatureRoute::OriginalCompilerReverse,
            ] {
                assert!(matches!(
                    model.source_capabilities().resolve(SourceRequest {
                        mode,
                        order,
                        curvature
                    }),
                    Err(E::MissingPrimal { .. })
                ));
            }
        }
    }
    assert!(matches!(model.rust_primal(), Err(E::MissingPrimal { .. })));
    assert!(matches!(
        model.rust_objective_primal(),
        Err(E::MissingPrimal { .. })
    ));
    // Boundary admission/score only; no huge-count log-gamma accuracy claim.
    let target = model.into_target().unwrap();
    let mut g = [99.];
    assert!(target.logp_grad(&[0.], &mut g).unwrap().is_finite());
    assert_eq!(g[0], (limit - 1) as f64);
    for count in [limit + 1, u64::MAX] {
        for zero in [false, true] {
            let target = poisson_rows_model(vec![0, count], zero)
                .into_target()
                .unwrap();
            let mut g = [99.; 2];
            let result = target.logp_grad(&[0., 0.], &mut g);
            assert!(
                matches!(result,Err(ModelError::Evaluation(EvaluationError::Model(E::Factor(ref source)))) if matches!(source.downcast_ref::<DistributionError>(),Some(DistributionError::Numerical)))
            );
            assert_eq!(g, [99.; 2]);
        }
    }
}

#[test]
fn poisson_tails_zero_seed_overflow_staging_and_recovery_match_the_native_law() {
    for zero in [false, true] {
        let target = poisson_rows_model(vec![0, 1, 2], zero)
            .into_target()
            .unwrap();
        let mut g = [99.; 3];
        let valid = [-1000., 2_f64.ln(), 3_f64.ln()];
        for bad in [[0., 800., 0.], [0., f64::NAN, 0.], [0., f64::INFINITY, 0.]] {
            g.fill(99.);
            assert!(target.logp_grad(&bad, &mut g).is_err());
            assert_eq!(g, [99.; 3]);
            let value = target.logp_grad(&valid, &mut g).unwrap();
            close(value, if zero { 0. } else { 2. * 3_f64.ln() - 5. });
            for (actual, expected) in g
                .into_iter()
                .zip(if zero { [0.; 3] } else { [0., -1., -1.] })
            {
                close(actual, expected);
            }
        }
    }
    // Each row has finite primal/score, but their selected sum overflows. A zero
    // parent weight must not hide that forward failure.
    for zero in [false, true] {
        let target = poisson_rows_model(vec![0; 4], zero).into_target().unwrap();
        let mut g = [99.; 4];
        assert!(target.logp_grad(&[709.; 4], &mut g).is_err());
        assert_eq!(g, [99.; 4]);
        close(
            target.logp_grad(&[0.; 4], &mut g).unwrap(),
            if zero { 0. } else { -4. },
        );
        assert_eq!(g, if zero { [0.; 4] } else { [-1.; 4] });
    }
}

#[test]
fn poisson_block_offsets_agree_with_existing_regression_and_independent_small_counts() {
    use alea_distributions::modeling::regression::{RegressionData, poisson_log_regression};
    for n in [1, 5, 17, 129] {
        let (design, _, _) = data(n);
        let counts = poisson_counts(n);
        let offsets = poisson_offsets(n);
        let mut b = ModelBuilder::new();
        let beta = b.real_vector("beta", &["a", "b"]).unwrap();
        let beta = b.vector_value(&beta, shape(&[2, 1])).unwrap();
        let x = b.constant_block(shape(&[n, 2]), design.clone()).unwrap();
        let eta = b.matmul(x, beta).unwrap();
        let eta = b.reshape(eta, shape(&[n])).unwrap();
        let eta = add_offsets(&mut b, eta, offsets.clone());
        let rows = observations::<PoissonLog, _, _>(&mut b, counts.clone(), &[eta], |p| {
            PoissonLog::new(p[0])
        })
        .unwrap();
        let sum = b.sum_value(rows).unwrap();
        b.likelihood("poisson", sum).unwrap();
        let block = b.compile().unwrap().into_target().unwrap();
        let mut b = ModelBuilder::new();
        let beta = [b.real("a").unwrap().expr(), b.real("b").unwrap().expr()];
        let data = RegressionData::new(2, design.clone(), counts.clone())
            .unwrap()
            .with_offsets(offsets.clone())
            .unwrap();
        let term = poisson_log_regression(&mut b, &beta, data).unwrap();
        b.likelihood("poisson", term).unwrap();
        let native = b.compile().unwrap().into_target().unwrap();
        for q in [[0.3, -0.7], [-1.2, 0.5], [1., 1.]] {
            let mut expected = 0.;
            let mut grad = [0.; 2];
            for i in 0..n {
                let eta = design[2 * i] * q[0] + design[2 * i + 1] * q[1] + offsets[i];
                let rate = eta.exp();
                expected += counts[i] as f64 * eta - rate - log_factorial(counts[i]);
                for j in 0..2 {
                    grad[j] += (counts[i] as f64 - rate) * design[2 * i + j];
                }
            }
            let mut bg = [99.; 2];
            let mut ng = [99.; 2];
            close(block.logp_grad(&q, &mut bg).unwrap(), expected);
            close(native.logp_grad(&q, &mut ng).unwrap(), expected);
            for j in 0..2 {
                close(bg[j], grad[j]);
                close(ng[j], grad[j]);
            }
        }
    }
}

#[test]
fn poisson_invalid_nonlinear_predictor_rejects_before_law_invocation_and_recovers() {
    let (model, calls, maps) = build(5, 4);
    let target = model.into_target().unwrap();
    let mut g = [99.; 3];
    assert!(target.logp_grad(&[1e200, 0., 0.], &mut g).is_err());
    assert_eq!(g, [99.; 3]);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(maps.load(Ordering::Relaxed), 1);
    let q = [0.3, -0.7, -0.2];
    let (value, expected) = oracle(5, &q, 4);
    close(target.logp_grad(&q, &mut g).unwrap(), value);
    for (a, b) in g.into_iter().zip(expected) {
        close(a, b);
    }
    assert_eq!(calls.load(Ordering::Relaxed), 5);
    assert_eq!(maps.load(Ordering::Relaxed), 2);
}
