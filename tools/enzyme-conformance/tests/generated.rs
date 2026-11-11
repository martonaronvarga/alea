#[path = "../models.rs"]
mod models;
use alea::{
    capability::{PotentialHessianDerivative, PotentialHessianVector},
    target::LogDensityGradient,
};
use alea_enzyme_conformance as generated;

#[track_caller]
fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

fn record_nonfinite<T: std::fmt::Debug>(
    failures: &mut Vec<String>,
    operation: &str,
    x: f64,
    result: Result<T, alea::autodiff::autodiff::AutodiffError>,
) {
    if !matches!(
        result,
        Err(alea::autodiff::autodiff::AutodiffError::NonFinite)
    ) {
        failures.push(format!("{operation} invalid={x} result={result:?}"));
    }
}

#[test]
fn unrelated_trusted_or_checkpoint_factor_preserves_compiler_reverse() {
    use models::{RouteCompanion as B, route_isolation};
    for (companion, selected, explicit, checkpoint) in [
        (B::None, false, false, false),
        (B::Trusted, false, true, false),
        (B::Checkpoint, false, true, true),
        (B::Opaque, false, false, false),
        (B::TrustedAndOpaque, false, true, false),
        (B::Trusted, true, false, false),
        (B::Checkpoint, true, false, false),
        (B::Opaque, true, false, false),
        (B::MissingPrimal, true, false, false),
    ] {
        let model = route_isolation(companion, selected).unwrap();
        assert!(model.source_capabilities().supports_rust_primal());
        let (a_index, a) = model
            .graph()
            .enumerate()
            .find(|(_, n)| n.factors.iter().any(|f| f.name() == Some("A")))
            .unwrap();
        assert!(a.active);
        assert_eq!(a.inputs.len(), 2);
        assert_eq!(a.inputs[0], a.inputs[1]);
        let plan = model
            .source_first_order(alea::model::assembly::SourceMode::Density)
            .unwrap();
        assert_eq!(plan.is_some(), explicit);
        if let Some(first) = plan {
            let source = first.program.source;
            let actual_checkpoint = first.checkpointed;
            assert_eq!(actual_checkpoint, checkpoint);
            // The checked tensor helper admits local scores, but compiler
            // reverse supplies A's gradient in every composed route.
            assert!(source.contains(&format!("fn __tensor{a_index}(")));
            assert!(source.contains(&format!("__factor_reverse{a_index},")));
            assert!(source.contains(&format!("__factor_reverse{a_index}((&*args)")));
            if checkpoint {
                let b = model
                    .graph()
                    .position(|node| {
                        node.factors
                            .iter()
                            .any(|f| f.name() == Some("checkpoint B"))
                    })
                    .unwrap();
                assert!(source.contains(&format!("fn __tensor{b}(")));
                assert!(!source.contains(&format!("__factor_reverse{b},")));
            }
            if matches!(companion, B::TrustedAndOpaque) {
                let opaque = model
                    .graph()
                    .position(|n| n.factors.iter().any(|f| f.name() == Some("opaque B")))
                    .unwrap();
                // The same composition still compiler-differentiates the
                // opaque source leaf; it must not borrow its native callback.
                assert!(source.contains(&format!("__factor_reverse{opaque},")));
            }
        }
    }
    let missing = route_isolation(B::MissingPrimal, false).unwrap();
    assert!(!missing.source_capabilities().supports_rust_primal());
    assert!(matches!(
        missing.rust_primal(),
        Err(alea::model::assembly::AssemblyError::MissingPrimal { .. })
    ));
    assert!(matches!(
        missing.source_score_admission(alea::model::assembly::SourceMode::Density),
        Err(alea::model::assembly::AssemblyError::MissingPrimal { .. })
    ));
}

#[test]
fn compiler_tensor_admission_preserves_zero_seeds_inactive_scores_and_lazy_branches() {
    use alea::autodiff::autodiff::AutodiffError;
    use generated::tensor_score_admission as model;
    const {
        assert!(!model::SOURCE_EXPLICIT_FIRST_ORDER);
        assert!(model::SOURCE_TENSOR_SCORE_VALIDATION);
    }
    fn check(target: impl PotentialHessianDerivative<Error = AutodiffError>) {
        let native = models::tensor_score_admission()
            .unwrap()
            .into_target()
            .unwrap();
        for x in [0_f64, -0., -1.5, 0.25] {
            let mut g = [41.; 2];
            let q = [x, -1.];
            close(native.logp_grad(&q, &mut g).unwrap(), x * x, 1e-12);
            assert_eq!(g, [2. * x, 0.]);
            close(target.logp_grad(&q, &mut g).unwrap(), x * x, 1e-12);
            assert_eq!(g, [2. * x, 0.]);
            target.potential_hvp(&q, &[0.5, 1.], &mut g).unwrap();
            assert_eq!(g, [-1., 0.]);
            let mut h = [41.; 4];
            let mut d = [41.; 8];
            target
                .potential_hessian_and_derivatives(&q, &mut h, &mut d)
                .unwrap();
            assert_eq!(h, [-2., 0., 0., 0.]);
            assert_eq!(d, [0.; 8]);
        }
        let q = [0., 1.];
        let mut g = [41.; 2];
        assert!(native.logp_grad(&q, &mut g).is_err());
        assert_eq!(g, [41.; 2]);
        assert_eq!(model::log_density(&q), 0.);
        assert!(matches!(
            target.logp_grad(&q, &mut g),
            Err(AutodiffError::NonFinite)
        ));
        for v in [[0., 0.], [1., 0.]] {
            assert!(matches!(
                target.potential_hvp(&q, &v, &mut g),
                Err(AutodiffError::NonFinite)
            ));
        }
        assert!(matches!(
            target.potential_hessian_and_derivatives(&q, &mut [41.; 4], &mut [41.; 8]),
            Err(AutodiffError::NonFinite)
        ));
        close(target.logp_grad(&[1.5, 1.], &mut g).unwrap(), 1.5, 1e-12);
        assert_eq!(g, [1., 0.]);
        close(target.logp_grad(&[0., -1.], &mut g).unwrap(), 0., 1e-12);
        assert_eq!(g, [0., 0.]);
    }
    check(model::target());
    check(model::composition::objective());
}

fn check_isolated_route(
    label: &str,
    target: impl PotentialHessianDerivative<Error = alea::autodiff::autodiff::AutodiffError>,
    value_only: fn(&[f64; 1]) -> f64,
    companion: models::RouteCompanion,
    selected: bool,
    flags: (bool, bool, bool),
) {
    use models::{RouteCompanion as B, route_isolation};
    let trusted = !selected && matches!(companion, B::Trusted | B::TrustedAndOpaque);
    let opaque = !selected && matches!(companion, B::Opaque | B::TrustedAndOpaque);
    let checkpoint = !selected && matches!(companion, B::Checkpoint);
    assert_eq!(
        flags,
        (trusted || checkpoint, checkpoint, !trusted && !checkpoint)
    );
    // A finite value remains valid when its derivative is singular. The extra
    // score validator belongs to derivative entries only.
    assert_eq!(value_only(&[0.]), 0.);
    let native = route_isolation(companion, selected)
        .unwrap()
        .into_target()
        .unwrap();
    let answer = |x: f64| {
        // Independent deterministic oracle: A = sqrt(x*x), trusted B =
        // x*x, opaque B = x^3, checkpoint B = x + 65*x. No sampler here.
        let t = f64::from(trusted);
        let o = f64::from(opaque);
        let c = f64::from(checkpoint);
        (
            x.abs() + t * x * x + o * x * x * x + 66. * c * x,
            x.signum() + 2. * t * x + 3. * o * x * x + 66. * c,
            -2. * t - 6. * o * x,
            -6. * o,
        )
    };
    for x in [-1.5, -0.4, 0.25, 1.5] {
        let (value, gradient, hessian, third) = answer(x);
        let mut g = [41.];
        close(native.logp_grad(&[x], &mut g).unwrap(), value, 2e-12);
        close(g[0], gradient, 2e-12);
        close(target.logp_grad(&[x], &mut g).unwrap(), value, 2e-12);
        close(g[0], gradient, 2e-12);
        for step in [1e-4, 1e-5, 1e-6] {
            let fd = (answer(x + step).0 - answer(x - step).0) / (2. * step);
            close(g[0], fd, 2e-8);
        }
        for direction in [0., 0.3, -1.] {
            target.potential_hvp(&[x], &[direction], &mut g).unwrap();
            close(g[0], hessian * direction, 2e-11);
        }
        let mut h = [41.];
        let mut d = [41.];
        target
            .potential_hessian_and_derivatives(&[x], &mut h, &mut d)
            .unwrap();
        close(h[0], hessian, 2e-11);
        close(d[0], third, 2e-11);
    }
    if selected {
        let mut g = [41.];
        // Excluded B's invalid-score sentinel must not poison A.
        close(target.logp_grad(&[2.], &mut g).unwrap(), 2., 2e-12);
        close(g[0], 1., 2e-12);
        target.potential_hvp(&[2.], &[0.], &mut g).unwrap();
        close(g[0], 0., 2e-12);
    }
    // Collect admission failures so an incorrect successful gradient cannot
    // hide curvature admission or recovery at the same point. These remain
    // ordinary failing assertions if any derivative admission regresses.
    let mut failures = Vec::new();
    for x in [f64::NAN, f64::INFINITY]
        .into_iter()
        .chain(trusted.then_some(2.))
        .chain([0.])
    {
        let mut g = [41.];
        assert!(native.logp_grad(&[x], &mut g).is_err());
        assert_eq!(g, [41.]); // Native target stages caller outputs.
        record_nonfinite(&mut failures, "gradient", x, target.logp_grad(&[x], &mut g));
        for direction in [0., 1.] {
            record_nonfinite(
                &mut failures,
                &format!("HVP direction={direction}"),
                x,
                target.potential_hvp(&[x], &[direction], &mut g),
            );
        }
        record_nonfinite(
            &mut failures,
            "Hessian/third",
            x,
            target.potential_hessian_and_derivatives(&[x], &mut g, &mut [41.]),
        );
        // Raw generated outputs are disposable after error, but successful
        // recovery must overwrite them with the complete independent answer.
        let (value, score, hessian, third) = answer(1.5);
        close(target.logp_grad(&[1.5], &mut g).unwrap(), value, 2e-12);
        close(g[0], score, 2e-12);
        close(native.logp_grad(&[1.5], &mut g).unwrap(), value, 2e-12);
        close(g[0], score, 2e-12);
        target.potential_hvp(&[1.5], &[1.], &mut g).unwrap();
        close(g[0], hessian, 2e-11);
        let mut d = [41.];
        target
            .potential_hessian_and_derivatives(&[1.5], &mut g, &mut d)
            .unwrap();
        close(g[0], hessian, 2e-11);
        close(d[0], third, 2e-11);
    }
    assert!(failures.is_empty(), "{label}:\n{}", failures.join("\n"));
}
macro_rules! isolated_route_case {
    ($module:ident, $b:ident, $selected:expr) => {
        #[test]
        fn $module() {
            check_isolated_route(
                stringify!($module),
                generated::$module::target(),
                generated::$module::log_density,
                models::RouteCompanion::$b,
                $selected,
                (
                    generated::$module::SOURCE_EXPLICIT_FIRST_ORDER,
                    generated::$module::SOURCE_CHECKPOINTED,
                    generated::$module::SOURCE_TENSOR_SCORE_VALIDATION,
                ),
            );
            check_isolated_route(
                concat!(stringify!($module), " objective"),
                generated::$module::composition::objective(),
                generated::$module::composition::value,
                models::RouteCompanion::$b,
                $selected,
                (
                    generated::$module::composition::SOURCE_EXPLICIT_FIRST_ORDER,
                    generated::$module::composition::SOURCE_CHECKPOINTED,
                    generated::$module::composition::SOURCE_TENSOR_SCORE_VALIDATION,
                ),
            );
        }
    };
}
isolated_route_case!(route_alone, None, false);
isolated_route_case!(route_trusted, Trusted, false);
isolated_route_case!(route_checkpoint, Checkpoint, false);
isolated_route_case!(route_opaque, Opaque, false);
isolated_route_case!(route_trusted_opaque, TrustedAndOpaque, false);
isolated_route_case!(route_selected_trusted, Trusted, true);
isolated_route_case!(route_selected_checkpoint, Checkpoint, true);
isolated_route_case!(route_selected_opaque, Opaque, true);
isolated_route_case!(route_selected_missing, MissingPrimal, true);

#[test]
fn nonzero_third_derivatives_match_independent_polynomial() {
    let t = generated::polynomial::target();
    for [x, y] in [[0.3, -0.7], [0.0, 0.0], [-1.2, 0.5]] {
        let q = [x, y];
        let mut g = [41.0; 2];
        let lp = t.logp_grad(&q, &mut g).unwrap();
        close(
            lp,
            -0.5 * (x * x + y * y) - 0.1 * x * x * y * y - 0.01 * x * x * x * x,
            1e-13,
        );
        close(g[0], -x - 0.2 * x * y * y - 0.04 * x * x * x, 1e-13);
        close(g[1], -y - 0.2 * x * x * y, 1e-13);
        let mut h = [41.0; 4];
        let mut d = [41.0; 8];
        t.potential_hessian_and_derivatives(&q, &mut h, &mut d)
            .unwrap();
        let expected_h = [
            1.0 + 0.2 * y * y + 0.12 * x * x,
            0.4 * x * y,
            0.4 * x * y,
            1.0 + 0.2 * x * x,
        ];
        let expected_d = [
            0.24 * x,
            0.4 * y,
            0.4 * y,
            0.4 * x,
            0.4 * y,
            0.4 * x,
            0.4 * x,
            0.0,
        ];
        for (actual, expected) in h.into_iter().zip(expected_h) {
            close(actual, expected, 1e-13);
        }
        for (actual, expected) in d.into_iter().zip(expected_d) {
            close(actual, expected, 1e-13);
        }
        let v = [0.4, -0.2];
        let mut hv = [41.0; 2];
        t.potential_hvp(&q, &v, &mut hv).unwrap();
        for i in 0..2 {
            close(
                hv[i],
                expected_h[2 * i] * v[0] + expected_h[2 * i + 1] * v[1],
                1e-13,
            );
        }
    }
}

fn check_model<T: PotentialHessianDerivative>(name: &str, target: T) {
    let model = models::all()
        .unwrap()
        .into_iter()
        .find(|(n, _)| *n == name)
        .unwrap()
        .1
        .into_target()
        .unwrap();
    let n = target.dimension();
    for scale in [-0.7, 0.0, 0.4] {
        let q: Vec<_> = (0..n).map(|i| scale + i as f64 * 0.13).collect();
        let mut g = vec![0.0; n];
        let mut expected = vec![0.0; n];
        close(
            target.logp_grad(&q, &mut g).unwrap(),
            model.logp_grad(&q, &mut expected).unwrap(),
            2e-12,
        );
        for (&a, &b) in g.iter().zip(&expected) {
            close(a, b, 2e-11);
        }
        let mut h = vec![0.0; n * n];
        let mut third = vec![0.0; n * n * n];
        target
            .potential_hessian_and_derivatives(&q, &mut h, &mut third)
            .unwrap();
        let mut plus = q.clone();
        let mut minus = q.clone();
        let mut gp = g.clone();
        let mut gm = g.clone();
        let mut hp = h.clone();
        let mut hm = h.clone();
        let mut scratch = third.clone();
        let mut v = vec![0.0; n];
        let mut product = g.clone();
        for k in 0..n {
            plus[k] += 1e-5;
            minus[k] -= 1e-5;
            model.logp_grad(&plus, &mut gp).unwrap();
            model.logp_grad(&minus, &mut gm).unwrap();
            v[k] = 1.0;
            target.potential_hvp(&q, &v, &mut product).unwrap();
            for i in 0..n {
                close(h[i * n + k], -(gp[i] - gm[i]) / 2e-5, 2e-7);
                close(product[i], h[i * n + k], 2e-12);
            }
            target
                .potential_hessian_and_derivatives(&plus, &mut hp, &mut scratch)
                .unwrap();
            target
                .potential_hessian_and_derivatives(&minus, &mut hm, &mut scratch)
                .unwrap();
            for i in 0..n * n {
                close(third[k * n * n + i], (hp[i] - hm[i]) / 2e-5, 2e-6);
            }
            plus[k] = q[k];
            minus[k] = q[k];
            v[k] = 0.0;
        }
    }
}

#[test]
fn checkpoint_composition_higher_derivatives_match_native_score_differences() {
    // Qualify the shipped composition: higher orders deliberately retain the
    // compiler reverse, NOT forward AD of the explicit checkpoint pullback.
    check_model("tensor_checkpoint", generated::tensor_checkpoint::target());
    check_model(
        "tensor_checkpoint_long",
        generated::tensor_checkpoint_long::target(),
    );
    check_model(
        "tensor_checkpoint_mixed",
        generated::tensor_checkpoint_mixed::target(),
    );
    check_model(
        "tensor_checkpoint_matrix",
        generated::tensor_checkpoint_matrix::target(),
    );
    check_model(
        "tensor_checkpoint_vector",
        generated::tensor_checkpoint_vector::target(),
    );
    check_model(
        "tensor_checkpoint_routes",
        generated::tensor_checkpoint_routes::target(),
    );
    check_model(
        "tensor_checkpoint_constant",
        generated::tensor_checkpoint_constant::target(),
    );
    check_model(
        "tensor_checkpoint_immutable",
        generated::tensor_checkpoint_immutable::target(),
    );
}

#[test]
fn typed_models_and_every_transform_include_one_jacobian_through_third_order() {
    macro_rules! check { ($($name:ident),* $(,)?) => {$(check_model(stringify!($name), generated::$name::target());)*}; }
    check!(
        native_normal_batch,
        native_logit_batch,
        native_poisson_batch,
        native_normal_large,
        native_logit_large,
        native_poisson_large,
        normal_vector,
        normal_vector_observed,
        regression,
        logistic,
        interval,
        lower,
        upper,
        ordered,
        positive_ordered,
        simplex,
        unit_vector,
        cholesky_covariance,
        cholesky_correlation,
        covariance,
        correlation,
        covariance3,
        correlation3
    );
}

#[test]
fn native_batch_source_matches_tail_and_scale_admission_without_clipping() {
    fn compare<T: LogDensityGradient>(family: usize, target: T, points: &[Vec<f64>]) {
        let native = models::native_batch(family).unwrap().into_target().unwrap();
        for q in points {
            let mut actual = vec![91.; q.len()];
            let mut expected = vec![92.; q.len()];
            let a = target.logp_grad(q, &mut actual);
            let b = native.logp_grad(q, &mut expected);
            assert_eq!(a.is_ok(), b.is_ok(), "family {family}, q={q:?}");
            if let (Ok(a), Ok(b)) = (a, b) {
                close(a, b, 2e-11);
                for (a, b) in actual.into_iter().zip(expected) {
                    close(a, b, 2e-10);
                }
            }
        }
    }
    compare(
        0,
        generated::native_normal_batch::target(),
        &[
            vec![0.3, -0.4, -100.],
            vec![0., 0., -1000.],
            vec![0., 0., 1000.],
        ],
    );
    compare(
        1,
        generated::native_logit_batch::target(),
        &[vec![1000., 0.], vec![-1000., 0.], vec![0., 1000.]],
    );
    compare(
        2,
        generated::native_poisson_batch::target(),
        &[vec![-1000., 0.], vec![0., 0.], vec![1000., 0.]],
    );
}

#[test]
fn native_first_order_contexts_are_reusable_and_transactional_on_independent_threads() {
    fn check<T: LogDensityGradient>(target: T, q: &[f64]) {
        let mut g = vec![0.; q.len()];
        let expected = target.logp_grad(q, &mut g).unwrap(); // admit this target's first evaluation
        let expected_g = g.clone();
        let allocations = allocation_counter::measure(|| {
            for _ in 0..16 {
                assert_eq!(
                    target.logp_grad(q, &mut g).unwrap().to_bits(),
                    expected.to_bits()
                );
                assert_eq!(g, expected_g);
            }
        });
        assert_eq!(allocations.count_total, 0);
        let mut bad = q.to_vec();
        bad[0] = f64::MAX;
        // Low-level derivative outputs are scratch on failure. Transactional
        // publication belongs to PointState, not the raw Enzyme adapter.
        let mut point = alea::target::PointState::new(
            &target,
            alea::math::buffer::OwnedBuffer::from_fn(q.len(), |i| q[i]),
        )
        .unwrap();
        if point
            .try_update(&bad, &mut alea::target::EvaluationWorkspace::new(q.len()))
            .is_err()
        {
            assert_eq!(point.position(), q);
            assert_eq!(point.gradient(), expected_g);
        }
        assert_eq!(
            target.logp_grad(q, &mut g).unwrap().to_bits(),
            expected.to_bits()
        );
        assert_eq!(g, expected_g);
    }
    const {
        assert!(generated::native_logit_batch::SOURCE_EXPLICIT_FIRST_ORDER);
        assert!(!generated::native_logit_batch::SOURCE_CHECKPOINTED);
    }
    // Each target initializes independently, then reuses its frozen descriptor and
    // private scratch, including when constructed on independent threads.
    let handles: Vec<_> = (0..4)
        .map(|_| {
            std::thread::spawn(|| {
                check(generated::native_normal_batch::target(), &[0.3, -0.4, 0.]);
                check(generated::native_logit_batch::target(), &[0.3, -0.4]);
                check(generated::native_poisson_batch::target(), &[0.3, -0.4]);
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn tensor_regions_match_native_scores_and_finite_difference_curvature() {
    check_model("tensor_matrix", generated::tensor_matrix::target());
    check_model("tensor_broadcast", generated::tensor_broadcast::target());
    check_model(
        "tensor_checkpoint_broadcast",
        generated::tensor_checkpoint_broadcast::target(),
    );
    check_model("tensor_control", generated::tensor_control::target());
}

#[test]
fn broadcast_aliases_have_independent_quartic_derivatives_on_both_source_routes() {
    const {
        assert!(!generated::tensor_broadcast::SOURCE_CHECKPOINTED);
        assert!(generated::tensor_checkpoint_broadcast::SOURCE_CHECKPOINTED);
    }
    fn check(target: impl PotentialHessianDerivative) {
        for [a, b] in [[0., 0.], [0.25, -0.5], [-1., 2.]] {
            let norm = a * a + b * b;
            let mut g = [41.; 2];
            close(
                target.logp_grad(&[a, b], &mut g).unwrap(),
                -0.5 * norm * norm,
                1e-12,
            );
            close(g[0], -2. * a * norm, 1e-12);
            close(g[1], -2. * b * norm, 1e-12);
            let mut h = [41.; 4];
            let mut third = [41.; 8];
            target
                .potential_hessian_and_derivatives(&[a, b], &mut h, &mut third)
                .unwrap();
            for (actual, expected) in h.into_iter().zip([
                2. * norm + 4. * a * a,
                4. * a * b,
                4. * a * b,
                2. * norm + 4. * b * b,
            ]) {
                close(actual, expected, 1e-12);
            }
            for (actual, expected) in third.into_iter().zip([
                12. * a,
                4. * b,
                4. * b,
                4. * a,
                4. * b,
                4. * a,
                4. * a,
                12. * b,
            ]) {
                close(actual, expected, 1e-12);
            }
        }
    }
    check(generated::tensor_broadcast::target());
    check(generated::tensor_checkpoint_broadcast::target());
}

#[test]
fn repeated_calls_overwrite_outputs_allocate_nothing_and_reject_bad_shapes() {
    let t = generated::polynomial::target();
    let mut g = [1.0; 2];
    let mut h = [1.0; 4];
    let mut d = [1.0; 8];
    let allocations = allocation_counter::measure(|| {
        for _ in 0..32 {
            t.logp_grad(&[0.3, -0.7], &mut g).unwrap();
            t.potential_hvp(&[0.3, -0.7], &[1.0, 0.0], &mut g).unwrap();
            t.potential_hessian_and_derivatives(&[0.3, -0.7], &mut h, &mut d)
                .unwrap();
        }
    });
    assert_eq!(allocations.count_total, 0);
    let saved = g;
    assert!(t.logp_grad(&[0.0], &mut g).is_err());
    assert_eq!(g, saved);
    assert!(t.potential_hvp(&[0.0; 2], &[0.0], &mut g).is_err());
    assert_eq!(g, saved);
    assert!(
        t.potential_hessian_and_derivatives(&[0.0; 2], &mut h[..3], &mut d)
            .is_err()
    );
    assert!(t.logp_grad(&[f64::NAN, 0.0], &mut g).is_err());
    let transformed = generated::regression::target();
    assert!(transformed.logp_grad(&[0.0, -1000.0], &mut g).is_err());
    assert!(
        transformed
            .potential_hvp(&[0.0, -1000.0], &[1.0, 0.0], &mut g)
            .is_err()
    );
}

#[test]
fn larger_generated_target_has_correct_hessian_and_zero_third_tensor() {
    let target = generated::gaussian16::target();
    let q = [0.3; 16];
    let mut g = [41.0; 16];
    close(
        target.logp_grad(&q, &mut g).unwrap(),
        -0.5 * 16.0 * 0.3 * 0.3,
        1e-13,
    );
    for v in g {
        close(v, -0.3, 1e-13);
    }
    let mut h = [41.0; 256];
    let mut third = [41.0; 4096];
    target
        .potential_hessian_and_derivatives(&q, &mut h, &mut third)
        .unwrap();
    for i in 0..16 {
        for j in 0..16 {
            close(h[i * 16 + j], f64::from(i == j), 1e-13);
        }
    }
    assert!(third.iter().all(|&v| v == 0.0));
}

#[test]
fn rounded_singular_covariances_and_collapsed_intervals_fail_all_derivative_paths() {
    fn reject<T: PotentialHessianDerivative>(target: T, q: &[f64]) {
        let n = q.len();
        let mut g = vec![41.0; n];
        let mut h = vec![41.0; n * n];
        let mut third = vec![41.0; n * n * n];
        assert!(target.logp_grad(q, &mut g).is_err());
        assert!(target.potential_hvp(q, &vec![1.0; n], &mut g).is_err());
        assert!(
            target
                .potential_hessian_and_derivatives(q, &mut h, &mut third)
                .is_err()
        );
        target.logp_grad(&vec![0.0; n], &mut g).unwrap();
    }
    reject(generated::covariance::target(), &[0.0, 1e150, 0.0]);
    reject(generated::interval::target(), &[1000.0]);
    reject(generated::correlation::target(), &[1000.0]);
}

fn objective_check<T: PotentialHessianDerivative>(
    model: alea::model::assembly::NativeModel,
    generated: T,
) {
    let d = model.layout().unconstrained_dimension();
    let objective = model.into_objective().unwrap();
    for scale in [-0.4, 0., 0.5] {
        let q: Vec<_> = (0..d).map(|i| scale + 0.03 * i as f64).collect();
        let mut g = vec![0.; d];
        let mut reference = g.clone();
        close(
            generated.logp_grad(&q, &mut g).unwrap(),
            objective.logp_grad(&q, &mut reference).unwrap(),
            1e-11,
        );
        for (a, b) in g.iter().zip(&reference) {
            close(*a, *b, 1e-10);
        }
        let mut h = vec![0.; d * d];
        let mut third = vec![0.; d * d * d];
        generated
            .potential_hessian_and_derivatives(&q, &mut h, &mut third)
            .unwrap();
        for k in 0..d {
            let step = 1e-5;
            let mut plus = q.clone();
            let mut minus = q.clone();
            plus[k] += step;
            minus[k] -= step;
            let mut gp = vec![0.; d];
            let mut gm = gp.clone();
            objective.logp_grad(&plus, &mut gp).unwrap();
            objective.logp_grad(&minus, &mut gm).unwrap();
            for i in 0..d {
                close(h[i * d + k], -(gp[i] - gm[i]) / (2. * step), 2e-6);
            }
        }
        assert!(third.iter().all(|x| x.is_finite()));
    }
}

#[test]
fn objective_generated_scores_and_curvature_match_every_chart() {
    macro_rules! check { ($($name:ident),* $(,)?) => {$(
        let model=models::all().unwrap().into_iter().find(|(name,_)|*name==stringify!($name)).unwrap().1;
        objective_check(model,generated::$name::composition::objective());
    )*}; }
    check!(
        polynomial,
        regression,
        logistic,
        interval,
        lower,
        upper,
        ordered,
        positive_ordered,
        simplex,
        unit_vector,
        cholesky_covariance,
        cholesky_correlation,
        covariance,
        correlation,
        covariance3,
        correlation3,
        gaussian16,
        selected_likelihood,
        selected_posterior,
        tensor_checkpoint_mixed,
        tensor_checkpoint_matrix,
        tensor_checkpoint_vector,
        tensor_checkpoint_constant
    );
}

#[test]
fn objective_higher_derivatives_include_chart_curvature_but_no_density_terms() {
    let t = generated::lower::composition::objective();
    for q in [-0.7_f64, 0., 0.4] {
        let e = q.exp();
        let x = -2. + e;
        let mut g = [0.];
        let mut h = [0.];
        let mut third = [0.];
        close(t.logp_grad(&[q], &mut g).unwrap(), -0.5 * x * x, 1e-13);
        close(g[0], -x * e, 1e-13);
        t.potential_hessian_and_derivatives(&[q], &mut h, &mut third)
            .unwrap();
        close(h[0], e * e + x * e, 1e-13);
        close(third[0], 3. * e * e + x * e, 1e-13);
    }
}

#[test]
fn independently_requested_polynomial_orders_have_independent_answers() {
    use generated::polynomial::{first, hvp, value};
    for [x, y] in [[0.3, -0.7], [-1.2, 0.5], [0., 0.]] {
        let q = [x, y];
        let expected = -0.5 * (x * x + y * y) - 0.1 * x * x * y * y - 0.01 * x * x * x * x;
        close(value::log_density(&q), expected, 1e-13);
        close(value::composition::value(&q), expected, 1e-13);
        {
            let target = first::target();
            let mut g = [f64::NAN; 2];
            close(target.logp_grad(&q, &mut g).unwrap(), expected, 1e-13);
            close(g[0], -x - 0.2 * x * y * y - 0.04 * x * x * x, 1e-13);
            close(g[1], -y - 0.2 * x * x * y, 1e-13);
            assert!(matches!(
                target.logp_grad(&[], &mut g),
                Err(alea::autodiff::autodiff::AutodiffError::Dimension)
            ));
            assert!(target.logp_grad(&[f64::NAN, y], &mut g).is_err());
            close(target.logp_grad(&q, &mut g).unwrap(), expected, 1e-13);
        }
        let mut g = [41.; 2];
        close(
            first::composition::objective()
                .logp_grad(&q, &mut g)
                .unwrap(),
            expected,
            1e-13,
        );
        let direction = [0.4, -0.2];
        let mut out = [f64::NAN; 2];
        hvp::target()
            .potential_hvp(&q, &direction, &mut out)
            .unwrap();
        close(
            out[0],
            (1. + 0.2 * y * y + 0.12 * x * x) * direction[0] + 0.4 * x * y * direction[1],
            1e-13,
        );
        close(
            out[1],
            0.4 * x * y * direction[0] + (1. + 0.2 * x * x) * direction[1],
            1e-13,
        );
        let mut objective_out = [f64::NAN; 2];
        hvp::composition::objective()
            .potential_hvp(&q, &direction, &mut objective_out)
            .unwrap();
        for (a, b) in objective_out.into_iter().zip(out) {
            close(a, b, 1e-13);
        }
    }
}

#[test]
fn first_only_routes_preserve_singular_admission_and_negative_branch_recovery() {
    fn check<T: LogDensityGradient>(target: T, linear: f64, quadratic: f64, cubic: f64)
    where
        T::Error: std::fmt::Debug,
    {
        let mut g = [41.];
        assert!(target.logp_grad(&[0.], &mut g).is_err());
        for x in [-0.7_f64, 0.4] {
            let expected = x.abs() + linear * x + quadratic * x * x + cubic * x * x * x;
            close(target.logp_grad(&[x], &mut g).unwrap(), expected, 1e-12);
            close(
                g[0],
                x.signum() + linear + 2. * quadratic * x + 3. * cubic * x * x,
                1e-12,
            );
        }
    }
    check(generated::route_alone::first::target(), 0., 0., 0.);
    check(generated::route_trusted::first::target(), 0., 1., 0.);
    check(generated::route_checkpoint::first::target(), 66., 0., 0.);
    check(generated::route_opaque::first::target(), 0., 0., 1.);
    check(generated::route_trusted_opaque::first::target(), 0., 1., 1.);
    check(
        generated::route_checkpoint::first::composition::objective(),
        66.,
        0.,
        0.,
    );
    fn check_hvp<T: PotentialHessianVector>(target: T, quadratic: f64, cubic: f64)
    where
        T::Error: std::fmt::Debug,
    {
        let mut out = [41.];
        // Zero direction must still admit the actual first-order result.
        assert!(target.potential_hvp(&[0.], &[0.], &mut out).is_err());
        for x in [-0.7, 0.4] {
            target.potential_hvp(&[x], &[0.3], &mut out).unwrap();
            close(out[0], -(2. * quadratic + 6. * cubic * x) * 0.3, 1e-12);
        }
    }
    check_hvp(generated::route_trusted::hvp::target(), 1., 0.);
    check_hvp(generated::route_checkpoint::hvp::target(), 0., 0.);
    check_hvp(generated::route_opaque::hvp::target(), 0., 1.);
    check_hvp(
        generated::route_trusted_opaque::hvp::composition::objective(),
        1.,
        1.,
    );
    // Value-only keeps primal admission; it must not run the singular pullback.
    close(generated::route_alone::value::log_density(&[0.]), 0., 0.);
    close(
        generated::route_checkpoint::value::composition::value(&[0.]),
        0.,
        0.,
    );
}

#[test]
fn requested_chart_orders_keep_density_and_objective_jacobians_separate() {
    for q in [-0.6_f64, 0.3] {
        let e = q.exp();
        let x = -2. + e;
        let objective = -0.5 * x * x;
        close(
            generated::lower::value::log_density(&[q]),
            objective + q,
            1e-13,
        );
        close(
            generated::lower::value::composition::value(&[q]),
            objective,
            1e-13,
        );
        let mut g = [41.];
        close(
            generated::lower::first::target()
                .logp_grad(&[q], &mut g)
                .unwrap(),
            objective + q,
            1e-13,
        );
        close(g[0], -x * e + 1., 1e-13);
        close(
            generated::lower::first::composition::objective()
                .logp_grad(&[q], &mut g)
                .unwrap(),
            objective,
            1e-13,
        );
        close(g[0], -x * e, 1e-13);
        let mut out = [41.];
        generated::lower::hvp::target()
            .potential_hvp(&[q], &[0.7], &mut out)
            .unwrap();
        close(out[0], (e * e + x * e) * 0.7, 1e-13);
        generated::lower::hvp::composition::objective()
            .potential_hvp(&[q], &[0.7], &mut out)
            .unwrap();
        close(out[0], (e * e + x * e) * 0.7, 1e-13);
    }
    let target = generated::route_trusted::first::target();
    let mut g = [41.];
    assert!(matches!(
        target.logp_grad(&[2.], &mut g),
        Err(alea::autodiff::autodiff::AutodiffError::NonFinite)
    ));
    close(generated::route_trusted::value::log_density(&[2.]), 6., 0.);
    close(target.logp_grad(&[-0.5], &mut g).unwrap(), 0.75, 1e-13);
    close(g[0], -2., 1e-13);
}

#[test]
fn compiled_marker_leaves_remain_verbatim_in_compiler_and_hybrid_modules() {
    let receipt = "__WORKSPACE__ __PUBLIC_TARGET__ __TENSOR_SCORE_VALIDATION__ __VALIDATION__";
    for source in [
        include_str!(concat!(env!("OUT_DIR"), "/route_opaque.rs")),
        include_str!(concat!(env!("OUT_DIR"), "/route_opaque_first.rs")),
        include_str!(concat!(env!("OUT_DIR"), "/route_trusted_opaque.rs")),
        include_str!(concat!(
            env!("OUT_DIR"),
            "/route_trusted_opaque_first_objective.rs"
        )),
    ] {
        assert!(source.contains(receipt));
    }
    // Existing isolated-route tests execute these exact modules against the
    // independent cubic answers, including both signs and recovery after error.
}

#[test]
fn generated_native_contexts_are_owned_by_each_target_and_recover_independently() {
    fn check<T: LogDensityGradient>(factory: impl Fn() -> T, q: &[f64])
    where
        T::Error: std::fmt::Debug,
    {
        let (mut left, mut right) = (None, None);
        let a = allocation_counter::measure(|| left = Some(factory()));
        let b = allocation_counter::measure(|| right = Some(factory()));
        // Each instance constructs its native context; a warmed TLS cache would
        // allocate nothing for either target constructor and fail this check.
        assert!(a.count_total > 0 && b.count_total > 0);
        let (left, right) = (left.unwrap(), right.unwrap());
        let (mut g, mut h) = (vec![19.; q.len()], vec![23.; q.len()]);
        let value = left.logp_grad(q, &mut g).unwrap();
        assert_eq!(
            right.logp_grad(q, &mut h).unwrap().to_bits(),
            value.to_bits()
        );
        assert_eq!(g, h);
        let mut bad = q.to_vec();
        // Finite extreme coefficients can be admissible in stable GLM tails.
        bad[0] = f64::NAN;
        assert!(left.logp_grad(&bad, &mut g).is_err());
        assert_eq!(
            right.logp_grad(q, &mut h).unwrap().to_bits(),
            value.to_bits()
        );
        let allocations = allocation_counter::measure(|| {
            for _ in 0..16 {
                assert_eq!(
                    left.logp_grad(q, &mut g).unwrap().to_bits(),
                    value.to_bits()
                );
                assert_eq!(
                    right.logp_grad(q, &mut h).unwrap().to_bits(),
                    value.to_bits()
                );
                assert_eq!(g, h);
            }
        });
        assert_eq!(allocations.count_total, 0);
    }
    check(
        generated::native_normal_batch::first::target,
        &[0.3, -0.4, 0.],
    );
    check(generated::native_logit_batch::first::target, &[0.3, -0.4]);
    check(
        generated::native_poisson_batch::first::composition::objective,
        &[0.3, -0.4],
    );
}

#[test]
fn tail_and_empty_rank_three_maps_match_dense_polynomials_on_both_source_routes() {
    fn check<T: PotentialHessianDerivative<Error = alea::autodiff::autodiff::AutodiffError>>(
        target: T,
        value: fn(&[f64; 2]) -> f64,
        empty: bool,
    ) {
        for [a, b] in [[-0.75, 0.5], [0.25, -0.5], [0., 0.], [-1., 2.]] {
            // Explicit dense coordinates for [2,1,3] x [1,3,1], no map helper.
            let mut expected = 0.;
            if !empty {
                let left = [a, b, a, b, a, b];
                let right = [a, b, a];
                for i in 0..2 {
                    for &r in &right {
                        for k in 0..3 {
                            let product = left[i * 3 + k] * r;
                            expected -= 0.25 * product * product;
                        }
                    }
                }
            }
            let scale = if empty { 0. } else { 1. };
            close(value(&[a, b]), expected, 1e-12);
            let mut g = [41.; 2];
            close(target.logp_grad(&[a, b], &mut g).unwrap(), expected, 1e-12);
            close(g[0], scale * (-6. * a * a * a - 4.5 * a * b * b), 1e-12);
            close(g[1], scale * (-4.5 * a * a * b - 3. * b * b * b), 1e-12);
            let h = [
                18. * a * a + 4.5 * b * b,
                9. * a * b,
                9. * a * b,
                4.5 * a * a + 9. * b * b,
            ]
            .map(|v| scale * v);
            let d = [
                36. * a,
                9. * b,
                9. * b,
                9. * a,
                9. * b,
                9. * a,
                9. * a,
                18. * b,
            ]
            .map(|v| scale * v);
            let (mut matrix, mut third) = ([41.; 4], [41.; 8]);
            target
                .potential_hessian_and_derivatives(&[a, b], &mut matrix, &mut third)
                .unwrap();
            for (x, y) in matrix.into_iter().zip(h) {
                close(x, y, 1e-12);
            }
            for (x, y) in third.into_iter().zip(d) {
                close(x, y, 1e-12);
            }
            for direction in [[0., 0.], [0.3, -0.7]] {
                target.potential_hvp(&[a, b], &direction, &mut g).unwrap();
                for i in 0..2 {
                    close(
                        g[i],
                        h[2 * i] * direction[0] + h[2 * i + 1] * direction[1],
                        1e-12,
                    );
                }
            }
        }
        let mut g = [41.; 2];
        assert!(matches!(
            target.logp_grad(&[f64::NAN, 0.], &mut g),
            Err(alea::autodiff::autodiff::AutodiffError::NonFinite)
        ));
        assert!(matches!(
            target.potential_hvp(&[f64::NAN, 0.], &[0., 0.], &mut g),
            Err(alea::autodiff::autodiff::AutodiffError::NonFinite)
        ));
        assert!(target.logp_grad(&[-0.75, 0.5], &mut g).is_ok());
    }
    macro_rules! case {
        ($name:ident,$empty:expr) => {
            check(
                generated::$name::target(),
                generated::$name::value::log_density,
                $empty,
            );
            check(
                generated::$name::composition::objective(),
                generated::$name::value::composition::value,
                $empty,
            );
            // Actual first-only emission also runs the same map score route.
            let first = generated::$name::first::target();
            let mut g = [41.; 2];
            let full = generated::$name::target();
            let mut expected = [41.; 2];
            close(
                first.logp_grad(&[-0.75, 0.5], &mut g).unwrap(),
                full.logp_grad(&[-0.75, 0.5], &mut expected).unwrap(),
                1e-12,
            );
            for (a, b) in g.into_iter().zip(expected) {
                close(a, b, 1e-12);
            }
        };
    }
    case!(tensor_broadcast_tail, false);
    case!(tensor_checkpoint_broadcast_tail, false);
    case!(tensor_broadcast_empty, true);
    case!(tensor_checkpoint_broadcast_empty, true);
}
