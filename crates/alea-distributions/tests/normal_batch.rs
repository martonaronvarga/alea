use alea_core::{
    distribution::{Distribution, ParameterScore, ScalarEventScore},
    model::{
        ConstrainedModel,
        assembly::{AssemblyError, ModelBuilder, Primitive},
    },
    target::LogDensityGradient,
};
use alea_distributions::{
    modeling::{NormalBatch, NormalFactor, normal, normal_batch},
    scalar::Normal,
};

fn close(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() <= tolerance * (1. + b.abs()), "{a} != {b}");
}

#[test]
fn broadcasting_aliases_packet_tails_and_every_gradient_match_scalar_laws_and_differences() {
    for rows in [1, 3, 4, 5, 17] {
        for mask in 0..8 {
            // With three scalars there is no declared observation axis: N=1.
            if mask == 0 && rows != 1 {
                continue;
            }
            let mut b = ModelBuilder::new();
            let mut handles = [Vec::new(), Vec::new(), Vec::new()];
            let mut q = Vec::new();
            for (j, h) in handles.iter_mut().enumerate() {
                for i in 0..if mask & (1 << j) == 0 { 1 } else { rows } {
                    h.push(b.real(&format!("{j}-{i}")).unwrap().expr());
                    q.push(if j == 2 {
                        0.7 + i as f64 / 11.
                    } else {
                        (i as f64 - 2.) / 7. + j as f64 / 3.
                    });
                }
            }
            let f = normal_batch(&mut b, &handles[0], &handles[1], &handles[2]).unwrap();
            b.likelihood("data", f).unwrap();
            let model = b.compile().unwrap();
            let mut g = vec![0.; q.len()];
            let value = model.logp_grad(&q, &mut g).unwrap();
            let primitive = model
                .graph()
                .find(|n| n.expression == f)
                .unwrap()
                .primitive
                .unwrap()
                .descriptor::<NormalBatch>()
                .unwrap();
            assert_eq!(primitive.rows(), rows);
            let lens = handles.each_ref().map(Vec::len);
            let mut expected = 0.;
            let mut expected_g = vec![0.; g.len()];
            for i in 0..rows {
                let at = [
                    if lens[0] == 1 { 0 } else { i },
                    lens[0] + if lens[1] == 1 { 0 } else { i },
                    lens[0] + lens[1] + if lens[2] == 1 { 0 } else { i },
                ];
                let law = Normal::new(q[at[1]], q[at[2]]).unwrap();
                expected += law.log_prob(&q[at[0]]).unwrap();
                let mut p = [0.; 2];
                law.parameter_score(&q[at[0]], &mut p).unwrap();
                for (at, score) in
                    at.into_iter()
                        .zip([law.event_score(q[at[0]]).unwrap(), p[0], p[1]])
                {
                    expected_g[at] += score;
                }
            }
            close(value, expected, 2e-14);
            for j in 0..g.len() {
                close(g[j], expected_g[j], 2e-14);
                let mut plus = q.clone();
                let mut minus = q.clone();
                plus[j] += 1e-5;
                minus[j] -= 1e-5;
                let diff = (model.logp_grad(&plus, &mut vec![0.; g.len()]).unwrap()
                    - model.logp_grad(&minus, &mut vec![0.; g.len()]).unwrap())
                    / 2e-5;
                close(g[j], diff, 2e-8);
            }
            let count = allocation_counter::measure(|| {
                for _ in 0..8 {
                    model.logp_grad(&q, &mut g).unwrap();
                }
            });
            assert_eq!(count.count_total, 0);
        }
    }
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    let sigma = b.constant(2.).unwrap();
    let f = normal_batch(&mut b, &[x, x, x, x, x], &[x], &[sigma]).unwrap();
    b.factor(f).unwrap();
    let mut g = [17.];
    b.compile().unwrap().logp_grad(&[0.3], &mut g).unwrap();
    assert_eq!(g, [0.]);
}

#[test]
fn packet_fallback_preserves_first_typed_error_checks_inactive_scores_and_recovers() {
    let mut b = ModelBuilder::new();
    let handles: Vec<_> = (0..15)
        .map(|i| b.real(&format!("x{i}")).unwrap().expr())
        .collect();
    let batch = NormalBatch::new(&handles[..5], &handles[5..10], &handles[10..]).unwrap();
    let mut work = batch.workspace();
    let mut q = vec![0.; 15];
    q[10..].fill(1.);
    for (event, mu, sigma) in [
        (f64::MAX, -f64::MAX, 1.),
        (f64::INFINITY, 0., 1.),
        (f64::NAN, 0., 1.),
        (0., f64::INFINITY, -1.),
        (1., 0., f64::from_bits(1)),
    ] {
        q[0] = event;
        q[5] = mu;
        q[10] = sigma;
        q[11] = -1.; // later constructor failure
        let scalar = NormalFactor::new(handles[0], handles[5], handles[10]);
        let expected = scalar
            .evaluate(&[event, mu, sigma], &mut [0.; 3], &mut ())
            .unwrap_err();
        let mut g = vec![17.; 15];
        let actual = batch.evaluate(&q, &mut g, &mut work).unwrap_err();
        assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
        assert_eq!(g, vec![17.; 15]);
        q.fill(0.);
        q[10..].fill(1.);
        assert!(batch.evaluate(&q, &mut g, &mut work).unwrap().is_finite());
    }
    assert!(matches!(
        batch.evaluate(&q[..14], &mut [0.; 15], &mut work),
        Err(AssemblyError::Shape)
    ));
    let shared = NormalBatch::observed(vec![1.; 5], &[handles[0]], &[handles[1]]).unwrap();
    assert!(
        shared
            .evaluate(&[0., 1e-200], &mut [0.; 2], &mut shared.workspace())
            .is_err()
    );
}

#[test]
fn fixed_data_rows_rebinding_source_and_one_scale_jacobian_use_existing_interfaces() {
    let mut b = ModelBuilder::new();
    let mu = b.real("mu").unwrap();
    let sigma = b.positive("sigma").unwrap();
    let primitive =
        NormalBatch::observed(vec![-1., 0.5, 2., -0.25, 1.], &[mu.expr()], &[sigma.expr()])
            .unwrap();
    assert_eq!(
        allocation_counter::measure(|| {
            std::hint::black_box(primitive.batch_workspace());
        })
        .count_total,
        0
    );
    let f = b.primitive(primitive).unwrap();
    b.likelihood("data", f).unwrap();
    let model = b.compile().unwrap();
    assert_eq!(model.expression_count(), 3);
    assert!(
        model
            .rust_first_order_gradient(true)
            .unwrap()
            .unwrap()
            .0
            .contains("NormalBatch")
    );
    let batch = model.batch_likelihood().unwrap();
    let q = [0.2, 0.3];
    let mut rows = [0.; 5];
    let likelihood = batch.evaluate(&q, &mut rows).unwrap();
    close(likelihood, rows.iter().sum(), 1e-14);
    let model = batch
        .into_model()
        .rebind::<NormalBatch>(f, |p| p.with_observations(vec![0.; 5]))
        .unwrap();
    let target = model.into_target().unwrap();
    let mut g = [0.; 2];
    let value = target.logp_grad(&q, &mut g).unwrap();
    let law = Normal::new(q[0], q[1].exp()).unwrap();
    close(value, 5. * law.log_prob(&0.).unwrap() + q[1], 1e-14);
    for j in 0..2 {
        let mut plus = q;
        let mut minus = q;
        plus[j] += 1e-5;
        minus[j] -= 1e-5;
        close(
            g[j],
            (target.logp_grad(&plus, &mut [0.; 2]).unwrap()
                - target.logp_grad(&minus, &mut [0.; 2]).unwrap())
                / 2e-5,
            1e-8,
        );
    }
    // Shape/owner rejection must not register a factor or allocate its workspace.
    let mut b = ModelBuilder::new();
    let x = b.real("x").unwrap().expr();
    let mut foreign = ModelBuilder::new();
    let y = foreign.real("y").unwrap().expr();
    assert!(normal_batch(&mut b, &[], &[x], &[x]).is_err());
    assert!(normal_batch(&mut b, &[x, x, x], &[x, x], &[x]).is_err());
    assert!(normal_batch(&mut b, &[x], &[y], &[x]).is_err());
    let scalar = normal(&mut b, x, 0., 1.).unwrap();
    b.factor(scalar).unwrap();
    assert_eq!(b.compile().unwrap().expression_count(), 4);
}
