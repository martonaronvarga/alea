use alea_core::distribution::{
    Distribution, DistributionError as E, EventShape, ParameterScore, ScalarEventScore, Support,
};
use alea_distributions::{mixture::FiniteMixture, scalar::Normal};
use std::{cell::Cell, rc::Rc};

struct Counted {
    calls: Rc<Cell<usize>>,
    value: Rc<Cell<Result<f64, E>>>,
    score: Rc<Cell<Result<f64, E>>>,
}
impl ScalarEventScore for Counted {
    fn event_score(&self, _: f64) -> Result<f64, E> {
        self.score.get()
    }
}
impl ParameterScore for Counted {
    fn parameter_count(&self) -> usize {
        1
    }
    fn parameter_score(&self, _: &f64, output: &mut [f64]) -> Result<(), E> {
        if output.len() != 1 {
            return Err(E::Shape);
        }
        output[0] = self.score.get()?;
        Ok(())
    }
}
impl Distribution for Counted {
    type Event = f64;
    type Error = E;
    fn event_shape(&self) -> EventShape {
        EventShape::Scalar
    }
    fn support(&self) -> Support {
        Support::Real
    }
    fn reference_measure(&self) -> alea_core::distribution::ReferenceMeasure {
        alea_core::distribution::ReferenceMeasure::Lebesgue { dimension: 1 }
    }
    fn log_prob(&self, _: &f64) -> Result<f64, E> {
        self.calls.set(self.calls.get() + 1);
        self.value.get()
    }
}

#[test]
fn mixture_cache_reuses_terms_preserves_order_and_stays_allocation_free_and_sync() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<FiniteMixture<Normal>>();
    for n in [2, 8, 31, 32, 33, 65] {
        let weights: Vec<_> = (0..n)
            .map(|i| if i == 0 { 0. } else { 1. / (n - 1) as f64 })
            .collect();
        let values: Vec<_> = (0..n)
            .map(|i| [-1e300, -745., -40., 0., f64::NEG_INFINITY][i % 5])
            .collect();
        let counters: Vec<_> = (0..n).map(|_| Rc::new(Cell::new(0))).collect();
        let components = (0..n)
            .map(|i| Counted {
                calls: counters[i].clone(),
                score: Rc::new(Cell::new(Ok(2.))),
                value: Rc::new(Cell::new(if i == 0 {
                    Err(E::Observation)
                } else {
                    Ok(values[i])
                })),
            })
            .collect();
        let mixture = FiniteMixture::new(weights.clone(), components).unwrap();
        let terms: Vec<_> = weights
            .iter()
            .zip(values)
            .filter(|(w, _)| **w > 0.)
            .map(|(w, v)| w.ln() + v)
            .collect();
        let max = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let expected = max + terms.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
        assert_eq!(mixture.log_prob(&0.).unwrap().to_bits(), expected.to_bits());
        for (i, calls) in counters.iter().enumerate() {
            assert_eq!(
                calls.get(),
                if i == 0 {
                    0
                } else if i < 32 {
                    1
                } else {
                    2
                }
            );
        }
        let allocation = allocation_counter::measure(|| {
            std::hint::black_box(mixture.log_prob(&0.).unwrap());
        });
        assert_eq!(allocation.count_total, 0);
    }
}

#[test]
fn mixture_infinite_density_does_not_hide_later_failures_and_recovery_is_clean() {
    let value = Rc::new(Cell::new(Err(E::Observation)));
    let mixture = FiniteMixture::new(
        vec![0.5, 0.5],
        vec![
            Counted {
                calls: Rc::new(Cell::new(0)),
                value: Rc::new(Cell::new(Ok(f64::INFINITY))),
                score: Rc::new(Cell::new(Ok(2.))),
            },
            Counted {
                calls: Rc::new(Cell::new(0)),
                value: value.clone(),
                score: Rc::new(Cell::new(Ok(2.))),
            },
        ],
    )
    .unwrap();
    assert!(matches!(mixture.log_prob(&0.), Err(E::Observation)));
    value.set(Ok(f64::NAN));
    assert!(matches!(mixture.log_prob(&0.), Err(E::Numerical)));
    value.set(Ok(f64::NEG_INFINITY));
    assert_eq!(mixture.log_prob(&0.).unwrap(), f64::INFINITY);
}

#[test]
fn derivative_requests_reuse_prefix_densities_and_allocate_nothing() {
    for n in [1, 2, 8, 31, 32, 33, 65] {
        let counters: Vec<_> = (0..n).map(|_| Rc::new(Cell::new(0))).collect();
        let components = (0..n)
            .map(|i| Counted {
                calls: counters[i].clone(),
                value: Rc::new(Cell::new(Ok(
                    [-1e300, -745., -40., 0., f64::NEG_INFINITY][i % 5]
                ))),
                score: Rc::new(Cell::new(Ok(2.))),
            })
            .collect();
        let mixture = FiniteMixture::new(vec![1. / n as f64; n], components).unwrap();
        let mut scores = vec![0.; mixture.parameter_count()];
        for parameter in [false, true] {
            for calls in &counters {
                calls.set(0);
            }
            if parameter {
                mixture.parameter_score(&0., &mut scores).unwrap();
            } else {
                assert!(mixture.event_score(0.).unwrap().is_finite());
            }
            for (i, calls) in counters.iter().enumerate() {
                assert_eq!(calls.get(), if i < 32 { 1 } else { 3 }, "n={n},i={i}");
            }
        }
        let allocation = allocation_counter::measure(|| {
            std::hint::black_box(mixture.event_score(0.).unwrap());
            mixture.parameter_score(&0., &mut scores).unwrap();
        });
        assert_eq!(allocation.count_total, 0);
    }
}

#[test]
fn scores_keep_original_bits_across_cache_boundaries_and_extreme_events() {
    for n in [1, 2, 8, 31, 32, 33, 65] {
        let weights = vec![1. / n as f64; n];
        for scale in [0.7, 1e-154, 1e154] {
            let components: Vec<_> = (0..n)
                .map(|i| Normal::new(i as f64 / 3., scale).unwrap())
                .collect();
            let mixture = FiniteMixture::new(weights.clone(), components.clone()).unwrap();
            for x in [
                -1e150,
                -2.1,
                -0.1,
                0.,
                0.5,
                10.,
                1e150,
                f64::INFINITY,
                f64::NAN,
            ] {
                // The previous public-law sequence, not replacement Normal math.
                let reference = (|| {
                    let logp = mixture.log_prob(&x)?;
                    if !logp.is_finite() {
                        return Err(E::Numerical);
                    }
                    let mut event = 0.;
                    let mut parameters = vec![0.; 3 * n];
                    for (i, (weight, law)) in weights.iter().zip(&components).enumerate() {
                        let r = (weight.ln() + law.log_prob(&x)? - logp).exp();
                        parameters[i] = r / weight;
                        if r > 0. {
                            event += r * law.event_score(x)?;
                            let score = &mut parameters[n + 2 * i..n + 2 * i + 2];
                            law.parameter_score(&x, score)?;
                            for g in score {
                                *g *= r;
                            }
                        }
                    }
                    if !event.is_finite() || parameters.iter().any(|g| !g.is_finite()) {
                        return Err(E::Numerical);
                    }
                    Ok((event, parameters))
                })();
                let mut scores = vec![0.; mixture.parameter_count()];
                let actual = mixture.event_score(x).and_then(|event| {
                    mixture.parameter_score(&x, &mut scores)?;
                    Ok((event, scores))
                });
                match (actual, reference) {
                    (Ok((event, scores)), Ok((expected, parameters))) => {
                        assert_eq!(event.to_bits(), expected.to_bits(), "n={n},x={x}");
                        assert_eq!(
                            scores.iter().map(|g| g.to_bits()).collect::<Vec<_>>(),
                            parameters.iter().map(|g| g.to_bits()).collect::<Vec<_>>()
                        );
                    }
                    (Err(a), Err(b)) => assert_eq!(a, b, "n={n},scale={scale},x={x}"),
                    (a, b) => panic!("n={n},scale={scale},x={x}: {a:?} != {b:?}"),
                }
            }
        }
    }
}

#[test]
fn derivative_errors_and_underflowed_responsibilities_remain_visible() {
    let first = Rc::new(Cell::new(Ok(0.)));
    let second = Rc::new(Cell::new(Err(E::Observation)));
    let score = Rc::new(Cell::new(Err(E::Parameter("component score"))));
    let calls = Rc::new(Cell::new(0));
    let mixture = FiniteMixture::new(
        vec![0.5, 0.5],
        vec![
            Counted {
                calls: calls.clone(),
                value: first.clone(),
                score: Rc::new(Cell::new(Ok(2.))),
            },
            Counted {
                calls: calls.clone(),
                value: second.clone(),
                score: score.clone(),
            },
        ],
    )
    .unwrap();
    let mut wrong = [17.; 3];
    assert_eq!(mixture.parameter_score(&0., &mut wrong), Err(E::Shape));
    assert_eq!(wrong, [17.; 3]);
    assert_eq!(calls.get(), 0);
    let mut output = [17.; 4];
    assert_eq!(mixture.event_score(0.), Err(E::Observation));
    assert_eq!(
        mixture.parameter_score(&0., &mut output),
        Err(E::Observation)
    );
    assert_eq!(output, [17.; 4]);
    first.set(Ok(f64::INFINITY));
    assert_eq!(mixture.event_score(0.), Err(E::Observation));
    assert_eq!(
        mixture.parameter_score(&0., &mut output),
        Err(E::Observation)
    );
    first.set(Ok(0.));
    second.set(Ok(-1000.));
    // An exactly underflowed responsibility must not request its undefined score.
    assert_eq!(mixture.event_score(0.), Ok(2.));
    mixture.parameter_score(&0., &mut output).unwrap();
    assert_eq!(output, [2., 0., 2., 0.]);
    second.set(Ok(0.));
    let error = E::Parameter("component score");
    assert_eq!(mixture.event_score(0.), Err(error));
    assert_eq!(mixture.parameter_score(&0., &mut output), Err(error));
    score.set(Ok(f64::NAN));
    assert_eq!(mixture.event_score(0.), Err(E::Numerical));
    assert_eq!(mixture.parameter_score(&0., &mut output), Err(E::Numerical));
    score.set(Ok(2.));
    assert_eq!(mixture.event_score(0.), Ok(2.));
    mixture.parameter_score(&0., &mut output).unwrap();
    assert_eq!(output, [1., 1., 1., 1.]);

    let zero_weight = FiniteMixture::new(
        vec![0., 1.],
        vec![
            Counted {
                calls: Rc::new(Cell::new(0)),
                value: Rc::new(Cell::new(Err(E::Observation))),
                score: Rc::new(Cell::new(Err(error))),
            },
            Counted {
                calls,
                value: second,
                score,
            },
        ],
    )
    .unwrap();
    assert_eq!(zero_weight.event_score(0.), Ok(2.));
    output.fill(17.);
    assert_eq!(
        zero_weight.parameter_score(&0., &mut output),
        Err(E::Observation)
    );
    assert_eq!(output, [17.; 4]);
}

#[test]
fn native_mixture_pullbacks_match_finite_differences_and_stage_failed_gradients() {
    use alea_core::{
        model::assembly::{ModelBuilder, Simplex},
        target::LogDensityGradient,
        transform::Transform,
    };
    use alea_distributions::modeling;

    for n in [2, 3, 8, 33] {
        let mut b = ModelBuilder::new();
        let labels: Vec<_> = (0..n).map(|i| format!("c{i}")).collect();
        let names: Vec<_> = labels.iter().map(String::as_str).collect();
        let weights = b
            .vector::<Simplex>(
                "weights",
                "component",
                &names,
                "simplex",
                Transform::simplex(n).unwrap(),
            )
            .unwrap();
        let event = b.real("event").unwrap();
        let mut parameters = weights.expressions().to_vec();
        for i in 0..n {
            parameters.push(b.real(&format!("mean{i}")).unwrap().expr());
            parameters.push(b.positive(&format!("scale{i}")).unwrap().expr());
        }
        let factor = modeling::continuous(&mut b, event.expr(), &parameters, move |p| {
            FiniteMixture::new(
                p[..n].to_vec(),
                p[n..]
                    .chunks_exact(2)
                    .map(|p| Normal::new(p[0], p[1]))
                    .collect::<Result<Vec<_>, _>>()?,
            )
        })
        .unwrap();
        b.likelihood("mixture", factor).unwrap();
        let target = b.compile().unwrap().into_target().unwrap();
        let q: Vec<_> = (0..3 * n).map(|i| (i as f64 / 7.).sin() * 0.2).collect();
        let mut gradient = vec![0.; q.len()];
        let value = target.logp_grad(&q, &mut gradient).unwrap();
        let expected = gradient.clone();
        for j in 0..q.len() {
            let mut plus = q.clone();
            let mut minus = q.clone();
            plus[j] += 1e-5;
            minus[j] -= 1e-5;
            let fd = (target.logp_grad(&plus, &mut gradient).unwrap()
                - target.logp_grad(&minus, &mut gradient).unwrap())
                / 2e-5;
            assert!(
                (fd - expected[j]).abs() < 2e-7 * (1. + expected[j].abs()),
                "n={n},j={j}: {fd} != {}",
                expected[j]
            );
        }
        let mut bad = q.clone();
        bad[n + 1] = -1000.;
        gradient.fill(17.);
        assert!(target.logp_grad(&bad, &mut gradient).is_err());
        assert!(gradient.iter().all(|x| *x == 17.));
        assert_eq!(
            target.logp_grad(&q, &mut gradient).unwrap().to_bits(),
            value.to_bits()
        );
        assert_eq!(gradient, expected);
    }
}

#[test]
fn unsized_vector_events_reuse_component_workspaces_and_preserve_scores() {
    use alea_distributions::multivariate::MultivariateNormal;
    for n in [2, 33] {
        let construct = |i: usize| {
            MultivariateNormal::new(
                vec![i as f64 / 5., -(i as f64) / 7.],
                vec![1.3, 0., 0.2, 0.7],
            )
            .unwrap()
        };
        let weights = vec![1. / n as f64; n];
        let mixture = FiniteMixture::new(weights.clone(), (0..n).map(construct).collect()).unwrap();
        let event = [0.3, -0.4];
        let logp = mixture.log_prob(&event).unwrap();
        let mut expected = vec![0.; mixture.parameter_count()];
        for (i, weight) in weights.iter().enumerate() {
            let law = construct(i);
            let r = (weight.ln() + law.log_prob(&event).unwrap() - logp).exp();
            expected[i] = r / weight;
            let score = &mut expected[n + 5 * i..n + 5 * i + 5];
            law.parameter_score(&event, score).unwrap();
            for g in score {
                *g *= r;
            }
        }
        let mut actual = vec![0.; expected.len()];
        mixture.parameter_score(&event, &mut actual).unwrap();
        assert_eq!(
            actual.iter().map(|g| g.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|g| g.to_bits()).collect::<Vec<_>>()
        );
        let allocation = allocation_counter::measure(|| {
            mixture.parameter_score(&event, &mut actual).unwrap();
        });
        assert_eq!(allocation.count_total, 0);
        actual.fill(17.);
        assert_eq!(mixture.parameter_score(&[0.], &mut actual), Err(E::Shape));
        assert!(actual.iter().all(|g| *g == 17.));
        assert_eq!(
            mixture.parameter_score(&[0., f64::NAN], &mut actual),
            Err(E::Observation)
        );
        assert!(actual.iter().all(|g| *g == 17.));
        mixture.parameter_score(&event, &mut actual).unwrap();
        assert_eq!(actual, expected);
    }
}
