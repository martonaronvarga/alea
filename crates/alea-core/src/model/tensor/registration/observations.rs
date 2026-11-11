//! R0.2c: one private observed-law map for existing sized-event parameter-score
//! laws. Whole tensor edges stay on the existing graph. No law formula, public
//! per-family tensor wrapper, source rule or independent graph is introduced.
use super::*;
// The unit-test crate and the core library used by our distribution dev-dependency
// have distinct Rust identities. Use that dependency's public trait/laws together;
// graph ownership remains exclusively this existing private test graph.
use alea::distribution::ParameterScore;
use alea_math::numeric::CompensatedSum;

fn observe<D, F>(
    builder: &mut RegionBuilder,
    events: Vec<D::Event>,
    parameters: &[Value],
    construct: F,
) -> Result<Value, E>
where
    D: ParameterScore + 'static,
    D::Event: Sized + Send + Sync + 'static,
    F: Fn(&[f64]) -> Result<D, D::Error> + Send + Sync + 'static,
{
    let rows = events.len();
    let count = parameters.len();
    if count == 0 {
        return Err(E::Shape);
    }
    let shape = Shape::new(vec![rows])?;
    let mut types = Vec::with_capacity(count);
    let mut lengths = Vec::with_capacity(count);
    for &value in parameters {
        let node = builder.real_node(value)?;
        if node.shape != Shape::scalar() && node.shape != shape {
            return Err(E::Shape);
        }
        types.push(BlockType::of(node));
        lengths.push(node.shape.elements);
    }
    let scores = rows.checked_mul(count).ok_or(E::Shape)?;
    let scratch = extent(scores, count)?;
    let signature = Signature {
        inputs: types,
        output: BlockType {
            dtype: ElementType::Real,
            shape,
        },
        scratch,
    };
    builder.register(signature, parameters, move || {
        Ok(Kernel {
            binding: None,
            forward: Box::new(move |inputs, out, scratch| {
                let (scores, point) = scratch.split_at_mut(scores);
                for (i, event) in events.iter().enumerate() {
                    for (j, parameter) in point.iter_mut().enumerate() {
                        let input = inputs.real(j)?;
                        *parameter = input[if input.len() == 1 { 0 } else { i }];
                    }
                    let law = construct(point).map_err(|e| E::Factor(Box::new(e)))?;
                    if law.parameter_count() != count {
                        return Err(E::Shape);
                    }
                    let local = &mut scores[i * count..(i + 1) * count];
                    local.fill(f64::NAN);
                    let value = law
                        .log_prob_parameter_score(event, local)
                        .map_err(|e| E::Factor(Box::new(e)))?;
                    // Admit local scores even for fixed/inactive inputs or zero
                    // reverse seeds. Saving them must not defer forward failures.
                    if !value.is_finite() || local.iter().any(|v| !v.is_finite()) {
                        return Err(E::Numerical);
                    }
                    out[i] = value;
                }
                Ok(())
            }),
            vjp: Some(Box::new(move |_, _, seed, g, scratch| {
                let mut start = 0;
                for (j, &length) in lengths.iter().enumerate() {
                    if length == 1 {
                        let mut sum = CompensatedSum::default();
                        for (i, &seed) in seed.iter().enumerate() {
                            sum.add(seed * scratch[i * count + j]);
                        }
                        g[start] = sum.total();
                    } else {
                        for (i, &seed) in seed.iter().enumerate() {
                            g[start + i] = seed * scratch[i * count + j];
                        }
                    }
                    start += length;
                }
                Ok(())
            })),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::LogDensityGradient;
    use alea::{model::assembly::ModelBuilder, target::LogDensityGradient as _};
    use alea_distributions::{
        modeling,
        scalar::{BernoulliLogit, Normal},
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * (1. + b.abs()), "{a} != {b}");
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
    // The opaque map is deliberately nonlinear. Both families share its one
    // output edge, so reverse contributions must meet at that existing node.
    fn build(n: usize, family: u8) -> (TensorProgram, Arc<AtomicUsize>) {
        let (_, events, binary) = data(n);
        let calls = Arc::new(AtomicUsize::new(0));
        let mut b = RegionBuilder::new();
        let beta = b.input("beta", Shape::new(vec![2, 1]).unwrap()).unwrap();
        let rho = b.input("rho", Shape::scalar()).unwrap();
        let design = b.data("design", Shape::new(vec![n, 2]).unwrap()).unwrap();
        let eta = b.matmul(design, beta).unwrap();
        let eta = b.reshape(eta, Shape::new(vec![n]).unwrap()).unwrap();
        let mapped = b
            .register(
                Signature {
                    inputs: vec![BlockType::of(b.node(eta).unwrap())],
                    output: BlockType {
                        dtype: ElementType::Real,
                        shape: Shape::new(vec![n]).unwrap(),
                    },
                    scratch: 0,
                },
                &[eta],
                || {
                    Ok(Kernel {
                        binding: None,
                        forward: Box::new(|inputs, out, _| {
                            for (out, &x) in out.iter_mut().zip(inputs.real(0)?) {
                                *out = x + 0.1 * x * x;
                            }
                            Ok(())
                        }),
                        vjp: Some(Box::new(|inputs, _, seed, g, _| {
                            for ((g, &x), &seed) in g.iter_mut().zip(inputs.real(0)?).zip(seed) {
                                *g = seed * (1. + 0.2 * x);
                            }
                            Ok(())
                        })),
                    })
                },
            )
            .unwrap();
        let sigma = b.unary(UnaryOp::Exp, rho).unwrap();
        let mut terms = Vec::new();
        if family & 1 != 0 {
            let calls = Arc::clone(&calls);
            let value = observe::<Normal, _>(&mut b, events, &[mapped, sigma], move |p| {
                calls.fetch_add(1, Ordering::Relaxed);
                Normal::new(p[0], p[1])
            })
            .unwrap();
            terms.push(b.sum(value).unwrap());
        }
        if family & 2 != 0 {
            let calls = Arc::clone(&calls);
            let value = observe::<BernoulliLogit, _>(&mut b, binary, &[mapped], move |p| {
                calls.fetch_add(1, Ordering::Relaxed);
                BernoulliLogit::new(p[0])
            })
            .unwrap();
            terms.push(b.sum(value).unwrap());
        }
        let output = if terms.len() == 1 {
            terms[0]
        } else {
            b.binary(BinaryOp::Add, terms[0], terms[1]).unwrap()
        };
        (b.compile(output).unwrap(), calls)
    }
    // Independent formulas are test oracles only; all evaluated laws above and
    // in the scalar control below use Alea's production implementations.
    fn oracle(design: &[f64], q: &[f64], family: u8) -> (f64, [f64; 3]) {
        let (_, events, binary) = data(design.len() / 2);
        let sigma = q[2].exp();
        let mut value = 0.;
        let mut gradient = [0.; 3];
        for (i, row) in design.chunks_exact(2).enumerate() {
            let eta = row[0] * q[0] + row[1] * q[1];
            let mu = eta + 0.1 * eta * eta;
            let mut score = 0.;
            if family & 1 != 0 {
                let z = (events[i] - mu) / sigma;
                value += -0.5 * z * z - q[2] - 0.5 * std::f64::consts::TAU.ln();
                score += z / sigma;
                gradient[2] += z * z - 1.;
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
            for j in 0..2 {
                gradient[j] += score * (1. + 0.2 * eta) * row[j];
            }
        }
        (value, gradient)
    }
    // Exercise the production owned adapter; the old lifecycle-only target is removed.
    fn target(
        program: &TensorProgram,
        data: &[(&str, &[f64])],
    ) -> Result<impl crate::target::LogDensityGradient + use<>, E> {
        let binding = program.bind(data)?;
        let mut builder = crate::model::assembly::ModelBuilder::new();
        let inputs = (0..binding.dimension())
            .map(|i| builder.real(&format!("x{i}")).map(|x| x.expr()))
            .collect::<Result<Vec<_>, _>>()?;
        let output = builder.apply_region(&inputs, binding)?;
        builder.factor(output)?;
        Ok(builder
            .compile()?
            .into_target()
            .expect("builder's identity layout"))
    }
    fn scalar(design: &[f64], family: u8) -> impl alea::target::LogDensityGradient {
        let (_, events, binary) = data(design.len() / 2);
        let mut b = ModelBuilder::new();
        let beta = [b.real("a").unwrap().expr(), b.real("b").unwrap().expr()];
        let rho = b.real("rho").unwrap().expr();
        let sigma = b.exp(rho).unwrap();
        for (i, row) in design.chunks_exact(2).enumerate() {
            let a = b.constant(row[0]).unwrap();
            let a = b.multiply(a, beta[0]).unwrap();
            let c = b.constant(row[1]).unwrap();
            let c = b.multiply(c, beta[1]).unwrap();
            let eta = b.add(a, c).unwrap();
            let square = b.multiply(eta, eta).unwrap();
            let tenth = b.constant(0.1).unwrap();
            let correction = b.multiply(tenth, square).unwrap();
            let mu = b.add(eta, correction).unwrap();
            if family & 1 != 0 {
                let value = modeling::normal(&mut b, events[i], mu, sigma).unwrap();
                b.likelihood(&format!("normal-{i}"), value).unwrap();
            }
            if family & 2 != 0 {
                let value = modeling::bernoulli_logit(&mut b, binary[i], mu).unwrap();
                b.likelihood(&format!("logit-{i}"), value).unwrap();
            }
        }
        b.compile().unwrap().into_target().unwrap()
    }
    #[test]
    fn both_families_share_block_path_match_full_densities_scalar_graph_and_differences() {
        for family in [1, 2, 3] {
            let mut nodes = None;
            for n in [1, 3, 4, 5, 17, 257] {
                let (design, _, _) = data(n);
                let (p, calls) = build(n, family);
                assert_eq!(*nodes.get_or_insert(p.node_count()), p.node_count());
                let bound = p.bind(&[("design", &design)]).unwrap();
                let target = target(&p, &[("design", &design)]).unwrap();
                let scalar = scalar(&design, family);
                for q in [[0.3, -0.7, -0.2], [-1.2, 0.8, 0.4], [0., 0., 0.]] {
                    let before = calls.load(Ordering::Relaxed);
                    let mut g = [0.; 3];
                    let lp = target.logp_grad(&q, &mut g).unwrap();
                    assert_eq!(
                        calls.load(Ordering::Relaxed) - before,
                        n * (usize::from(family & 1 != 0) + usize::from(family & 2 != 0))
                    );
                    let (expected, dg) = oracle(&design, &q, family);
                    close(lp, expected, 3e-13);
                    let mut sg = [0.; 3];
                    close(lp, scalar.logp_grad(&q, &mut sg).unwrap(), 3e-13);
                    for j in 0..3 {
                        close(g[j], dg[j], 3e-12);
                        close(g[j], sg[j], 3e-12);
                        for h in [1e-4, 1e-5, 1e-6] {
                            let mut a = q;
                            let mut b = q;
                            a[j] += h;
                            b[j] -= h;
                            let fd = (oracle(&design, &a, family).0
                                - oracle(&design, &b, family).0)
                                / (2. * h);
                            close(g[j], fd, 3e-7);
                        }
                    }
                    let mut out = [17.];
                    let mut vg = [19.; 3];
                    bound
                        .value_and_vjp(&q, &[-0.7], &mut out, &mut vg, &mut bound.workspace())
                        .unwrap();
                    close(out[0], lp, 1e-14);
                    for j in 0..3 {
                        close(vg[j], -0.7 * g[j], 3e-12);
                    }
                }
                let changed: Vec<_> = design.iter().map(|v| v * 0.5).collect();
                let new = bound.rebind("design", &changed).unwrap();
                let mut out = [0.];
                let mut g = [0.; 3];
                let mut w = bound.workspace();
                for (binding, x) in [(&new, &changed), (&bound, &design)] {
                    binding
                        .value_and_vjp(&[0.3, -0.7, -0.2], &[1.], &mut out, &mut g, &mut w)
                        .unwrap();
                    close(out[0], oracle(x, &[0.3, -0.7, -0.2], family).0, 3e-13);
                }
                let allocations = allocation_counter::measure(|| {
                    for _ in 0..4 {
                        target.logp_grad(&[0.3, -0.7, -0.2], &mut g).unwrap();
                    }
                });
                assert_eq!(allocations.count_total, 0);
            }
        }
    }

    #[test]
    fn block_shape_admission_empty_maps_extremes_and_failure_recovery() {
        let mut b = RegionBuilder::new();
        let x = b.input("x", Shape::new(vec![2, 1]).unwrap()).unwrap();
        let mut foreign = RegionBuilder::new();
        let foreign = foreign.input("x", Shape::scalar()).unwrap();
        let before = b.nodes.len();
        assert!(matches!(
            observe::<BernoulliLogit, _>(&mut b, vec![true, true], &[foreign], |x| {
                BernoulliLogit::new(x[0])
            }),
            Err(E::ForeignHandle)
        ));
        assert!(matches!(
            observe::<BernoulliLogit, _>(&mut b, vec![true, true], &[x], |x| BernoulliLogit::new(
                x[0]
            )),
            Err(E::Shape)
        ));
        assert_eq!(b.nodes.len(), before);
        let discrete = b
            .typed_data(Shape::scalar(), Data::Bool(vec![true].into()))
            .unwrap();
        assert!(matches!(
            observe::<BernoulliLogit, _>(&mut b, vec![true], &[discrete], |x| BernoulliLogit::new(
                x[0]
            )),
            Err(E::Shape)
        ));
        assert!(matches!(
            observe::<BernoulliLogit, _>(&mut b, vec![true], &[], |x| BernoulliLogit::new(x[0])),
            Err(E::Shape)
        ));
        for family in [1, 2, 3] {
            let (p, calls) = build(0, family);
            let target = target(&p, &[("design", &[])]).unwrap();
            let mut g = [17.; 3];
            assert_eq!(target.logp_grad(&[0., 0., 0.], &mut g).unwrap(), 0.);
            assert_eq!(g, [0.; 3]);
            assert_eq!(calls.load(Ordering::Relaxed), 0);
        }
        // Scalar broadcast and repeated logical arguments must accumulate as
        // separate input occurrences, not overwrite the same destination.
        let mut b = RegionBuilder::new();
        let mu = b.input("shared", Shape::scalar()).unwrap();
        let value = observe::<Normal, _>(&mut b, vec![0.2, -0.3, 0.7], &[mu, mu], |p| {
            Normal::new(p[0], p[1])
        })
        .unwrap();
        let value = b.sum(value).unwrap();
        let p = b.compile(value).unwrap();
        let target = target(&p, &[]).unwrap();
        let mut g = [17.];
        let lp = target.logp_grad(&[1.2], &mut g).unwrap();
        let expected: f64 = [0.2, -0.3, 0.7]
            .iter()
            .map(|y| {
                let z = (y - 1.2) / 1.2;
                z / 1.2 + (z * z - 1.) / 1.2
            })
            .sum();
        close(g[0], expected, 2e-13);
        assert!(lp.is_finite());
        let mut out = [17.];
        let mut g = [19.];
        let bound = p.bind(&[]).unwrap();
        let mut w = bound.workspace();
        for seed in [0., 1.] {
            let error = bound
                .value_and_vjp(&[-1.], &[seed], &mut out, &mut g, &mut w)
                .unwrap_err();
            assert!(
                matches!(error,E::Factor(e) if matches!(e.downcast_ref::<alea::distribution::DistributionError>(),Some(alea::distribution::DistributionError::Parameter(_))))
            );
            assert_eq!((out, g), ([17.], [19.]));
        }
        bound
            .value_and_vjp(&[1.2], &[1.], &mut out, &mut g, &mut w)
            .unwrap();
        close(g[0], expected, 2e-13);
        // Extreme predictor failure cannot be hidden by a zero reverse seed.
        let (p, _) = build(5, 3);
        let (design, _, _) = data(5);
        let bound = p.bind(&[("design", &design)]).unwrap();
        let mut w = bound.workspace();
        let (mut out, mut g) = ([17.], [19.; 3]);
        assert!(
            bound
                .value_and_vjp(&[f64::MAX, 0., 0.], &[0.], &mut out, &mut g, &mut w)
                .is_err()
        );
        assert_eq!((out, g), ([17.], [19.; 3]));
        assert!(
            bound
                .value_and_vjp(&[0., 0., -1000.], &[0.], &mut out, &mut g, &mut w)
                .is_err()
        );
        assert_eq!((out, g), ([17.], [19.; 3]));
        bound
            .value_and_vjp(&[0.3, -0.7, 0.], &[1.], &mut out, &mut g, &mut w)
            .unwrap();
        close(out[0], oracle(&design, &[0.3, -0.7, 0.], 3).0, 3e-13);
    }

    #[test]
    fn incomplete_external_scores_fail_during_forward_even_with_zero_seeds_and_recover() {
        use alea::distribution::{
            Distribution, DistributionError, EventShape, ReferenceMeasure, Support,
        };
        struct FaultyScore {
            native: BernoulliLogit,
            incomplete: bool,
        }
        impl Distribution for FaultyScore {
            type Error = DistributionError;
            type Event = bool;
            fn event_shape(&self) -> EventShape {
                self.native.event_shape()
            }
            fn support(&self) -> Support {
                self.native.support()
            }
            fn reference_measure(&self) -> ReferenceMeasure {
                self.native.reference_measure()
            }
            fn log_prob(&self, event: &bool) -> Result<f64, DistributionError> {
                self.native.log_prob(event)
            }
        }
        impl ParameterScore for FaultyScore {
            fn parameter_count(&self) -> usize {
                1
            }
            fn parameter_score(
                &self,
                event: &bool,
                out: &mut [f64],
            ) -> Result<(), DistributionError> {
                self.native.parameter_score(event, out)
            }
            fn log_prob_parameter_score(
                &self,
                event: &bool,
                out: &mut [f64],
            ) -> Result<f64, DistributionError> {
                if self.incomplete {
                    self.native.log_prob(event)
                } else {
                    self.native.log_prob_parameter_score(event, out)
                }
            }
        }
        let incomplete = Arc::new(AtomicUsize::new(1));
        let flag = Arc::clone(&incomplete);
        let mut b = RegionBuilder::new();
        let x = b.input("x", Shape::scalar()).unwrap();
        let value = observe::<FaultyScore, _>(&mut b, vec![true, false], &[x], move |x| {
            Ok(FaultyScore {
                native: BernoulliLogit::new(x[0])?,
                incomplete: flag.load(Ordering::Relaxed) != 0,
            })
        })
        .unwrap();
        let value = b.sum(value).unwrap();
        let bound = b.compile(value).unwrap().bind(&[]).unwrap();
        let mut w = bound.workspace();
        let (mut out, mut g) = ([17.], [19.]);
        assert!(matches!(
            bound.evaluate(&[0.], &mut out, &mut w),
            Err(E::Numerical)
        ));
        for seed in [0., 1.] {
            assert!(matches!(
                bound.value_and_vjp(&[0.], &[seed], &mut out, &mut g, &mut w),
                Err(E::Numerical)
            ));
            assert_eq!((out, g), ([17.], [19.]));
        }
        incomplete.store(0, Ordering::Relaxed);
        bound
            .value_and_vjp(&[0.], &[1.], &mut out, &mut g, &mut w)
            .unwrap();
        close(out[0], -2. * 2_f64.ln(), 1e-14);
        assert_eq!(g, [0.]);
    }
}
