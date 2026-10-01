use alea_core::target::EvaluationError;
use alea_core::{capability::*, target::LogDensityGradient};
use std::convert::Infallible;
use std::{
    cell::Cell,
    panic::{AssertUnwindSafe, catch_unwind},
};
struct Target {
    partial: bool,
}
impl LogDensityGradient for Target {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        2
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
        g[0] = -q[0];
        g[1] = -4.0 * q[1];
        Ok(-0.5 * (q[0] * q[0] + 4.0 * q[1] * q[1]))
    }
}
impl HessianVector for Target {
    fn potential_hvp(&self, _: &[f64], v: &[f64], o: &mut [f64]) -> Result<(), Infallible> {
        o[0] = v[0];
        if !self.partial {
            o[1] = 4.0 * v[1];
        }
        Ok(())
    }
}
impl BatchLogDensityGradient for Target {
    fn batch_logp_grad(
        &self,
        s: BatchShape,
        q: &[f64],
        g: &mut [f64],
        r: &mut [BatchLane<Infallible>],
    ) {
        for (lane, result) in r.iter_mut().enumerate() {
            let a = s.index(lane, 0).unwrap();
            let b = s.index(lane, 1).unwrap();
            g[a] = -q[a];
            if !self.partial {
                g[b] = -4.0 * q[b];
            }
            *result = BatchLane::Complete(Ok(-0.5 * (q[a] * q[a] + 4.0 * q[b] * q[b])));
        }
    }
}
#[test]
fn potential_hvp_sign_shapes_and_partial_outputs() {
    let t = Target { partial: false };
    let mut out = [9.0; 2];
    evaluate_hvp(&t, &[1.0, 2.0], &[3.0, -4.0], &mut out).unwrap();
    assert_eq!(out, [3.0, -16.0]);
    assert!(evaluate_hvp(&t, &[1.0], &[0.0; 2], &mut out).is_err());
    assert_eq!(out, [3.0, -16.0]);
    assert!(matches!(
        evaluate_hvp(&Target { partial: true }, &[0.0; 2], &[1.0; 2], &mut out),
        Err(HvpError::NonFiniteOutput { index: 1 })
    ));
}
#[test]
fn batched_layouts_preserve_lane_identity_and_detect_partial_output() {
    for layout in [BatchLayout::ChainMajor, BatchLayout::CoordinateMajor] {
        let s = BatchShape::new(2, 3, layout).unwrap();
        let mut q = [0.0; 6];
        for lane in 0..3 {
            q[s.index(lane, 0).unwrap()] = lane as f64;
            q[s.index(lane, 1).unwrap()] = 1.0;
        }
        let mut g = [0.0; 6];
        let mut results = std::array::from_fn::<_, 3, _>(|_| BatchLane::Pending);
        evaluate_batch(&Target { partial: false }, s, &q, &mut g, &mut results).unwrap();
        for (lane, result) in results.iter().enumerate() {
            match result {
                BatchLane::Complete(Ok(v)) => assert_eq!(*v, -0.5 * ((lane * lane) as f64 + 4.0)),
                _ => panic!("lane not complete"),
            }
        }
        evaluate_batch(&Target { partial: true }, s, &q, &mut g, &mut results).unwrap();
        assert!(
            results
                .iter()
                .all(|r| matches!(r, BatchLane::Complete(Err(_))))
        );
        assert_eq!(s.index(3, 0), None);
    }
    assert!(BatchShape::new(usize::MAX, 2, BatchLayout::ChainMajor).is_err());
}

#[derive(Debug, thiserror::Error)]
#[error("injected backend failure {0}")]
struct BackendFailure(u32);

#[derive(Clone, Copy)]
enum Behavior {
    Good,
    Error,
    Panic,
    Mixed,
    Missing,
}

struct FaultTarget {
    behavior: Cell<Behavior>,
    calls: Cell<usize>,
}
impl FaultTarget {
    fn new(behavior: Behavior) -> Self {
        Self {
            behavior: Cell::new(behavior),
            calls: Cell::new(0),
        }
    }
}
impl LogDensityGradient for FaultTarget {
    type Error = BackendFailure;
    fn dimension(&self) -> usize {
        2
    }
    fn logp_grad(&self, _: &[f64], _: &mut [f64]) -> Result<f64, Self::Error> {
        panic!("optional capability must not fall back to scalar evaluation")
    }
}
impl HessianVector for FaultTarget {
    fn potential_hvp(&self, _: &[f64], v: &[f64], out: &mut [f64]) -> Result<(), Self::Error> {
        self.calls.set(self.calls.get() + 1);
        out[0] = v[0];
        match self.behavior.get() {
            Behavior::Error => return Err(BackendFailure(17)),
            Behavior::Panic => panic!("injected HVP panic"),
            _ => out[1] = v[1],
        }
        Ok(())
    }
}
impl BatchLogDensityGradient for FaultTarget {
    fn batch_logp_grad(
        &self,
        shape: BatchShape,
        _: &[f64],
        g: &mut [f64],
        r: &mut [BatchLane<Self::Error>],
    ) {
        self.calls.set(self.calls.get() + 1);
        for (lane, result) in r.iter_mut().enumerate() {
            let a = shape.index(lane, 0).unwrap();
            let b = shape.index(lane, 1).unwrap();
            g[a] = 2.0;
            if matches!(self.behavior.get(), Behavior::Panic) {
                // A completed status is not trustworthy until validation returns.
                *result = BatchLane::Complete(Ok(f64::INFINITY));
                panic!("injected batch panic");
            }
            if matches!(self.behavior.get(), Behavior::Missing) && lane == 0 {
                continue;
            }
            *result = match (self.behavior.get(), lane) {
                (Behavior::Mixed, 1) => {
                    BatchLane::Complete(Err(EvaluationError::Model(BackendFailure(29))))
                }
                (Behavior::Mixed, 2) | (Behavior::Missing, 1) => {
                    BatchLane::Complete(Ok(f64::INFINITY))
                }
                (Behavior::Mixed, 3) => BatchLane::Complete(Ok(-1.0)), // Missing second gradient.
                _ => {
                    g[b] = -3.0;
                    BatchLane::Complete(Ok(-1.0))
                }
            };
        }
    }
}

#[test]
fn hvp_preflight_never_invokes_backend_or_changes_output() {
    let target = FaultTarget::new(Behavior::Panic);
    for (q, v, n) in [
        (vec![0.0], vec![0.0; 2], 2),
        (vec![0.0; 2], vec![0.0], 2),
        (vec![0.0; 2], vec![0.0; 2], 1),
        (vec![0.0, f64::NAN], vec![0.0; 2], 2),
        (vec![0.0; 2], vec![f64::INFINITY, 0.0], 2),
    ] {
        let mut out = vec![123.0; n];
        assert!(evaluate_hvp(&target, &q, &v, &mut out).is_err());
        assert_eq!(out, vec![123.0; n]); // Exact unchanged sentinels.
    }
    assert_eq!(target.calls.get(), 0);
}

#[test]
fn hvp_preserves_original_error_and_recovers_after_partial_write_or_unwind() {
    let target = FaultTarget::new(Behavior::Error);
    let mut out = [123.0; 2];
    assert!(matches!(
        evaluate_hvp(&target, &[0.0; 2], &[1.0, 2.0], &mut out),
        Err(HvpError::Evaluation(EvaluationError::Model(
            BackendFailure(17)
        )))
    ));
    assert!(out[1].is_nan());
    target.behavior.set(Behavior::Panic);
    assert!(
        catch_unwind(AssertUnwindSafe(|| evaluate_hvp(
            &target,
            &[0.0; 2],
            &[1.0, 2.0],
            &mut out
        )))
        .is_err()
    );
    target.behavior.set(Behavior::Good);
    evaluate_hvp(&target, &[0.0; 2], &[1.0, 2.0], &mut out).unwrap();
    assert_eq!(out, [1.0, 2.0]); // Identity action, exact copies.
    assert_eq!(target.calls.get(), 3);
}

#[test]
fn batch_failures_are_lane_local_in_both_layouts() {
    for layout in [BatchLayout::ChainMajor, BatchLayout::CoordinateMajor] {
        let shape = BatchShape::new(2, 5, layout).unwrap();
        let target = FaultTarget::new(Behavior::Mixed);
        let mut g = [123.0; 10];
        let mut r = std::array::from_fn::<_, 5, _>(|_| BatchLane::Pending);
        evaluate_batch(&target, shape, &[0.0; 10], &mut g, &mut r).unwrap();
        assert!(matches!(
            r[1],
            BatchLane::Complete(Err(EvaluationError::Model(BackendFailure(29))))
        ));
        assert!(matches!(
            r[2],
            BatchLane::Complete(Err(EvaluationError::NonFiniteLogDensity))
        ));
        assert!(matches!(
            r[3],
            BatchLane::Complete(Err(EvaluationError::NonFiniteGradient { index: 1 }))
        ));
        for lane in [0, 4] {
            assert!(matches!(r[lane], BatchLane::Complete(Ok(-1.0))));
            assert_eq!(g[shape.index(lane, 1).unwrap()], -3.0);
        }
        assert_eq!(target.calls.get(), 1);
    }
}

#[test]
fn batch_preflight_preserves_status_and_gradient_without_invoking_backend() {
    let target = FaultTarget::new(Behavior::Panic);
    for layout in [BatchLayout::ChainMajor, BatchLayout::CoordinateMajor] {
        for (d, q, gradient_len, lanes) in [
            (1, vec![0.0; 2], 2, 2),
            (2, vec![0.0; 3], 4, 2),
            (2, vec![0.0; 4], 3, 2),
            (2, vec![0.0; 4], 4, 1),
            (2, vec![0.0, f64::NEG_INFINITY, 0.0, 0.0], 4, 2),
        ] {
            let mut g = vec![123.0; gradient_len];
            let mut r: Vec<_> = (0..lanes).map(|_| BatchLane::Complete(Ok(17.0))).collect();
            assert!(
                evaluate_batch(
                    &target,
                    BatchShape::new(d, 2, layout).unwrap(),
                    &q,
                    &mut g,
                    &mut r
                )
                .is_err()
            );
            assert_eq!(g, vec![123.0; gradient_len]);
            assert!(r.iter().all(|r| matches!(r, BatchLane::Complete(Ok(17.0)))));
        }
    }
    assert_eq!(target.calls.get(), 0);
}

#[test]
fn batch_reuse_clears_stale_results_after_unwind_and_validates_lanes_after_missing_status() {
    for layout in [BatchLayout::ChainMajor, BatchLayout::CoordinateMajor] {
        let shape = BatchShape::new(2, 3, layout).unwrap();
        let target = FaultTarget::new(Behavior::Panic);
        let mut g = [123.0; 6];
        let mut r = std::array::from_fn::<_, 3, _>(|_| BatchLane::Complete(Ok(17.0)));
        assert!(
            catch_unwind(AssertUnwindSafe(|| evaluate_batch(
                &target, shape, &[0.0; 6], &mut g, &mut r
            )))
            .is_err()
        );
        // After unwinding none of these raw outputs is promised validated.
        target.behavior.set(Behavior::Missing);
        assert!(matches!(
            evaluate_batch(&target, shape, &[0.0; 6], &mut g, &mut r),
            Err(BatchError::Incomplete { lane: 0 })
        ));
        assert!(matches!(r[0], BatchLane::Pending));
        assert!(matches!(
            r[1],
            BatchLane::Complete(Err(EvaluationError::NonFiniteLogDensity))
        ));
        assert!(matches!(r[2], BatchLane::Complete(Ok(-1.0))));
        target.behavior.set(Behavior::Good);
        evaluate_batch(&target, shape, &[0.0; 6], &mut g, &mut r).unwrap();
        assert!(r.iter().all(|r| matches!(r, BatchLane::Complete(Ok(-1.0)))));
        assert!(g.iter().all(|v| v.is_finite()));
        assert_eq!(target.calls.get(), 3);
    }
}
