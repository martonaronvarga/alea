//! Composition checks for independent, heterogeneous fits in caller-owned pools.
use alea::{
    distributions::modeling::regression::{RegressionData, normal_regression},
    model::assembly::ModelBuilder,
    optimize::{
        LbfgsOptions, MaximizeError, ObjectiveDescription, ObjectiveKind, OptimizationResult,
        OptimizationTermination, maximize_with_cancel,
    },
    target::{EvaluationError, LogDensityGradient, TargetSpace},
};
use rayon::prelude::*;
use std::{
    cell::Cell,
    error::Error,
    fmt,
    sync::{
        Barrier,
        atomic::{AtomicUsize, Ordering},
    },
};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Copy)]
enum Policy {
    Ordinary,
    Fail(usize),
    Cancel(usize),
}
#[derive(Clone, Copy)]
struct Job {
    id: usize,
    policy: Policy,
}

#[derive(Debug)]
enum ProbeError<E> {
    Model(E),
    Injected { id: usize, call: usize },
}
impl<E: fmt::Display> fmt::Display for ProbeError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(e) => e.fmt(f),
            Self::Injected { id, call } => write!(f, "injected fit {id} call {call}"),
        }
    }
}
impl<E: Error + 'static> Error for ProbeError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Model(e) => Some(e),
            Self::Injected { .. } => None,
        }
    }
}
struct Probe<'a, T> {
    target: T,
    job: Job,
    calls: Cell<usize>,
    token: &'a CancellationToken,
    cancel_during_call: bool,
}
impl<T: TargetSpace> TargetSpace for Probe<'_, T> {
    type Error = ProbeError<T::Error>;
    fn dimension(&self) -> usize {
        self.target.dimension()
    }
}
impl<T: LogDensityGradient> LogDensityGradient for Probe<'_, T> {
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        let call = self.calls.get() + 1;
        self.calls.set(call);
        if matches!(self.job.policy, Policy::Fail(n) if n == call) {
            g.fill(f64::NAN);
            return Err(ProbeError::Injected {
                id: self.job.id,
                call,
            });
        }
        if self.cancel_during_call {
            self.token.cancel();
        }
        // Cancellation during this call must not interrupt its value/score writes.
        self.target.logp_grad(q, g).map_err(ProbeError::Model)
    }
}

#[derive(Default)]
struct Live {
    current: AtomicUsize,
    peak: AtomicUsize,
}
struct Lease<'a>(&'a Live);
impl Live {
    fn enter(&self) -> Lease<'_> {
        let count = self.current.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(count, Ordering::SeqCst);
        Lease(self)
    }
}
impl Drop for Lease<'_> {
    fn drop(&mut self) {
        self.0.current.fetch_sub(1, Ordering::SeqCst);
    }
}
#[derive(Debug)]
struct Record {
    id: usize,
    result: OptimizationResult,
    calls: usize,
    fault: Option<usize>,
}

fn fit(
    job: Job,
    live: &Live,
    barrier: Option<&Barrier>,
    token: &CancellationToken,
    cancel_during_call: bool,
) -> Record {
    // Lease spans model construction, optimization and candidate verification.
    let _lease = live.enter();
    if let Some(barrier) = barrier {
        barrier.wait();
    }
    let columns = 1 + job.id % 4;
    let pairs = [4, 32, 128][job.id % 3];
    let means: Vec<_> = (0..columns)
        .map(|j| (job.id + 1) as f64 / 8. + j as f64 / 4.)
        .collect();
    let mut b = ModelBuilder::new();
    let beta: Vec<_> = (0..columns)
        .map(|j| b.real(&format!("mean{j}")).unwrap().expr())
        .collect();
    let sigma = b.positive("sigma").unwrap();
    let mut design = Vec::new();
    let mut observations = Vec::new();
    for _ in 0..pairs {
        for (j, mean) in means.iter().enumerate() {
            for noise in [-1., 1.] {
                design.extend((0..columns).map(|k| if j == k { 1. } else { 0. }));
                observations.push(mean + noise);
            }
        }
    }
    // Orthogonal group indicators; independent analytic MLE is (means, sigma=1).
    let data = RegressionData::new(columns, design, observations).unwrap();
    let factor = normal_regression(&mut b, &beta, &sigma, data).unwrap();
    b.likelihood("observations", factor).unwrap();
    let target = b
        .compile()
        .unwrap()
        .select_likelihood()
        .unwrap()
        .into_objective()
        .unwrap();
    let probe = Probe {
        target,
        job,
        calls: Cell::new(0),
        token,
        cancel_during_call,
    };
    let initial = vec![0.; columns + 1];
    let outcome = maximize_with_cancel(
        &probe,
        &initial,
        ObjectiveDescription {
            kind: ObjectiveKind::Likelihood,
            measure: "original means and sigma".into(),
            model: format!("Gaussian indicators {columns}"),
            data: format!("job {} pairs {pairs}", job.id),
            source: "independent-fit composition test".into(),
        },
        LbfgsOptions::new(500, 5000)
            .unwrap()
            .with_score_tolerance(2e-7)
            .unwrap(),
        || {
            token.is_cancelled()
                || matches!(job.policy, Policy::Cancel(n) if probe.calls.get() >= n)
        },
    );
    let (result, fault) = match outcome {
        Ok(result) => (result, None),
        Err(MaximizeError::Evaluation(error)) => {
            let call = match error.source {
                EvaluationError::Model(ProbeError::Injected { id, call }) => {
                    assert_eq!(id, job.id);
                    call
                }
                source => panic!("unexpected error: {source:?}"),
            };
            (*error.result, Some(call))
        }
        Err(error) => panic!("invalid test fit: {error}"),
    };
    assert_eq!(result.evaluations, probe.calls.get());
    if let Some(candidate) = &result.candidate {
        let mut g = vec![f64::NAN; columns + 1];
        let value = probe.target.logp_grad(&candidate.position, &mut g).unwrap();
        assert_eq!(value, candidate.value);
        assert_eq!(g, candidate.gradient);
        assert!(candidate.gradient.iter().all(|g| g.is_finite()));
        // Collapse each +/-1 observation pair algebraically, independently of
        // the per-row density implementation and the optimizer's trial buffers.
        let count = (2 * pairs * columns) as f64;
        let q = &candidate.position;
        let variance = (2. * q[columns]).exp();
        let rss = (2 * pairs) as f64
            * means
                .iter()
                .zip(q)
                .map(|(m, q)| (m - q).powi(2) + 1.)
                .sum::<f64>();
        let expected_value =
            -count * (q[columns] + 0.5 * std::f64::consts::TAU.ln()) - 0.5 * rss / variance;
        assert!((candidate.value - expected_value).abs() < 1e-10 * count);
        for (j, mean) in means.iter().enumerate() {
            let expected_score = (2 * pairs) as f64 * (mean - q[j]) / variance;
            assert!((candidate.gradient[j] - expected_score).abs() < 1e-10 * count);
        }
        assert!((candidate.gradient[columns] - (rss / variance - count)).abs() < 1e-10 * count);
    } else {
        assert!(result.evaluations <= 1);
    }
    if matches!(job.policy, Policy::Ordinary) && !token.is_cancelled() {
        let q = &result.candidate.as_ref().unwrap().position;
        for (q, mean) in q.iter().zip(means) {
            assert!((q - mean).abs() < 1e-6);
        }
        assert!(q[columns].abs() < 1e-6);
    }
    Record {
        id: job.id,
        result,
        calls: probe.calls.get(),
        fault,
    }
}

fn jobs() -> Vec<Job> {
    (0..12)
        .map(|id| Job {
            id,
            policy: match id % 6 {
                0 | 5 => Policy::Ordinary,
                1 => Policy::Fail(1),
                2 => Policy::Fail(2),
                3 => Policy::Cancel(0),
                _ => Policy::Cancel(1),
            },
        })
        .collect()
}

#[test]
fn heterogeneous_outcomes_match_sequential_under_a_smaller_workspace_budget() {
    let jobs = jobs();
    let token = CancellationToken::new();
    let live = Live::default();
    let sequential: Vec<_> = jobs
        .iter()
        .map(|&j| fit(j, &live, None, &token, false))
        .collect();
    assert_eq!(live.peak.load(Ordering::SeqCst), 1);
    let requested_workers = 4;
    let workspace_budget = 2;
    // A caller can enforce a live-workspace count simply by bounding its pool.
    let workers = requested_workers.min(workspace_budget);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .unwrap();
    let live = Live::default();
    let barrier = Barrier::new(workers);
    let parallel: Vec<_> = pool.install(|| {
        jobs.par_iter()
            .map(|&j| fit(j, &live, Some(&barrier), &token, false))
            .collect()
    });
    assert_eq!(live.peak.load(Ordering::SeqCst), workspace_budget);
    assert_eq!(live.current.load(Ordering::SeqCst), 0);
    assert_eq!(parallel.len(), jobs.len());
    for (a, b) in sequential.iter().zip(&parallel) {
        assert_eq!(a.id, b.id);
        assert_eq!(a.fault, b.fault);
        assert_eq!(a.calls, b.calls);
        assert_eq!(a.result.termination, b.result.termination);
        assert_eq!(a.result.iterations, b.result.iterations);
        assert_eq!(a.result.evaluations, b.result.evaluations);
        assert_eq!(a.result.objective, b.result.objective);
        match (&a.result.candidate, &b.result.candidate) {
            (Some(a), Some(b)) => {
                assert_eq!(a.position, b.position);
                assert_eq!(a.gradient, b.gradient);
                assert_eq!(a.value, b.value);
            }
            (None, None) => {}
            _ => panic!("candidate availability changed"),
        }
        let (termination, calls, candidate) = match jobs[b.id].policy {
            Policy::Ordinary => {
                assert!(matches!(
                    b.result.termination,
                    OptimizationTermination::ScoreTolerance
                        | OptimizationTermination::LineSearchFailed
                ));
                let norm = b
                    .result
                    .candidate
                    .as_ref()
                    .unwrap()
                    .gradient
                    .iter()
                    .fold(0_f64, |a, x| a.max(x.abs()));
                if b.result.termination == OptimizationTermination::ScoreTolerance {
                    assert!(norm <= b.result.options.score_tolerance());
                } else {
                    assert!(b.result.line_search_failure.is_some());
                    assert!(norm > b.result.options.score_tolerance());
                }
                (b.result.termination, b.calls, true)
            }
            Policy::Fail(n) => (OptimizationTermination::EvaluationFailed, n, n > 1),
            Policy::Cancel(n) => (OptimizationTermination::Cancelled, n, n > 0),
        };
        assert_eq!(b.result.termination, termination, "{b:?}");
        assert_eq!(b.calls, calls);
        assert_eq!(b.result.candidate.is_some(), candidate);
        eprintln!(
            "fit={} termination={:?} calls={} fault={:?} candidate={}",
            b.id,
            b.result.termination,
            b.calls,
            b.fault,
            b.result.candidate.is_some()
        );
    }
}

#[test]
fn shared_cancellation_retains_all_ids_and_complete_inflight_evaluations() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    for precancelled in [false, true] {
        let token = CancellationToken::new();
        if precancelled {
            token.cancel();
        }
        let live = Live::default();
        let records: Vec<_> = pool.install(|| {
            (0..12)
                .into_par_iter()
                .map(|id| {
                    fit(
                        Job {
                            id,
                            policy: Policy::Ordinary,
                        },
                        &live,
                        None,
                        &token,
                        true,
                    )
                })
                .collect()
        });
        assert_eq!(records.len(), 12);
        assert!(token.is_cancelled());
        assert!(live.peak.load(Ordering::SeqCst) <= 2);
        assert_eq!(live.current.load(Ordering::SeqCst), 0);
        for (id, record) in records.iter().enumerate() {
            assert_eq!(record.id, id);
            assert_eq!(
                record.result.termination,
                OptimizationTermination::Cancelled
            );
            assert!(record.calls <= 1);
            assert_eq!(record.result.candidate.is_some(), record.calls == 1);
            assert!(record.fault.is_none());
        }
        if precancelled {
            assert!(records.iter().all(|r| r.calls == 0));
        } else {
            assert!(records.iter().any(|r| r.calls == 1));
        }
    }
}

#[test]
fn unwinding_releases_reservations_and_remains_a_panic() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let live = Live::default();
    let token = CancellationToken::new();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.install(|| {
            (0..4).into_par_iter().for_each(|id| {
                if id == 2 {
                    let _lease = live.enter();
                    panic!("injected worker unwind");
                }
                fit(
                    Job {
                        id,
                        policy: Policy::Ordinary,
                    },
                    &live,
                    None,
                    &token,
                    false,
                );
            });
        });
    }));
    assert!(outcome.is_err());
    assert_eq!(live.current.load(Ordering::SeqCst), 0);
    assert!(live.peak.load(Ordering::SeqCst) <= 2);
    let result = pool.install(|| {
        fit(
            Job {
                id: 0,
                policy: Policy::Ordinary,
            },
            &live,
            None,
            &token,
            false,
        )
    });
    assert_eq!(
        result.result.termination,
        OptimizationTermination::ScoreTolerance
    );
    assert_eq!(live.current.load(Ordering::SeqCst), 0);
}
