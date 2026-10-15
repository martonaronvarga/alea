//! Bounded latest-only diagnostic worker, independent of Rayon and Tokio executors.
//!
//! Samplers replace an immutable snapshot in an ArcSwap mailbox and try-notify a
//! capacity-one channel. They never wait for diagnostics/subscribers. Tokio watch
//! carries only the latest computed report; subscription methods never expose a
//! guard that could hold the producer's lock across user code or `.await`.
use crate::{
    diagnostics::modern::{self, Error, Summary},
    ensemble::ChainProgress,
    output::{ChainId, TransitionStats},
};
use arc_swap::ArcSwapOption;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, SyncSender},
    },
    thread,
};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

/// All-run retained-transition counts. Missing diagnostics are counted separately.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counts {
    /// Retained transitions, including repeats.
    pub draws: usize,
    /// Actual Hamiltonian divergences.
    pub divergences: usize,
    /// Maximum-depth terminations.
    pub depth_saturations: usize,
    /// Transitions with a Hamiltonian-divergence indicator.
    pub divergence_observations: usize,
    /// Transitions with a tree-depth saturation indicator.
    pub depth_observations: usize,
}

pub(crate) struct Window {
    id: ChainId,
    capacity: usize,
    width: usize,
    data: Vec<f64>,
    energy: Vec<Option<f64>>,
    counts: Counts,
}
impl Window {
    pub(crate) fn new(id: ChainId, capacity: usize, width: usize) -> Self {
        Self {
            id,
            capacity,
            width,
            data: vec![0.0; capacity * width],
            energy: vec![None; capacity],
            counts: Counts::default(),
        }
    }
    pub(crate) fn push(&mut self, q: &[f64], coordinates: &[usize], stats: TransitionStats) {
        let slot = self.counts.draws % self.capacity;
        for (out, &i) in self.data[slot * self.width..(slot + 1) * self.width]
            .iter_mut()
            .zip(coordinates)
        {
            *out = q[i];
        }
        self.energy[slot] = stats.energy;
        self.counts.draws += 1;
        self.counts.divergences += usize::from(stats.divergent == Some(true));
        self.counts.depth_saturations += usize::from(stats.depth_saturated == Some(true));
        self.counts.divergence_observations += usize::from(stats.divergent.is_some());
        self.counts.depth_observations += usize::from(stats.depth_saturated.is_some());
    }
    fn snapshot(&self) -> ChainWindow {
        let len = self.counts.draws.min(self.capacity);
        let start = self.counts.draws - len;
        let mut values = Vec::with_capacity(len * self.width);
        let mut energy = Vec::with_capacity(len);
        for i in start..self.counts.draws {
            let slot = i % self.capacity;
            values.extend_from_slice(&self.data[slot * self.width..(slot + 1) * self.width]);
            energy.push(self.energy[slot]);
        }
        ChainWindow {
            id: self.id,
            start,
            end: self.counts.draws,
            values,
            energy,
            counts: self.counts,
        }
    }
}
struct ChainWindow {
    id: ChainId,
    start: usize,
    end: usize,
    values: Vec<f64>,
    energy: Vec<Option<f64>>,
    counts: Counts,
}

/// Immutable monitoring input; construction is restricted to the validated runner.
pub(crate) struct Snapshot {
    windows: Vec<ChainWindow>,
    progress: Vec<ChainProgress>,
    coordinates: Vec<usize>,
    finished: bool,
    pub(crate) failed: bool,
}
impl Snapshot {
    pub(crate) fn new(
        windows: &[Window],
        progress: &[ChainProgress],
        coordinates: &[usize],
        finished: bool,
    ) -> Self {
        Self {
            windows: windows.iter().map(Window::snapshot).collect(),
            progress: progress.to_vec(),
            coordinates: coordinates.to_vec(),
            finished,
            failed: false,
        }
    }
}

/// Per-chain all-run counts and window-local energy diagnostic.
#[derive(Debug, Clone)]
pub struct ChainSummary {
    /// Logical chain identity.
    pub chain: ChainId,
    /// Half-open retained-draw range used by diagnostics.
    pub range: std::ops::Range<usize>,
    /// Counts across all retained draws, not only the window.
    pub counts: Counts,
    /// E-BFMI for contiguous initial energies in this window; absent if any missing.
    pub ebfmi: Option<Result<f64, Error>>,
}

/// Latest immutable diagnostic result. Old reports may be skipped by subscribers.
#[derive(Debug, Clone)]
pub struct Report {
    /// Coordinate indices in the same order as `diagnostics`.
    pub coordinates: Vec<usize>,
    /// Each observable's result; errors never count as convergence.
    pub diagnostics: Vec<Result<Summary, Error>>,
    /// Per-chain ranges/energy/counts in input order.
    pub chains: Vec<ChainSummary>,
    /// Completed warmup/retained counts.
    pub progress: Vec<ChainProgress>,
    /// No further snapshots will be produced by this run.
    pub finished: bool,
    /// A typed chain/position error interrupted the run.
    pub failed: bool,
    /// This report met the optional policy; not a proof of convergence.
    pub stop_requested: bool,
}

impl Report {
    fn calculate(snapshot: &Snapshot) -> Self {
        let width = snapshot.coordinates.len();
        let first = &snapshot.windows[0];
        let aligned = snapshot
            .windows
            .iter()
            .all(|w| w.start == first.start && w.end == first.end);
        let diagnostics = (0..width)
            .map(|coordinate| {
                if !aligned {
                    return Err(Error::Shape);
                }
                let chains: Vec<Vec<f64>> = snapshot
                    .windows
                    .iter()
                    .map(|w| {
                        w.values
                            .chunks_exact(width)
                            .map(|row| row[coordinate])
                            .collect()
                    })
                    .collect();
                let views: Vec<_> = chains.iter().map(Vec::as_slice).collect();
                modern::summarize(&views)
            })
            .collect();
        let chains = snapshot
            .windows
            .iter()
            .map(|w| {
                let energy: Option<Vec<f64>> = w.energy.iter().copied().collect();
                ChainSummary {
                    chain: w.id,
                    range: w.start..w.end,
                    counts: w.counts,
                    ebfmi: energy.map(|e| modern::ebfmi(&e)),
                }
            })
            .collect();
        Self {
            coordinates: snapshot.coordinates.clone(),
            diagnostics,
            chains,
            progress: snapshot.progress.clone(),
            finished: snapshot.finished,
            failed: snapshot.failed,
            stop_requested: false,
        }
    }
}

/// Explicit opt-in stopping thresholds. Checks every selected observable and
/// requires zero observed divergences/depth saturation over the entire run.
/// Repeated diagnostic checking can bias stopped estimates; use a fixed budget
/// for confirmatory inference. A rolling window cannot certify earlier mixing.
#[derive(Debug, Clone, Copy)]
pub struct StopPolicy {
    /// Minimum draws per chain *in the diagnostic window*, at least four.
    pub minimum_draws: usize,
    /// Maximum rank/folded R-hat, finite and at least one.
    pub maximum_rhat: f64,
    /// Minimum bulk AND tail ESS across all monitored chains, positive finite.
    pub minimum_ess: f64,
    /// Optional maximum absolute mean MCSE for every selected coordinate.
    pub maximum_mcse: Option<f64>,
    /// Optional minimum E-BFMI; missing energies fail this requirement.
    pub minimum_ebfmi: Option<f64>,
}
impl StopPolicy {
    fn valid(self) -> bool {
        self.minimum_draws >= 4
            && self.maximum_rhat.is_finite()
            && self.maximum_rhat >= 1.0
            && self.minimum_ess.is_finite()
            && self.minimum_ess > 0.0
            && [self.maximum_mcse, self.minimum_ebfmi]
                .into_iter()
                .flatten()
                .all(|x| x.is_finite() && x > 0.0)
    }
    /// Evaluates all configured gates, refusing failed, ragged or undefined reports.
    pub fn satisfied(self, report: &Report) -> bool {
        self.valid()
            && !report.failed
            && report.chains.len() >= 2
            && !report.diagnostics.is_empty()
            && report.chains.iter().all(|c| {
                c.range == report.chains[0].range
                    && c.range.end.saturating_sub(c.range.start) >= self.minimum_draws
                    && c.counts.divergences == 0
                    && c.counts.depth_saturations == 0
                    && self
                        .minimum_ebfmi
                        .is_none_or(|limit| matches!(c.ebfmi, Some(Ok(v)) if v >= limit))
            })
            && report.diagnostics.iter().all(|d| match d {
                Ok(d) => {
                    d.rhat.is_finite()
                        && d.rhat <= self.maximum_rhat
                        && d.bulk_ess.is_finite()
                        && d.bulk_ess >= self.minimum_ess
                        && d.tail_ess.is_finite()
                        && d.tail_ess >= self.minimum_ess
                        && self
                            .maximum_mcse
                            .is_none_or(|limit| d.mcse_mean.is_finite() && d.mcse_mean <= limit)
                }
                Err(_) => false,
            })
    }
}

struct Shared {
    latest: ArcSwapOption<Snapshot>,
    closed: AtomicBool,
}

/// One run's nonblocking snapshot mailbox. Pass it to the ensemble runner.
/// Do not share a publisher between simultaneous independent runs.
#[derive(Clone)]
pub struct Publisher {
    shared: Arc<Shared>,
    wake: SyncSender<()>,
}
impl Publisher {
    pub(crate) fn publish(&self, snapshot: Snapshot) {
        if self.shared.closed.load(Ordering::Acquire) {
            return;
        }
        self.shared.latest.store(Some(Arc::new(snapshot)));
        // Full means a notification is already pending; the mailbox still holds
        // the newest snapshot. Disconnected means the monitor has shut down.
        let _ = self.wake.try_send(());
    }
}

/// Latest-only subscription usable from synchronous code or a Tokio task.
/// Cloning creates an independent subscriber, not a queue of stale reports.
#[derive(Clone)]
pub struct Subscription {
    receiver: watch::Receiver<Option<Arc<Report>>>,
}
impl Subscription {
    /// Clones the latest immutable report, releasing the internal lock immediately.
    pub fn latest(&self) -> Option<Arc<Report>> {
        self.receiver.borrow().clone()
    }
    /// Waits for a newer report. Returns `None` after the final report is consumed
    /// and the worker closes. Cancellation of this future loses no notification.
    pub async fn changed(&mut self) -> Option<Arc<Report>> {
        self.receiver.changed().await.ok()?;
        self.receiver.borrow_and_update().clone()
    }
}

/// Lifecycle owner of a dedicated diagnostics thread. `finish` joins explicitly;
/// dropping also shuts down and joins (never leaves detached background work).
/// Shutdown waits for an in-flight calculation, never for a subscriber.
pub struct Monitor {
    publisher: Publisher,
    subscription: Subscription,
    worker: Option<thread::JoinHandle<()>>,
}

/// Monitor configuration/thread creation failure.
#[derive(Debug, thiserror::Error)]
pub enum MonitorError {
    /// Invalid stopping thresholds.
    #[error("invalid stopping thresholds")]
    Policy,
    /// OS worker creation failed.
    #[error("diagnostic thread could not start: {0}")]
    Thread(#[from] std::io::Error),
}
impl Monitor {
    /// Starts a dedicated CPU diagnostics thread, not a Tokio executor task.
    /// `None` disables automatic stopping. The supplied token is cancelled only
    /// when the explicit policy passes; dropping the monitor does not cancel runs.
    /// # Errors
    /// Returns invalid-policy or OS-thread creation errors.
    pub fn start(
        policy: Option<StopPolicy>,
        cancel: CancellationToken,
    ) -> Result<Self, MonitorError> {
        if policy.is_some_and(|p| !p.valid()) {
            return Err(MonitorError::Policy);
        }
        let shared = Arc::new(Shared {
            latest: ArcSwapOption::empty(),
            closed: AtomicBool::new(false),
        });
        let (wake, incoming) = mpsc::sync_channel(1);
        let (reports, receiver) = watch::channel(None);
        let state = Arc::clone(&shared);
        let worker = thread::Builder::new()
            .name("alea-diagnostics".into())
            .spawn(move || {
                let mut previous: Option<Arc<Snapshot>> = None;
                while incoming.recv().is_ok() {
                    loop {
                        if let Some(snapshot) = state.latest.load_full()
                            && !previous.as_ref().is_some_and(|p| Arc::ptr_eq(p, &snapshot))
                        {
                            let mut report = Report::calculate(&snapshot);
                            report.stop_requested = policy.is_some_and(|p| p.satisfied(&report));
                            if report.stop_requested {
                                cancel.cancel();
                            }
                            reports.send_replace(Some(Arc::new(report)));
                            previous = Some(snapshot);
                        }
                        if !state.closed.load(Ordering::Acquire) {
                            break;
                        }
                        // Shutdown may race an in-flight calculation. Drain the newest
                        // mailbox value before closing; never strand the final report.
                        let latest = state.latest.load_full();
                        if latest
                            .as_ref()
                            .is_none_or(|s| previous.as_ref().is_some_and(|p| Arc::ptr_eq(p, s)))
                        {
                            return;
                        }
                    }
                }
            })?;
        Ok(Self {
            publisher: Publisher { shared, wake },
            subscription: Subscription { receiver },
            worker: Some(worker),
        })
    }
    /// A cheap clone of the sampler mailbox.
    pub fn publisher(&self) -> Publisher {
        self.publisher.clone()
    }
    /// A cheap independent latest-report subscription.
    pub fn subscribe(&self) -> Subscription {
        self.subscription.clone()
    }
    fn close(&self) {
        self.publisher.shared.closed.store(true, Ordering::Release);
        let _ = self.publisher.wake.try_send(());
    }
    /// Processes the last published snapshot and joins. Stop publishing first.
    /// Returns the original panic payload if the diagnostics worker panicked.
    pub fn finish(mut self) -> thread::Result<()> {
        self.close();
        self.worker
            .take()
            .expect("monitor owns its worker until finish")
            .join()
    }
}
impl Drop for Monitor {
    fn drop(&mut self) {
        self.close();
        if let Some(worker) = self.worker.take() {
            // Destructors cannot return a panic payload; explicit finish exposes it.
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_is_bounded_contiguous_and_does_not_bridge_missing_energy() {
        let mut window = Window::new(ChainId(0), 4, 1);
        for i in 0..17 {
            window.push(
                &[i as f64],
                &[0],
                TransitionStats {
                    energy: (i != 15).then_some(i as f64),
                    divergent: Some(i == 3),
                    depth_saturated: Some(i == 6),
                    ..TransitionStats::default()
                },
            );
        }
        assert_eq!(window.data.len(), 4);
        let snapshot = Snapshot::new(&[window], &[], &[0], true);
        assert_eq!(snapshot.windows[0].values, [13.0, 14.0, 15.0, 16.0]);
        let report = Report::calculate(&snapshot);
        assert!(report.chains[0].ebfmi.is_none());
        assert_eq!(report.chains[0].range, 13..17);
        assert_eq!(report.chains[0].counts.divergences, 1);
        assert_eq!(report.chains[0].counts.depth_saturations, 1);
        assert_eq!(report.chains[0].counts.divergence_observations, 17);
        assert!(report.diagnostics[0].is_err()); // one chain cannot establish R-hat
    }

    #[test]
    fn final_snapshot_is_never_lost_during_worker_shutdown() {
        for _ in 0..40 {
            let monitor = Monitor::start(None, CancellationToken::new()).unwrap();
            let publisher = monitor.publisher();
            let subscription = monitor.subscribe();
            let mut windows = [
                Window::new(ChainId(0), 32, 1),
                Window::new(ChainId(1), 32, 1),
            ];
            for i in 0..40 {
                for (j, w) in windows.iter_mut().enumerate() {
                    w.push(
                        &[((i * 7 + j * 3) % 31) as f64],
                        &[0],
                        TransitionStats::default(),
                    );
                }
                publisher.publish(Snapshot::new(&windows, &[], &[0], i == 39));
            }
            monitor.finish().unwrap();
            let report = subscription.latest().unwrap();
            assert!(report.finished);
            assert!(report.chains.iter().all(|c| c.range == (8..40)));
        }
    }

    #[test]
    #[cfg(not(miri))]
    fn updating_window_has_no_per_draw_allocations() {
        let mut window = Window::new(ChainId(0), 64, 2);
        let allocated = allocation_counter::measure(|| {
            for i in 0..10_000 {
                window.push(
                    &[i as f64, -(i as f64)],
                    &[1, 0],
                    TransitionStats::default(),
                );
            }
        });
        assert_eq!(allocated.count_total, 0);
    }
}
