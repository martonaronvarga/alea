//! A separate test process permits one application-owned global subscriber.
#![cfg(feature = "tracing")]
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tracing::{Event, Subscriber};
use tracing_subscriber::{Layer, Registry, prelude::*};
struct Count(Arc<AtomicUsize>);
impl<S: Subscriber> Layer<S> for Count {
    fn on_event(&self, _: &Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}
#[test]
fn a_global_application_subscriber_receives_monitor_worker_events() {
    let count = Arc::new(AtomicUsize::new(0));
    tracing::subscriber::set_global_default(Registry::default().with(Count(count.clone())))
        .unwrap();
    alea_runtime::monitor::Monitor::start(None, tokio_util::sync::CancellationToken::new())
        .unwrap()
        .finish()
        .unwrap();
    assert_eq!(count.load(Ordering::Relaxed), 1);
}
