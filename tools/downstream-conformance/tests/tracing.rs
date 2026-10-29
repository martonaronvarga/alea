#![cfg(feature = "tracing")]
use std::{
    fmt,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use tracing::{
    Event, Subscriber,
    field::{Field, Visit},
};
use tracing_subscriber::{Layer, Registry, prelude::*};

#[derive(Clone)]
struct Estimated(Arc<AtomicBool>);
impl Visit for Estimated {
    fn record_debug(&mut self, _: &Field, _: &dyn fmt::Debug) {}
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "error_contract" && value == "estimated" {
            self.0.store(true, Ordering::Relaxed);
        }
    }
}
impl<S: Subscriber> Layer<S> for Estimated {
    fn on_event(&self, event: &Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
        event.record(&mut self.clone());
    }
}
#[test]
fn facade_soft_forwarding_traces_directional_normalization_with_its_error_label() {
    let observed = Arc::new(AtomicBool::new(false));
    let subscriber = Registry::default().with(Estimated(observed.clone()));
    tracing::subscriber::with_default(subscriber, || {
        let b = alea::distributions::directional::Bingham::new(vec![-3., -1., 0.]).unwrap();
        assert!(b.log_normalizer().is_finite());
    });
    assert!(observed.load(Ordering::Relaxed));
}
