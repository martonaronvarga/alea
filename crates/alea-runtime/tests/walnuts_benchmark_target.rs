use alea_core::target::LogDensityGradient;

#[path = "support/walnuts_targets.rs"]
mod walnuts_targets;
use walnuts_targets::{Kind, Target};

#[test]
fn benchmark_target_scores_match_central_differences() {
    for kind in [Kind::Funnel, Kind::StochasticVolatility] {
        let target = Target::new(kind);
        let q: Vec<_> = (0..kind.dimension())
            .map(|index| 0.2 * (0.31 * index as f64).sin() - 0.1)
            .collect();
        let mut score = vec![0.0; q.len()];
        let logp = target.logp_grad(&q, &mut score).unwrap();
        assert!(logp.is_finite());
        for coordinate in 0..q.len() {
            let scale = q[coordinate].abs().max(1.0);
            let step = f64::EPSILON.cbrt() * scale;
            let mut plus = q.clone();
            let mut minus = q.clone();
            plus[coordinate] += step;
            minus[coordinate] -= step;
            let mut scratch = vec![0.0; q.len()];
            let upper = target.logp_grad(&plus, &mut scratch).unwrap();
            let lower = target.logp_grad(&minus, &mut scratch).unwrap();
            let reference = (upper - lower) / (2.0 * step);
            assert!(
                (score[coordinate] - reference).abs() <= 2e-7 * reference.abs().max(1.0),
                "{kind:?} coordinate {coordinate}: analytic={} finite_difference={reference}",
                score[coordinate]
            );
        }
    }
}

#[test]
fn benchmark_observables_are_finite_and_call_counter_is_exact() {
    for kind in [Kind::Funnel, Kind::StochasticVolatility] {
        let target = Target::new(kind);
        assert!(!kind.name().is_empty());
        let q = vec![0.1; kind.dimension()];
        assert!(kind.observables(&q).into_iter().all(f64::is_finite));
        let mut score = vec![0.0; q.len()];
        for expected in 1..=3 {
            target.logp_grad(&q, &mut score).unwrap();
            assert_eq!(target.calls(), expected);
        }
    }
}
