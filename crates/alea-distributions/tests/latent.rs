//! Preserve the previously compiled latent-model adapters, not only RNG moments.
use alea_core::state_space::{ObservationModel, TransitionModel};
use alea_distributions::{
    latent::*,
    wiener::{Boundary, Wiener4, Wiener4Params, WienerObservation},
};
use rand::{SeedableRng, rngs::SmallRng};

fn projection() -> AffineLatentWienerMap {
    AffineLatentWienerMap::new(
        [0.2, 0.1, -0.05],
        [-2.0, 0.02, 0.01],
        [0.0, 0.3, -0.2],
        [0.5, -0.1, 0.2],
    )
}

#[test]
fn affine_scalar_and_soa_paths_match_and_preserve_unused_output_tails() {
    let map = projection();
    let latent = LatentFrameSoA {
        c: &[-1.0, 0.0, 2.0],
        m: &[0.5, -1.0, 1.0],
    };
    let mut alpha = [99.0; 4];
    let mut tau = alpha;
    let mut beta = alpha;
    let mut delta = alpha;
    assert!(map.map_into(latent, &mut alpha, &mut tau, &mut beta, &mut delta));
    for i in 0..3 {
        let frame = LatentFrame {
            c: latent.c[i],
            m: latent.m[i],
        };
        let p = map.map_frame(frame);
        for (a, b) in [alpha[i], tau[i], beta[i], delta[i]]
            .into_iter()
            .zip([p.alpha, p.tau, p.beta, p.delta])
        {
            assert!((a - b).abs() < 1e-12);
        }
        assert!((p.alpha - (0.2 + 0.1 * frame.c - 0.05 * frame.m).exp()).abs() < 1e-14);
        assert!((p.beta - 1.0 / (1.0 + (-0.3 * frame.c + 0.2 * frame.m).exp())).abs() < 1e-14);
    }
    assert_eq!([alpha[3], tau[3], beta[3], delta[3]], [99.0; 4]);
    let before = [alpha, tau, beta, delta];
    // This existing entry point is serial, despite its name. Preserve outputs,
    // not a nonexistent parallel implementation or performance promise.
    assert!(map.map_into_parallel(latent, &mut alpha, &mut tau, &mut beta, &mut delta));
    for (a, b) in [alpha, tau, beta, delta]
        .iter()
        .flatten()
        .zip(before.iter().flatten())
    {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn affine_batch_shape_failures_do_not_partially_write_outputs() {
    let map = projection();
    for invalid in [
        LatentFrameSoA { c: &[1.0], m: &[] },
        LatentFrameSoA {
            c: &[1.0, 2.0],
            m: &[3.0, 4.0],
        },
    ] {
        let (mut a, mut t, mut b, mut d) = ([99.0; 2], [99.0; 2], [99.0; 1], [99.0; 2]);
        assert!(!map.map_into(invalid, &mut a, &mut t, &mut b, &mut d));
        assert_eq!((a, t, b, d), ([99.0; 2], [99.0; 2], [99.0; 1], [99.0; 2]));
    }
    let empty = LatentFrameSoA { c: &[], m: &[] };
    assert!(empty.is_empty());
    assert!(map.map_into(empty, &mut [], &mut [], &mut [], &mut []));
}

struct Project;
impl LatentParameterMap<f64, LatentFrame> for Project {
    fn write_params(&self, context: &f64, latent: &LatentFrame, out: &mut Wiener4Params) {
        *out = projection().map_frame(*latent);
        out.delta += context;
    }
}

#[test]
fn observation_layers_match_direct_wiener_evaluation_on_both_boundaries() {
    let frame = LatentFrame { c: 0.5, m: -0.25 };
    let model = LatentObservationModel::new(projection(), Wiener4);
    let state = LatentState::new(0.0, frame);
    assert_eq!(*state.context(), 0.0);
    assert_eq!(*state.latent(), frame);
    assert_eq!(state.into_parts(), (0.0, frame));
    let layer = ObservationLayer::new(Project, Wiener4);
    let params = projection().map_frame(frame);
    for boundary in [Boundary::Lower, Boundary::Upper] {
        for rt in [0.05, 0.6, 1.2] {
            let obs = WienerObservation { rt, boundary };
            let expected = Wiener4.log_prob(&obs, &params, 1e-12).log_prob;
            let a = model.log_likelihood(&frame, &obs);
            let b = layer.log_likelihood(&state, &obs);
            if rt < params.tau {
                assert_eq!((a, b), (f64::NEG_INFINITY, f64::NEG_INFINITY));
            } else {
                assert!((a - expected).abs() < 1e-12);
                assert!((b - expected).abs() < 1e-12);
            }
        }
    }
}

#[derive(Clone)]
struct Increment;
impl TransitionModel<LatentState<u64, f64>> for Increment {
    fn log_transition(
        &self,
        prev: &LatentState<u64, f64>,
        next: &LatentState<u64, f64>,
        t: usize,
    ) -> f64 {
        next.latent - prev.latent + t as f64
    }
    fn sample_next<R: rand::Rng + ?Sized>(
        &self,
        prev: &LatentState<u64, f64>,
        _: &mut R,
    ) -> LatentState<u64, f64> {
        LatentState::new(prev.context, prev.latent + 1.0)
    }
}

#[test]
fn transition_layer_preserves_context_time_and_rng_contract() {
    let model = TransitionLayer::new(Increment);
    let previous = LatentState::new(17, 0.5);
    let mut rng = SmallRng::seed_from_u64(812);
    let before = rng.clone();
    let next = model.sample_next(&previous, &mut rng);
    assert_eq!(next.into_parts(), (17, 1.5));
    assert_eq!(model.log_transition(&previous, &next, 9), 10.0);
    assert_eq!(rng, before);
    let walk = LatentRandomWalk::new(2.0, 0.5).unwrap();
    let a = LatentFrame { c: 1.0, m: -2.0 };
    let b = LatentFrame { c: 3.0, m: -1.5 };
    let expected = -1.0 - (2.0 * std::f64::consts::PI).ln();
    assert!((walk.log_transition(&a, &b, 0) - expected).abs() < 1e-12);
}
