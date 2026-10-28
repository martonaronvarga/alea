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
    // Reusing the same caller-owned buffers gives the same mapped values.
    assert!(map.map_into(latent, &mut alpha, &mut tau, &mut beta, &mut delta));
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
impl WienerParameterMap<f64, LatentFrame> for Project {
    fn write_params(
        &self,
        context: &f64,
        latent: &LatentFrame,
        _t: usize,
        out: &mut Wiener4Params,
    ) -> alea_distributions::wiener::Result<()> {
        *out = projection().map_frame(*latent);
        out.delta += context;
        Ok(())
    }
}

#[test]
fn observation_layers_match_direct_wiener_evaluation_on_both_boundaries() {
    let frame = LatentFrame { c: 0.5, m: -0.25 };
    let model = LatentWienerObservation::new(projection(), Wiener4);
    let state = LatentState::new(0.0, frame);
    assert_eq!(*state.context(), 0.0);
    assert_eq!(*state.latent(), frame);
    assert_eq!(state.into_parts(), (0.0, frame));
    let layer = WienerObservationModel::new(Project, Wiener4);
    let params = projection().map_frame(frame);
    for boundary in [Boundary::Lower, Boundary::Upper] {
        for rt in [0.05, 0.6, 1.2] {
            let obs = WienerObservation { rt, boundary };
            let expected = Wiener4.log_prob(&obs, &params, 1e-12).log_prob;
            let a = model.log_likelihood(&frame, &obs, 3).unwrap();
            let b = layer.log_likelihood(&state, &obs, 3).unwrap();
            if rt < params.tau {
                assert_eq!((a, b), (f64::NEG_INFINITY, f64::NEG_INFINITY));
            } else {
                assert!((a - expected).abs() < 1e-12);
                assert!((b - expected).abs() < 1e-12);
            }
        }
    }
}

struct Increment;
impl TransitionModel<LatentState<u64, f64>> for Increment {
    type Error = std::convert::Infallible;
    fn log_transition(
        &self,
        prev: &LatentState<u64, f64>,
        next: &LatentState<u64, f64>,
        t: usize,
    ) -> Result<f64, Self::Error> {
        Ok(if next.latent == prev.latent + t as f64 {
            0.
        } else {
            f64::NEG_INFINITY
        })
    }
    fn sample_next<R: rand::Rng + ?Sized>(
        &self,
        prev: &LatentState<u64, f64>,
        t: usize,
        _: &mut R,
    ) -> Result<LatentState<u64, f64>, Self::Error> {
        Ok(LatentState::new(prev.context, prev.latent + t as f64))
    }
}
#[test]
fn transition_conditions_density_and_sampling_on_the_same_time() {
    let previous = LatentState::new(17, 0.5);
    let mut rng = SmallRng::seed_from_u64(812);
    let before = rng.clone();
    for time in [0, 1, 9] {
        let next = Increment.sample_next(&previous, time, &mut rng).unwrap();
        assert_eq!(next.into_parts(), (17, 0.5 + time as f64));
        assert_eq!(
            Increment.log_transition(&previous, &next, time).unwrap(),
            0.
        );
    }
    assert_eq!(rng, before);
    let walk = LatentRandomWalk::new(2., 0.5).unwrap();
    let a = LatentFrame { c: 1., m: -2. };
    let b = LatentFrame { c: 3., m: -1.5 };
    let expected = -1. - (2. * std::f64::consts::PI).ln();
    assert!((walk.log_transition(&a, &b, 0).unwrap() - expected).abs() < 1e-12);
    let invalid = LatentFrame { c: f64::NAN, ..a };
    assert!(walk.sample_next(&invalid, 0, &mut rng).is_err());
    assert!(walk.log_transition(&invalid, &b, 0).is_err());
}
