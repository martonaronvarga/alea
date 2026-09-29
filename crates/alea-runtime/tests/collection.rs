use alea_distributions::Gaussian;
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{Rwmh, RwmhOptions};
use alea_runtime::{
    diagnostics::{acceptance_rate, autocorrelation_ess, classical_split_rhat},
    draws::{DrawError, collect_draws},
};
use rand::{SeedableRng, rngs::SmallRng};

#[test]
fn collection_checks_size_before_rng_and_keeps_rejected_draws() {
    let target = Gaussian::new(2);
    let mut chain = Rwmh::new(
        &target,
        OwnedBuffer::new(2),
        IdentityMetric::new(2),
        RwmhOptions::new(f64::MAX).unwrap(),
    )
    .unwrap();
    let mut rng = SmallRng::seed_from_u64(93);
    let before = rng.clone();
    assert!(matches!(
        collect_draws(&mut chain, usize::MAX, &mut rng),
        Err(DrawError::Size)
    ));
    assert_eq!(rng, before);
    assert_eq!(chain.point().position(), &[0.0, 0.0]);
    let empty = collect_draws(&mut chain, 0, &mut rng).unwrap();
    assert!(empty.is_empty());
    assert_eq!(rng, before);
    let draws = collect_draws(&mut chain, 8, &mut rng).unwrap();
    assert_eq!((draws.len(), draws.dimension()), (8, 2));
    for i in 0..8 {
        assert_eq!(draws.row(i), Some(&[0.0, 0.0][..]));
    }
    assert_eq!(draws.row(8), None);
    assert_eq!(draws.as_slice().len(), 16);
    assert_eq!(draws.as_slice().as_ptr() as usize % 64, 0);
}

#[test]
fn classical_diagnostics_preserve_calculations_but_reject_undefined_inputs() {
    assert_eq!(acceptance_rate(0, 0), 0.0);
    assert_eq!(acceptance_rate(3, 4), 0.75);
    assert_eq!(autocorrelation_ess(&[0.0, 1.0, 0.0, 1.0]).unwrap(), 4.0);
    let a = [0.0, 1.0, 2.0, 3.0];
    // Four split means [0.5,2.5,0.5,2.5], W=0.5, B=8/3.
    assert!((classical_split_rhat(&[&a, &a]).unwrap() - (19.0_f64 / 6.0).sqrt()).abs() < 1e-12);
    for values in [&[][..], &[0.0; 4], &[1.0, 2.0, f64::NAN, 4.0]] {
        assert!(autocorrelation_ess(values).is_err());
        assert!(classical_split_rhat(&[values, values]).is_err());
    }
    assert!(classical_split_rhat(&[&a, &[1.0; 5]]).is_err());
}
