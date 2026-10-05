#![cfg(feature = "faer")]

#[test]
fn fitting_does_not_access_process_global_parallelism() {
    // This integration-test binary contains only this test. Disabling global
    // access would panic in any high-level Faer operation using that policy.
    faer::disable_global_parallelism();
    let metric = alea_math::fisher::fit_low_rank(
        &[-2.0, 0.0, 2.0, 0.0, 0.0, -1.0, 0.0, 1.0],
        &[0.5, 0.0, -0.5, 0.0, 0.0, 1.0, 0.0, -1.0],
        &[1.0, 1.0],
        1e-5,
        2.0,
        2,
    )
    .unwrap();
    assert_eq!(metric.rank(), 1);
    // Fitting must not silently re-enable or replace the caller's policy either.
    assert!(std::panic::catch_unwind(faer::get_global_parallelism).is_err());
}
