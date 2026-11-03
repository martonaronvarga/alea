//! Executed downstream build-script/source-generation conformance.
#![feature(autodiff)]
#![forbid(unsafe_code)]

macro_rules! generated {
    ($($name:ident),* $(,)?) => {$(
        pub mod $name {
            include!(concat!(env!("OUT_DIR"), "/", stringify!($name), ".rs"));
            pub mod composition {
                include!(concat!(env!("OUT_DIR"), "/", stringify!($name), "_objective.rs"));
            }
        }
    )*};
}
generated!(
    polynomial,
    native_normal_batch,
    native_logit_batch,
    native_poisson_batch,
    regression,
    likelihood_location_scale,
    uncertainty_base,
    uncertainty_large,
    uncertainty_small,
    logistic,
    interval,
    lower,
    upper,
    ordered,
    positive_ordered,
    simplex,
    unit_vector,
    cholesky_covariance,
    cholesky_correlation,
    covariance,
    correlation,
    covariance3,
    correlation3,
    gaussian16,
    selected_likelihood,
    selected_posterior,
    tensor_matrix,
    tensor_checkpoint_matrix,
    tensor_control,
    tensor_checkpoint,
    tensor_checkpoint_zero,
    tensor_checkpoint_long,
    tensor_retained_long,
    tensor_checkpoint_mixed,
    tensor_checkpoint_activity,
    tensor_empty,
    tensor_checkpoint_empty,
    tensor_zero_scan
);
