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
            pub mod hvp {
                include!(concat!(env!("OUT_DIR"), "/", stringify!($name), "_hvp.rs"));
                pub mod composition {
                    include!(concat!(env!("OUT_DIR"), "/", stringify!($name), "_hvp_objective.rs"));
                }
            }
            pub mod first {
                include!(concat!(env!("OUT_DIR"), "/", stringify!($name), "_first.rs"));
                pub mod composition {
                    include!(concat!(env!("OUT_DIR"), "/", stringify!($name), "_first_objective.rs"));
                }
            }
            pub mod value {
                include!(concat!(env!("OUT_DIR"), "/", stringify!($name), "_value.rs"));
                pub mod composition {
                    include!(concat!(env!("OUT_DIR"), "/", stringify!($name), "_value_objective.rs"));
                }
            }
        }
    )*};
}
generated!(
    polynomial,
    route_alone,
    tensor_score_admission,
    route_trusted,
    route_checkpoint,
    route_opaque,
    route_trusted_opaque,
    route_selected_trusted,
    route_selected_checkpoint,
    route_selected_opaque,
    route_selected_missing,
    native_normal_batch,
    native_logit_batch,
    native_poisson_batch,
    native_normal_large,
    native_logit_large,
    native_poisson_large,
    normal_vector,
    normal_vector_observed,
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
    tensor_broadcast,
    tensor_checkpoint_broadcast,
    tensor_broadcast_tail,
    tensor_checkpoint_broadcast_tail,
    tensor_broadcast_empty,
    tensor_checkpoint_broadcast_empty,
    tensor_checkpoint_matrix,
    tensor_control,
    tensor_checkpoint,
    tensor_checkpoint_zero,
    tensor_checkpoint_long,
    tensor_retained_long,
    tensor_checkpoint_mixed,
    tensor_checkpoint_activity,
    tensor_checkpoint_immutable,
    tensor_checkpoint_constant,
    tensor_checkpoint_routes,
    tensor_checkpoint_vector,
    tensor_empty,
    tensor_checkpoint_empty,
    tensor_zero_scan
);
