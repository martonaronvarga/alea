#![cfg_attr(
    feature = "faer",
    doc = r#"
The faer first-order target is available, but carries no curvature capability.

```
use alea::target::LogDensityGradient;
let target = alea_enzyme_conformance::provider_faer_rect::target();
target.logp_grad(&[1.; 9], &mut [0.; 9]).unwrap();
```

```compile_fail
use alea::capability::PotentialHessianVector;
fn requires<T: PotentialHessianVector>(_: T) {}
requires(alea_enzyme_conformance::provider_faer_rect::target());
```

```compile_fail
use alea::capability::PotentialHessianDerivative;
fn requires<T: PotentialHessianDerivative>(_: T) {}
requires(alea_enzyme_conformance::provider_faer_rect::target());
```
"#
)]
#![cfg_attr(
    feature = "openblas",
    doc = r#"
The blas first-order target is available, but carries no curvature capability.

```
use alea::target::LogDensityGradient;
let target = alea_enzyme_conformance::provider_blas_rect::target();
target.logp_grad(&[1.; 9], &mut [0.; 9]).unwrap();
```

```compile_fail
use alea::capability::PotentialHessianVector;
fn requires<T: PotentialHessianVector>(_: T) {}
requires(alea_enzyme_conformance::provider_blas_rect::target());
```

```compile_fail
use alea::capability::PotentialHessianDerivative;
fn requires<T: PotentialHessianDerivative>(_: T) {}
requires(alea_enzyme_conformance::provider_blas_rect::target());
```
"#
)]
//! Executed downstream build-script/source-generation conformance.
#![feature(autodiff)]
#![forbid(unsafe_code)]

macro_rules! generated {
    (first: $($name:ident),* $(,)?) => {$(
        pub mod $name {
            include!(concat!(env!("OUT_DIR"), "/", stringify!($name), "_first.rs"));
            pub mod value { include!(concat!(env!("OUT_DIR"), "/", stringify!($name), "_value.rs")); }
            pub mod composition { include!(concat!(env!("OUT_DIR"), "/", stringify!($name), "_first_objective.rs")); }
        }
    )*};
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

#[cfg(feature = "faer")]
generated!(first: provider_faer_rect,provider_faer_alias,provider_faer_empty,provider_faer_zero,provider_faer_inverse,provider_faer_mixed,provider_faer_inactive,provider_faer_gemm,provider_faer_branch,provider_faer_scan,provider_faer_checkpoint,provider_faer_empty_inner,provider_faer_empty_columns);
#[cfg(feature = "openblas")]
generated!(first: provider_blas_rect,provider_blas_alias,provider_blas_empty,provider_blas_zero,provider_blas_inverse,provider_blas_mixed,provider_blas_inactive,provider_blas_gemm,provider_blas_branch,provider_blas_scan,provider_blas_checkpoint,provider_blas_empty_inner,provider_blas_empty_columns);
