//! Executed downstream build-script/source-generation conformance.
#![feature(autodiff)]
#![forbid(unsafe_code)]

macro_rules! generated {
    ($($name:ident),* $(,)?) => {$(
        pub mod $name {
            include!(concat!(env!("OUT_DIR"), "/", stringify!($name), ".rs"));
        }
    )*};
}
generated!(
    polynomial,
    regression,
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
    gaussian16
);
