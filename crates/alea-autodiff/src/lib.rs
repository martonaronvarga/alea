//! Derivative providers and finite-difference validation, isolated from sampler code.
#![forbid(unsafe_code)]
pub mod autodiff;
pub mod codegen;
pub mod gradient_check;
