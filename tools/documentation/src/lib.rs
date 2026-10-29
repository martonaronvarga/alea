//! Executable examples from the Alea user guide.
#[doc = include_str!("../../../docs/getting-started.md")]
pub mod getting_started {}
#[doc = include_str!("../../../docs/modeling.md")]
pub mod modeling {}
#[doc = include_str!("../../../docs/sampling.md")]
pub mod sampling {}
#[doc = include_str!("../../../docs/diagnostics.md")]
pub mod diagnostics {}
#[doc = include_str!("../../../docs/sessions.md")]
pub mod sessions {}
#[doc = include_str!("../../../docs/tracing.md")]
pub mod tracing {}
#[doc = include_str!("../../../docs/distributions.md")]
pub mod distributions {}
#[doc = include_str!("../../../docs/wiener.md")]
pub mod wiener {}

#[doc = include_str!("../../../docs/installation.md")]
pub mod installation {}

#[doc = include_str!("../../../crates/alea/README.md")]
pub mod alea {}

#[doc = include_str!("../../../crates/alea-core/README.md")]
pub mod alea_core {}

#[doc = include_str!("../../../crates/alea-math/README.md")]
pub mod alea_math {}

#[doc = include_str!("../../../crates/alea-mcmc/README.md")]
pub mod alea_mcmc {}

#[doc = include_str!("../../../crates/alea-runtime/README.md")]
pub mod alea_runtime {}

#[doc = include_str!("../../../crates/alea-distributions/README.md")]
pub mod alea_distributions {}

#[doc = include_str!("../../../docs/metric-contract.md")]
pub mod metric_contract {}

#[doc = include_str!("../../../docs/hierarchical-models.md")]
pub mod hierarchical_models {}

/// The complete public hierarchical logistic program.
#[doc = "```rust"]
#[doc = include_str!("../../../crates/alea/examples/hierarchical_logistic.rs")]
#[doc = "```"]
pub mod hierarchical_logistic {}

#[doc = include_str!("../../../docs/directional-distributions.md")]
pub mod directional_distributions {}
