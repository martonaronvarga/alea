#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

#[doc(inline)]
pub use alea_core::{
    capability, density, distribution, model, state_space, target, transform, transport,
};
#[doc(inline)]
pub use alea_math as math;
#[doc(inline)]
pub use alea_runtime::{diagnostics, draws, ensemble, monitor, output, run_chain, session};

/// Sampling algorithms, integrators, and warmup.
///
/// HMC, NUTS, and RWMH are available in ordinary inference sessions. Other
/// algorithms have explicit experimental or approximation requirements in their
/// owning modules; importing a type does not make it eligible for a session.
pub mod sampling {
    #[doc(inline)]
    pub use alea_mcmc::{
        Hmc, HmcOptions, HmcTransition, LogDensityGradient, MarkovChain, Nuts, NutsOptions,
        NutsTransition, Rwmh, RwmhOptions, RwmhTransition, adapt, advanced, approximate,
        atlas_sampler, batch, chees, config, ghmc, hamiltonian, hmc, integrator, mams, nuts,
        prelude, riemannian, rwmh, walnuts,
    };
}

#[cfg(feature = "autodiff")]
#[doc(inline)]
pub use alea_autodiff as autodiff;
#[cfg(feature = "distributions")]
#[doc(inline)]
pub use alea_distributions as distributions;
#[cfg(feature = "distributions")]
#[doc(inline)]
pub use alea_distributions::wiener;

/// Common imports for building models and running inference.
///
/// Optional distribution and derivative types follow the enabled Cargo features.
/// Specialized algorithms remain available through [`sampling`].
pub mod prelude {
    #[doc(inline)]
    pub use alea_core::prelude::*;
    #[cfg(feature = "distributions")]
    #[doc(inline)]
    pub use alea_distributions::prelude::*;
    #[doc(inline)]
    pub use alea_math::prelude::*;
    #[doc(inline)]
    pub use alea_mcmc::prelude::*;
    #[doc(inline)]
    pub use alea_runtime::prelude::*;
}
