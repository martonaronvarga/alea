use crate::core::state_space::ObservationModel;
use crate::dist::wiener::{Boundary, WienerFpt, WienerParams};

#[derive(Debug, Clone, Copy)]
pub struct DdmObservation {
    pub rt: f64,
    pub boundary: Boundary,
}

#[derive(Debug, Clone, Copy)]
pub struct DdmState<Context, Latent> {
    pub context: Context,
    pub latent: Latent,
}

pub trait ParameterMap<Context, Latent> {
    fn write_params(&self, context: &Context, latent: &Latent, out: &mut WienerParams);
}

#[derive(Debug, Clone, Copy)]
pub struct DdmObservationModel<M> {
    pub map: M,
    pub density: WienerFpt,
}

impl<M> DdmObservationModel<M> {
    #[inline]
    pub fn new(map: M, density: WienerFpt) -> Self {
        Self { map, density }
    }
}

impl<Context, Latent, M> ObservationModel<DdmState<Context, Latent>, DdmObservation>
    for DdmObservationModel<M>
where
    M: ParameterMap<Context, Latent>,
{
    #[inline]
    fn log_likelihood(&self, state: &DdmState<Context, Latent>, obs: &DdmObservation) -> f64 {
        let mut params = WienerParams {
            alpha: 1.0,
            tau: 0.0,
            beta: 0.5,
            delta: 0.0,
        };
        self.map
            .write_params(&state.context, &state.latent, &mut params);
        self.density.log_pdf(obs.rt, params, obs.boundary)
    }
}
