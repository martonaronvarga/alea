//! Dimension-checked constrained Wiener models; transforms define the sampling coordinates.
use super::{
    Wiener4, Wiener4Params, Wiener5, Wiener5Params, Wiener7, Wiener7Params, WienerObservation,
    WienerObservations,
};
use alea_core::model::ConstrainedModel;
#[derive(Debug)]
pub struct WienerModel<F, D> {
    pub(super) family: F,
    pub(super) data: D,
}
impl<F, D> WienerModel<F, D> {
    pub fn new(family: F, data: D) -> Self {
        Self { family, data }
    }
    pub fn family(&self) -> &F {
        &self.family
    }
    pub fn data(&self) -> &D {
        &self.data
    }
}
#[derive(Debug, Clone, Copy, thiserror::Error)]
pub enum WienerModelError {
    #[error("Wiener model dimension mismatch")]
    Dimension,
    #[error("Wiener value or gradient is non-finite")]
    NonFinite,
}

impl ConstrainedModel for WienerModel<Wiener4, Vec<WienerObservation>> {
    type Error = WienerModelError;
    fn dimension(&self) -> usize {
        4
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        let q: [f64; 4] = q.try_into().map_err(|_| WienerModelError::Dimension)?;
        let g: &mut [f64; 4] = g.try_into().map_err(|_| WienerModelError::Dimension)?;
        let value = self.batch_value_gradient(&Wiener4Params::from_array(q), g);
        if !value.is_finite() || g.iter().any(|x| !x.is_finite()) {
            return Err(WienerModelError::NonFinite);
        }
        Ok(value)
    }
}

impl ConstrainedModel for WienerModel<Wiener4, WienerObservations> {
    type Error = WienerModelError;
    fn dimension(&self) -> usize {
        4
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        let q: [f64; 4] = q.try_into().map_err(|_| WienerModelError::Dimension)?;
        let g: &mut [f64; 4] = g.try_into().map_err(|_| WienerModelError::Dimension)?;
        let value = self.batch_value_gradient(&Wiener4Params::from_array(q), g);
        if !value.is_finite() || g.iter().any(|x| !x.is_finite()) {
            return Err(WienerModelError::NonFinite);
        }
        Ok(value)
    }
}

impl ConstrainedModel for WienerModel<Wiener5, Vec<WienerObservation>> {
    type Error = WienerModelError;
    fn dimension(&self) -> usize {
        5
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        let q: [f64; 5] = q.try_into().map_err(|_| WienerModelError::Dimension)?;
        let g: &mut [f64; 5] = g.try_into().map_err(|_| WienerModelError::Dimension)?;
        let value = self.batch_value_gradient(&Wiener5Params::from_array(q), g);
        if !value.is_finite() || g.iter().any(|x| !x.is_finite()) {
            return Err(WienerModelError::NonFinite);
        }
        Ok(value)
    }
}

impl ConstrainedModel for WienerModel<Wiener5, WienerObservations> {
    type Error = WienerModelError;
    fn dimension(&self) -> usize {
        5
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        let q: [f64; 5] = q.try_into().map_err(|_| WienerModelError::Dimension)?;
        let g: &mut [f64; 5] = g.try_into().map_err(|_| WienerModelError::Dimension)?;
        let value = self.batch_value_gradient(&Wiener5Params::from_array(q), g);
        if !value.is_finite() || g.iter().any(|x| !x.is_finite()) {
            return Err(WienerModelError::NonFinite);
        }
        Ok(value)
    }
}

impl ConstrainedModel for WienerModel<Wiener7, Vec<WienerObservation>> {
    type Error = WienerModelError;
    fn dimension(&self) -> usize {
        7
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        let q: [f64; 7] = q.try_into().map_err(|_| WienerModelError::Dimension)?;
        let g: &mut [f64; 7] = g.try_into().map_err(|_| WienerModelError::Dimension)?;
        let value = self.batch_value_gradient(&Wiener7Params::from_array(q), g);
        if !value.is_finite() || g.iter().any(|x| !x.is_finite()) {
            return Err(WienerModelError::NonFinite);
        }
        Ok(value)
    }
}

impl ConstrainedModel for WienerModel<Wiener7, WienerObservations> {
    type Error = WienerModelError;
    fn dimension(&self) -> usize {
        7
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        let q: [f64; 7] = q.try_into().map_err(|_| WienerModelError::Dimension)?;
        let g: &mut [f64; 7] = g.try_into().map_err(|_| WienerModelError::Dimension)?;
        let value = self.batch_value_gradient(&Wiener7Params::from_array(q), g);
        if !value.is_finite() || g.iter().any(|x| !x.is_finite()) {
            return Err(WienerModelError::NonFinite);
        }
        Ok(value)
    }
}
