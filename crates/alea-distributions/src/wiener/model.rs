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
    #[error("Wiener batch workspace is already borrowed")]
    WorkspaceBusy,
}

/// A Wiener4/5 model with reusable branch-bucket scratch.
///
/// The scratch is chain-local: this model is `Send` when its family is `Send`,
/// and is not `Sync`. Construct one per chain. No allocation occurs during a
/// valid evaluation after construction. Direct evaluation remains the default.
#[derive(Debug)]
pub struct WienerBatchModel<F> {
    model: WienerModel<F, WienerObservations>,
    workspace: std::cell::RefCell<super::WienerBatchWorkspace>,
}

macro_rules! cached_model {
    ($family:ty, $params:ty, $dim:literal) => {
        impl WienerModel<$family, WienerObservations> {
            /// Retain scratch with the model for repeated likelihood evaluations.
            pub fn with_workspace(self) -> WienerBatchModel<$family> {
                let workspace = super::WienerBatchWorkspace::new(self.data.len());
                WienerBatchModel {
                    model: self,
                    workspace: std::cell::RefCell::new(workspace),
                }
            }
        }
        impl ConstrainedModel for WienerBatchModel<$family> {
            type Error = WienerModelError;
            fn dimension(&self) -> usize {
                $dim
            }
            fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
                let q: [f64; $dim] = q.try_into().map_err(|_| WienerModelError::Dimension)?;
                let g: &mut [f64; $dim] = g.try_into().map_err(|_| WienerModelError::Dimension)?;
                let mut workspace = self
                    .workspace
                    .try_borrow_mut()
                    .map_err(|_| WienerModelError::WorkspaceBusy)?;
                let value = self.model.batch_value_gradient_with_workspace(
                    &<$params>::from_array(q),
                    g,
                    &mut workspace,
                );
                if !value.is_finite() || g.iter().any(|x| !x.is_finite()) {
                    return Err(WienerModelError::NonFinite);
                }
                Ok(value)
            }
        }
    };
}
cached_model!(Wiener4, Wiener4Params, 4);
cached_model!(Wiener5, Wiener5Params, 5);

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
