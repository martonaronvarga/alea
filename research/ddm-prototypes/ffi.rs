use super::wiener::{Boundary, WienerParams};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CBoundary(pub i32);

impl From<Boundary> for CBoundary {
    #[inline]
    fn from(value: Boundary) -> Self {
        Self(match value {
            Boundary::Upper => 0,
            Boundary::Lower => 1,
        })
    }
}

impl From<CBoundary> for Boundary {
    #[inline]
    fn from(value: CBoundary) -> Self {
        match value.0 {
            0 => Boundary::Upper,
            _ => Boundary::Lower,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CWienerParams {
    pub alpha: f64,
    pub tau: f64,
    pub beta: f64,
    pub delta: f64,
}

impl From<WienerParams> for CWienerParams {
    #[inline]
    fn from(value: WienerParams) -> Self {
        Self {
            alpha: value.alpha,
            tau: value.tau,
            beta: value.beta,
            delta: value.delta,
        }
    }
}

impl From<CWienerParams> for WienerParams {
    #[inline]
    fn from(value: CWienerParams) -> Self {
        Self {
            alpha: value.alpha,
            tau: value.tau,
            beta: value.beta,
            delta: value.delta,
        }
    }
}
