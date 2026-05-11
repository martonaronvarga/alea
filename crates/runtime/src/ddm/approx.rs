use kernels::buffer::OwnedBuffer;
use kernels::dist::wiener::WienerObservation;
use kernels::dist::wiener::{Boundary, Wiener4Params};
// Approximation layer for likelihood amortization.
//
// This stays separate from the exact distribution so that grids,
// splines, or neural surrogates remain optional and orthogonal to the
// statistical semantics.

#[derive(Debug, Clone, Copy)]
pub struct GridSpec {
    pub min: f64,
    pub max: f64,
    pub points: usize,
}

impl GridSpec {
    #[inline]
    pub fn new(min: f64, max: f64, points: usize) -> Option<Self> {
        if !(min.is_finite() && max.is_finite() && min < max && points >= 2) {
            return None;
        }
        Some(Self { min, max, points })
    }

    #[inline]
    pub fn step(&self) -> f64 {
        (self.max - self.min) / ((self.points - 1) as f64)
    }
}

/// Scalar linear spline, useful as a placeholder for precomputed likelihood
/// tables or cheap interpolation.
#[derive(Debug)]
pub struct LinearSpline1D {
    pub grid: Vec<f64>,
    pub values: Vec<f64>,
}

impl LinearSpline1D {
    #[inline]
    pub fn new(grid: Vec<f64>, values: Vec<f64>) -> Option<Self> {
        if grid.len() != values.len() || grid.len() < 2 {
            return None;
        }
        if !grid.windows(2).all(|w| w[0] < w[1]) {
            return None;
        }
        Some(Self { grid, values })
    }

    #[inline]
    pub fn eval(&self, x: f64) -> f64 {
        if x <= self.grid[0] {
            return self.values[0];
        }
        let last = self.grid.len() - 1;
        if x >= self.grid[last] {
            return self.values[last];
        }

        let mut lo = 0usize;
        let mut hi = last;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.grid[mid] <= x {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        let x0 = self.grid[lo];
        let x1 = self.grid[hi];
        let y0 = self.values[lo];
        let y1 = self.values[hi];
        y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApproximationMode {
    Exact,
    Grid,
    Spline,
    Surrogate,
}

pub trait LikelihoodApproximation {
    fn mode(&self) -> ApproximationMode;
    fn fit(&mut self, data: &[WienerObservation], params: &[Wiener4Params]);
    fn log_pdf(&self, obs: &WienerObservation, params: &Wiener4Params) -> f64;
}

#[derive(Debug)]
struct WienerSeriesConfig {
    // todo: TODO,
}
#[derive(Debug)]
pub struct WienerGridApproximation {
    pub mode: ApproximationMode,
    pub config: WienerSeriesConfig,
    pub decision_time_knots: OwnedBuffer,
    pub beta_knots: OwnedBuffer,
    pub values_upper: OwnedBuffer,
    pub values_lower: OwnedBuffer,
}

impl WienerGridApproximation {
    #[inline]
    pub fn new(config: WienerSeriesConfig, len: usize) -> Self {
        Self {
            mode: ApproximationMode::Grid,
            config,
            decision_time_knots: OwnedBuffer::new(len),
            beta_knots: OwnedBuffer::new(len),
            values_upper: OwnedBuffer::new(len),
            values_lower: OwnedBuffer::new(len),
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.decision_time_knots.clear();
        self.beta_knots.clear();
        self.values_upper.clear();
        self.values_lower.clear();
    }
}

impl LikelihoodApproximation for WienerGridApproximation {
    #[inline]
    fn mode(&self) -> ApproximationMode {
        self.mode
    }

    #[inline]
    fn fit(&mut self, _data: &[WienerObservation], _params: &[Wiener4Params]) {
        self.mode = ApproximationMode::Grid;
    }

    #[inline]
    fn log_pdf(&self, _obs: &WienerObservation, _params: &Wiener4Params) -> f64 {
        f64::NAN
    }
}
