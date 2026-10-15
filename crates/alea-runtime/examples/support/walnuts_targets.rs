//! Targets shared by the M7 benchmark and its derivative regression.

use alea_core::target::LogDensityGradient;
use std::{cell::Cell, convert::Infallible};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    Funnel,
    StochasticVolatility,
}

impl Kind {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Funnel => "funnel",
            Self::StochasticVolatility => "stochastic_volatility",
        }
    }

    pub(crate) fn dimension(self) -> usize {
        match self {
            Self::Funnel => 2,
            Self::StochasticVolatility => 50,
        }
    }

    pub(crate) fn observables(self, position: &[f64]) -> [f64; 5] {
        match self {
            Self::Funnel => {
                let standardized = position[1] * (-0.5 * position[0]).exp();
                [
                    position[0],
                    standardized,
                    position[0] * position[0],
                    standardized * standardized,
                    position[0] * standardized,
                ]
            }
            Self::StochasticVolatility => {
                let mean = position.iter().sum::<f64>() / position.len() as f64;
                let variance = position
                    .iter()
                    .map(|value| (value - mean).powi(2))
                    .sum::<f64>()
                    / position.len() as f64;
                [
                    mean,
                    variance,
                    position[0],
                    position[position.len() / 2],
                    position[position.len() - 1],
                ]
            }
        }
    }
}

pub(crate) struct Target {
    kind: Kind,
    calls: Cell<usize>,
    observations: Vec<f64>,
}

impl Target {
    pub(crate) fn new(kind: Kind) -> Self {
        let observations = (0..kind.dimension())
            .map(|index| {
                let time = index as f64;
                0.35 * (0.37 * time).sin() + 0.15 * (0.11 * time).cos()
            })
            .collect();
        Self {
            kind,
            calls: Cell::new(0),
            observations,
        }
    }

    pub(crate) fn calls(&self) -> usize {
        self.calls.get()
    }
}

impl LogDensityGradient for Target {
    type Error = Infallible;

    fn dimension(&self) -> usize {
        self.kind.dimension()
    }

    fn logp_grad(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
        self.calls.set(self.calls.get() + 1);
        if matches!(self.kind, Kind::Funnel) {
            let precision = (-q[0]).exp();
            gradient[0] = -q[0] - 0.5 + 0.5 * q[1] * q[1] * precision;
            gradient[1] = -q[1] * precision;
            return Ok(-0.5 * q[0] * q[0] - 0.5 * q[0] - 0.5 * q[1] * q[1] * precision);
        }

        // x_0 ~ N(0, sigma^2 / (1 - phi^2)), x_t | x_{t-1} ~
        // N(phi*x_{t-1}, sigma^2), y_t | x_t ~ N(0, exp(x_t)).
        const PHI: f64 = 0.95;
        const SIGMA: f64 = 0.2;
        let precision = 1.0 / (SIGMA * SIGMA);
        gradient.fill(0.0);
        let mut logp = -0.5 * (1.0 - PHI * PHI) * q[0] * q[0] * precision;
        gradient[0] -= (1.0 - PHI * PHI) * q[0] * precision;
        for time in 1..q.len() {
            let residual = q[time] - PHI * q[time - 1];
            logp -= 0.5 * residual * residual * precision;
            gradient[time] -= residual * precision;
            gradient[time - 1] += PHI * residual * precision;
        }
        for ((&latent, &observation), gradient) in
            q.iter().zip(&self.observations).zip(gradient.iter_mut())
        {
            let scaled_square = observation * observation * (-latent).exp();
            logp -= 0.5 * (latent + scaled_square);
            *gradient += -0.5 + 0.5 * scaled_square;
        }
        Ok(logp)
    }
}
