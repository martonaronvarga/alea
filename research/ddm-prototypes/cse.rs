use kernels::dist::wiener::Wiener4Params;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Congruency {
    Congruent,
    Incongruent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CseCondition {
    CC,
    CI,
    IC,
    II,
}

impl CseCondition {
    #[inline]
    pub fn index(self) -> usize {
        match self {
            Self::CC => 0,
            Self::CI => 1,
            Self::IC => 2,
            Self::II => 3,
        }
    }

    #[inline]
    pub fn all() -> [Self; 4] {
        [Self::CC, Self::CI, Self::IC, Self::II]
    }

    /// The chosen basis is signed and centered to suit affine latent maps:
    ///   CC -> (+1, +1)
    ///   CI -> (+1, -1)
    ///   IC -> (-1, +1)
    ///   II -> (-1, -1)
    #[inline]
    pub const fn as_f64(self) -> (f64, f64) {
        match self {
            Self::CC => (1.0, 1.0),
            Self::CI => (1.0, -1.0),
            Self::IC => (-1.0, 1.0),
            Self::II => (-1.0, -1.0),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CseObservation<'a> {
    pub condition: &'a [CseCondition],
    pub rt: &'a [f64],
    pub boundary: &'a [Boundary],
}

impl<'a> CseObservation<'a> {
    #[inline]
    pub fn len(&self) -> usize {
        self.condition
            .len()
            .min(self.rt.len())
            .min(self.boundary.len())
    }

    #[inline]
    pub fn validate(&self) -> bool {
        self.condition.len() == self.rt.len() && self.rt.len() == self.boundary.len()
    }

    #[inline]
    pub fn wiener_view(&self) -> WienerObservation<'_> {
        WienerObservation {
            rt: self.rt,
            boundary: self.boundary,
        }
    }
}

/// Feature encoder for CSE to latent-state projection.
#[derive(Debug, Clone, Copy, Default)]
pub struct CseFeatureMap;

impl CseFeatureMap {
    #[inline]
    pub fn encode(&self, condition: CseCondition) -> (f64, f64) {
        condition.as_f64()
    }

    #[inline]
    pub fn encode_into(&self, condition: &[CseCondition], c: &mut [f64], m: &mut [f64]) -> bool {
        if condition.len() != c.len() || c.len() != m.len() {
            return false;
        }
        for (i, cond) in condition.iter().copied().enumerate() {
            let (ci, mi) = self.encode(cond);
            c[i] = ci;
            m[i] = mi;
        }
        true
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CseLatent {
    pub modulation: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct CseParameterMap {
    pub drifts: [f64; 4],
    pub alpha: [f64; 3],
    pub beta: [f64; 3],
    pub tau: [f64; 3],
}

impl CseParameterMap {
    #[inline]
    pub fn new(drifts: [f64; 4], alpha: [f64; 3], tau: [f64; 3], beta: [f64; 3]) -> Self {
        Self {
            drifts,
            alpha,
            tau,
            beta,
        }
    }

    #[inline]
    fn sigmoid(x: f64) -> f64 {
        if x >= 0.0 {
            let e = (-x).exp();
            1.0 / (1.0 + e)
        } else {
            let e = x.exp();
            e / (1.0 + e)
        }
    }

    #[inline]
    fn dot(coeffs: &[f64; 3], c: f64, m: f64) -> f64 {
        coeffs[0] + coeffs[1] * c + coeffs[2] * m
    }

    /// Map a single condition to canonical Wiener parameters.
    #[inline]
    pub fn map_condition(&self, condition: CseCondition) -> (f64, f64, f64, f64) {
        let (c, m) = condition.latent_features();
        (
            Self::dot(&self.alpha, c, m).exp(),
            Self::dot(&self.tau, c, m).exp(),
            Self::sigmoid(Self::dot(&self.beta, c, m)),
            self.drifts[condition.index()],
        )
    }
}
