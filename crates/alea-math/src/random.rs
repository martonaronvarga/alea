use rand::{Rng, RngExt};

/// Marsaglia polar normal draw. Reject zero to avoid log(0) and division by zero.
pub fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    loop {
        let u = 2.0 * rng.random::<f64>() - 1.0;
        let v = 2.0 * rng.random::<f64>() - 1.0;
        let s = u * u + v * v;
        if s > 0.0 && s < 1.0 {
            return u * (-2.0 * s.ln() / s).sqrt();
        }
    }
}
/// Polar normal generator retaining the second variate for the next call.
/// Chain-local state; preserving it avoids discarding half the generated normals.
#[derive(Debug, Clone, Default)]
pub struct NormalGenerator {
    spare: Option<f64>,
}
impl NormalGenerator {
    pub fn sample<R: Rng + ?Sized>(&mut self, rng: &mut R) -> f64 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        loop {
            let u = 2.0 * rng.random::<f64>() - 1.0;
            let v = 2.0 * rng.random::<f64>() - 1.0;
            let radius = u * u + v * v;
            if radius > 0.0 && radius < 1.0 {
                let scale = (-2.0 * radius.ln() / radius).sqrt();
                self.spare = Some(v * scale);
                return u * scale;
            }
        }
    }
}
