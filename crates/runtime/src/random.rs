use rand::{Rng, RngExt};

/// Marsaglia polar normal draw. Reject zero to avoid log(0) and division by zero.
pub(crate) fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    loop {
        let u = 2.0 * rng.random::<f64>() - 1.0;
        let v = 2.0 * rng.random::<f64>() - 1.0;
        let s = u * u + v * v;
        if s > 0.0 && s < 1.0 {
            return u * (-2.0 * s.ln() / s).sqrt();
        }
    }
}
