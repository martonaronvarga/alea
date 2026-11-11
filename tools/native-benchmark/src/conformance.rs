//! Shared native-model definitions, also consumed by the optional source exporter.
use alea::distributions::{
    modeling::{continuous, normal},
    scalar::Cauchy,
};
use alea::model::assembly::{AssemblyError as E, NativeModel, ModelBuilder};

/// Existing PosteriorDB eight-schools models, in their original parameter order.
/// Covariance is a separate synthetic Cholesky model, not a PosteriorDB port.
pub fn build(name: &str, n: usize, k: usize, x: &[f64], y: &[f64]) -> Result<NativeModel, E> {
    if n == 0 || y.len() != n || n.checked_mul(k) != Some(x.len()) {
        return Err(E::Shape);
    }
    let mut b = ModelBuilder::new();
    if matches!(name, "schools" | "schools_nc") {
        if k != 1 || x.iter().any(|v| !v.is_finite() || *v <= 0.) {
            return Err(E::Shape);
        }
        let effects = (0..n)
            .map(|i| b.real(&format!("effect{i}")).map(|p| p.expr()))
            .collect::<Result<Vec<_>, _>>()?;
        let mu = b.real("mu")?;
        let tau = b.positive("tau")?;
        let zero = b.constant(0.)?;
        let one = b.constant(1.)?;
        let five = b.constant(5.)?;
        let prior = normal(&mut b, mu, zero, five)?;
        b.prior("mu", prior)?;
        // This existing opaque law has no source rule. Source AD must report
        // unsupported, never replace the prior or invent a benchmark derivative.
        let prior = continuous(&mut b, tau.expr(), &[zero, five], |p| {
            Cauchy::new(p[0], p[1])
        })?;
        b.prior("tau", prior)?;
        for (i, &effect) in effects.iter().enumerate() {
            let centered = name == "schools";
            let prior = normal(
                &mut b,
                effect,
                if centered { mu.expr() } else { zero },
                if centered { tau.expr() } else { one },
            )?;
            b.prior(&format!("effect{i}"), prior)?;
            let theta = if centered {
                effect
            } else {
                let scaled = b.multiply(effect, tau)?;
                b.add(mu, scaled)?
            };
            let obs = b.constant(y[i])?;
            let sd = b.constant(x[i])?;
            let likelihood = normal(&mut b, obs, theta, sd)?;
            b.likelihood(&format!("observation{i}"), likelihood)?;
        }
    } else if name == "covariance" {
        if k != 2 {
            return Err(E::Shape);
        }
        let mu0 = b.real("mu0")?;
        let mu1 = b.real("mu1")?;
        let s0 = b.positive("s0")?;
        let s1 = b.positive("s1")?;
        let loading = b.real("loading")?;
        for (name, p, sd) in [
            ("mu0", mu0.expr(), 2.),
            ("mu1", mu1.expr(), 2.),
            ("s0", s0.expr(), 1.),
            ("s1", s1.expr(), 1.),
            ("loading", loading.expr(), 1.),
        ] {
            let zero = b.constant(0.)?;
            let sd = b.constant(sd)?;
            let prior = normal(&mut b, p, zero, sd)?;
            b.prior(name, prior)?;
        }
        // L=[[s0,0],[loading,s1]]: conditional native normals are precisely
        // multi_normal_cholesky(mu,L). No benchmark density/score implementation.
        let log_s0 = b.log(s0)?;
        let negative_log_s0 = b.multiply(log_s0, -1.)?;
        let inverse_s0 = b.exp(negative_log_s0)?;
        let slope = b.multiply(loading, inverse_s0)?;
        let negative_mu0 = b.multiply(mu0, -1.)?;
        for i in 0..n {
            let a = b.constant(x[2 * i])?;
            let d = b.add(a, negative_mu0)?;
            let d = b.multiply(d, slope)?;
            let mean = b.add(mu1, d)?;
            let c = b.constant(x[2 * i + 1])?;
            let first = normal(&mut b, a, mu0, s0)?;
            let second = normal(&mut b, c, mean, s1)?;
            let pair = b.add(first, second)?;
            b.likelihood(&format!("pair{i}"), pair)?;
        }
    } else {
        return Err(E::Shape);
    }
    b.compile()
}

/// Dyadic fixture shared by the generated target and runtime data admission.
#[cfg(any(test, feature = "enzyme"))]
pub fn covariance_data() -> (Vec<f64>, Vec<f64>) {
    let x = (0..32)
        .flat_map(|i| {
            let a = (i % 7) as f64 / 4. - 0.75;
            [a, 0.5 * a + (i % 11) as f64 / 8. - 0.625]
        })
        .collect();
    (x, vec![0.; 32])
}
