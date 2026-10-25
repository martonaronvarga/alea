//! Benchmark-only safe BridgeStan adapter. No library numerical code is changed.
use alea_core::{
    capability::{HessianVector, PotentialHessianDerivative},
    target::LogDensityGradient,
};
use bridgestan::{BridgeStanError, Model, StanLibrary};
use serde_json::{Value, json};
use std::{
    error::Error,
    ffi::CString,
    fmt,
    path::Path,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};
#[derive(Debug)]
pub struct Fault(pub String);
impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl Error for Fault {}
#[derive(Default)]
pub struct Count(AtomicU64);
impl Count {
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
    fn inc(&self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}
pub struct Target {
    pub model: Model<StanLibrary>,
    pub gradients: Count,
    pub hessians: Count,
    pub hvps: Count,
    pub domains: Count,
    pub last_domain: Mutex<Option<String>>,
    third_ad: Option<crate::third_ad::ThirdAd>,
    thirds: Count,
    #[cfg(feature = "native-enzyme")]
    native: Option<
        Box<
            dyn PotentialHessianDerivative<Error = alea_autodiff::autodiff::AutodiffError>
                + Send
                + Sync,
        >,
    >,
}
#[cfg(feature = "native-enzyme")]
mod native {
    include!(concat!(env!("OUT_DIR"), "/native.rs"));
}
impl Target {
    pub fn open(library: &Path, data: &Path) -> Result<Self, Box<dyn Error>> {
        let lib = bridgestan::open_library(library)?;
        let data = CString::new(data.to_string_lossy().as_bytes())?;
        Ok(Self {
            model: Model::new(lib, Some(data), 17001)?,
            gradients: Count::default(),
            hessians: Count::default(),
            hvps: Count::default(),
            domains: Count::default(),
            last_domain: Mutex::new(None),
            third_ad: None,
            thirds: Count::default(),
            #[cfg(feature = "native-enzyme")]
            native: None,
        })
    }
    pub fn backend(&self) -> &'static str {
        #[cfg(feature = "native-enzyme")]
        if self.native.is_some() {
            return "native Rust / Enzyme; ModelBuilder source export";
        }
        "BridgeStan / Stan AD"
    }
    pub fn enable_native(&mut self, case: &str) -> Result<(), Box<dyn Error>> {
        #[cfg(feature = "native-enzyme")]
        {
            let target = native::open(case).ok_or("native model unavailable; no fallback")?;
            if target.dimension() != self.dimension() {
                return Err("native/Stan dimension mismatch".into());
            }
            self.native = Some(target);
            Ok(())
        }
        #[cfg(not(feature = "native-enzyme"))]
        {
            let _ = case;
            Err("worker built without native Enzyme; no fallback".into())
        }
    }
    pub fn enable_third_ad(&mut self, library: &Path, data: &Path) -> Result<(), Box<dyn Error>> {
        let third = crate::third_ad::ThirdAd::open(library, data)?;
        if third.dimension() != self.dimension() {
            return Err("third-AD/BridgeStan dimension mismatch".into());
        }
        self.third_ad = Some(third);
        Ok(())
    }
    pub fn has_third_ad(&self) -> bool {
        #[cfg(feature = "native-enzyme")]
        if self.native.is_some() {
            return true;
        }
        self.third_ad.is_some()
    }
    pub fn validate_derivatives(&self, q: &[f64]) -> Result<Value, Box<dyn Error>> {
        let d = q.len();
        let mut g = vec![0.; d];
        self.logp_grad(q, &mut g)?;
        let mut point = q.to_vec();
        let mut scratch = vec![0.; d];
        let mut gradient_error: f64 = 0.;
        for k in 0..d {
            let h = 1e-5 * (1. + q[k].abs());
            point[k] = q[k] + h;
            let plus = self.logp_grad(&point, &mut scratch)?;
            point[k] = q[k] - h;
            let minus = self.logp_grad(&point, &mut scratch)?;
            point[k] = q[k];
            let numerical = (plus - minus) / (2. * h);
            if !numerical.is_finite() {
                return Err("nonfinite gradient validation".into());
            }
            gradient_error = gradient_error.max((numerical - g[k]).abs() / (1. + g[k].abs()));
        }
        let direction: Vec<f64> = (0..d).map(|k| if k % 2 == 0 { 1. } else { -1. }).collect();
        let h = 1e-5;
        let mut plus = vec![0.; d];
        let mut minus = vec![0.; d];
        for k in 0..d {
            point[k] = q[k] + h * direction[k];
        }
        self.logp_grad(&point, &mut plus)?;
        for k in 0..d {
            point[k] = q[k] - h * direction[k];
        }
        self.logp_grad(&point, &mut minus)?;
        if gradient_error > 1e-4 {
            return Err(format!("gradient validation: {gradient_error}").into());
        }
        let mut hvp = vec![0.; d];
        let hvp_result = self.potential_hvp(q, &direction, &mut hvp);
        let mut hvp_error: f64 = 0.;
        let mut hvp_failure = hvp_result.err().map(|e| e.to_string());
        for k in 0..d {
            let numerical = -(plus[k] - minus[k]) / (2. * h);
            if !numerical.is_finite() || !hvp[k].is_finite() {
                hvp_failure = Some("nonfinite HVP validation".to_owned());
                break;
            }
            hvp_error = hvp_error.max((numerical - hvp[k]).abs() / (1. + hvp[k].abs()));
        }
        if hvp_error > 1e-3 {
            hvp_failure = Some(format!("HVP relative error {hvp_error}"));
        }
        Ok(
            json!({"gradient_relative_error":gradient_error,"hvp_relative_error":if hvp_failure.is_none(){Some(hvp_error)}else{None},"hvp_error":hvp_failure,"scope":"central finite differences at first shared initial point; signed alternating HVP direction; HVP limitations do not reject gradient-only methods"}),
        )
    }

    pub fn counts(&self) -> Value {
        json!({"gradient_api_calls":self.gradients.get(),"hessian_api_calls":self.hessians.get(),"hvp_api_calls":self.hvps.get(),"third_ad_api_calls":self.thirds.get(),"domain_rejections":self.domains.get(),"last_domain_error":*self.last_domain.lock().unwrap_or_else(|p|p.into_inner()),"scope":"Actual target API calls; internal autodiff operations are not inferred","backend":self.backend()})
    }
    pub fn constrain(&self, q: &[f64], tp: bool) -> Result<Vec<f64>, Box<dyn Error>> {
        let mut out = vec![0.; self.model.param_num(tp, false)];
        self.model.param_constrain(
            q,
            tp,
            false,
            &mut out,
            None::<&mut bridgestan::Rng<&StanLibrary>>,
        )?;
        Ok(out)
    }
    pub fn hessian(&self, q: &[f64], out: &mut [f64]) -> Result<(), Fault> {
        self.hessians.inc();
        #[cfg(feature = "native-enzyme")]
        if let Some(native) = &self.native {
            let mut direction = vec![0.; q.len()];
            let mut product = vec![0.; q.len()];
            for j in 0..q.len() {
                direction[j] = 1.;
                native
                    .potential_hvp(q, &direction, &mut product)
                    .map_err(|e| Fault(e.to_string()))?;
                for i in 0..q.len() {
                    out[i * q.len() + j] = product[i];
                }
                direction[j] = 0.;
            }
            return Ok(());
        }
        let mut grad = vec![0.; q.len()];
        self.model
            .log_density_hessian(q, true, true, &mut grad, out)
            .map_err(|e| Fault(e.to_string()))?;
        for x in out {
            *x = -*x;
        }
        Ok(())
    }
}
impl LogDensityGradient for Target {
    type Error = Fault;
    fn dimension(&self) -> usize {
        self.model.param_unc_num()
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Fault> {
        self.gradients.inc();
        #[cfg(feature = "native-enzyme")]
        if let Some(native) = &self.native {
            return match native.logp_grad(q, g) {
                Ok(value) => Ok(value),
                Err(alea_autodiff::autodiff::AutodiffError::NonFinite) => {
                    self.domains.inc();
                    g.fill(f64::NAN);
                    Ok(f64::NEG_INFINITY)
                }
                Err(e) => Err(Fault(e.to_string())),
            };
        }
        match self.model.log_density_gradient(q, true, true, g) {
            Ok(v) => Ok(v),
            Err(BridgeStanError::EvaluationFailed(message)) => {
                self.domains.inc();
                *self.last_domain.lock().unwrap_or_else(|p| p.into_inner()) = Some(message);
                g.fill(f64::NAN);
                Ok(f64::NEG_INFINITY)
            }
            Err(e) => Err(Fault(e.to_string())),
        }
    }
}
impl HessianVector for Target {
    fn potential_hvp(&self, q: &[f64], v: &[f64], out: &mut [f64]) -> Result<(), Fault> {
        self.hvps.inc();
        #[cfg(feature = "native-enzyme")]
        if let Some(native) = &self.native {
            return native
                .potential_hvp(q, v, out)
                .map_err(|e| Fault(e.to_string()));
        }
        self.model
            .log_density_hessian_vector_product(q, v, true, true, out)
            .map_err(|e| Fault(e.to_string()))?;
        for x in out {
            *x = -*x;
        }
        Ok(())
    }
}
impl PotentialHessianDerivative for Target {
    fn potential_hessian_and_derivatives(
        &self,
        q: &[f64],
        hessian: &mut [f64],
        derivatives: &mut [f64],
    ) -> Result<(), Fault> {
        #[cfg(feature = "native-enzyme")]
        if let Some(native) = &self.native {
            self.thirds.inc();
            return native
                .potential_hessian_and_derivatives(q, hessian, derivatives)
                .map_err(|e| Fault(e.to_string()));
        }
        if let Some(third) = &self.third_ad {
            self.thirds.inc();
            return third.evaluate(q, hessian, derivatives);
        }
        let d = q.len();
        self.hessian(q, hessian)?;
        let mut point = q.to_vec();
        let mut plus = vec![0.; d * d];
        let mut minus = vec![0.; d * d];
        // Explicit benchmark-only third-derivative approximation. SoftAbs/RMHMC stays
        // exactness-pending; neither finite differences nor sampling diagnostics close it.
        for k in 0..d {
            let h = 1e-4 * (1. + q[k].abs());
            point[k] = q[k] + h;
            self.hessian(&point, &mut plus)?;
            point[k] = q[k] - h;
            self.hessian(&point, &mut minus)?;
            point[k] = q[k];
            for i in 0..d * d {
                derivatives[k * d * d + i] = (plus[i] - minus[i]) / (2. * h);
            }
        }
        Ok(())
    }
}
