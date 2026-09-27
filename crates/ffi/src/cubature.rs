//! The safe closure boundary is monomorphized and synchronous; C owns its scratch.
use crate::cubature_raw::{ErrorNorm, Integrand};
use std::{
    any::Any,
    os::raw::{c_int, c_void},
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
};
use thiserror::Error;

/// Invalid integration inputs, callback failure, or foreign backend failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum CubatureError {
    #[error("integration bounds have different lengths")]
    BoundsLength,
    #[error("integration bounds are invalid at coordinate {index}")]
    InvalidBounds { index: usize },
    #[error("dimensions exceed the C backend's indexing or allocation limits")]
    DimensionOverflow,
    #[error("output lengths must both equal {expected}, got {value} and {error}")]
    OutputLength {
        expected: usize,
        value: usize,
        error: usize,
    },
    #[error("tolerances must be finite and nonnegative, with at least one positive")]
    InvalidTolerance,
    #[error("a callback is required")]
    MissingCallback,
    #[error("callback dimensions differ from the integration request")]
    CallbackDimension,
    #[error("callback returned error code {code}")]
    Callback { code: c_int },
    #[error("callback output {index} is incomplete or non-finite")]
    NonFiniteOutput { index: usize },
    #[error("integration result or error estimate {index} is non-finite")]
    NonFiniteResult { index: usize },
    #[error("cubature backend returned error code {code}")]
    Backend { code: c_int },
}

/// Result of a checked cubature operation.
pub type CubatureResult<T> = Result<T, CubatureError>;

/// Compatibility alias for callers storing a borrowed closure. Safe entry points
/// accept generic callbacks and do not introduce their own trait-object dispatch.
pub type IntegrandFn<'a> = dyn FnMut(&[f64], &mut [f64]) -> c_int + 'a;

/// Hyperrectangle bounds. Entry points revalidate even directly constructed values.
#[derive(Debug, Clone, Copy)]
pub struct Bounds<'a> {
    pub xmin: &'a [f64],
    pub xmax: &'a [f64],
}

impl<'a> Bounds<'a> {
    /// Compatibility constructor. Prefer [`Self::try_new`] for user inputs.
    ///
    /// # Panics
    /// Panics when the bound lengths differ or a coordinate is invalid.
    pub fn new(xmin: &'a [f64], xmax: &'a [f64]) -> Self {
        Self::try_new(xmin, xmax).expect("invalid integration bounds")
    }

    /// # Errors
    /// Rejects unequal lengths, non-finite/reversed bounds, and overflowing centers/widths.
    pub fn try_new(xmin: &'a [f64], xmax: &'a [f64]) -> CubatureResult<Self> {
        let bounds = Self { xmin, xmax };
        bounds.validate()?;
        Ok(bounds)
    }

    fn validate(self) -> CubatureResult<u32> {
        if self.xmin.len() != self.xmax.len() {
            return Err(CubatureError::BoundsLength);
        }
        for (index, (&lo, &hi)) in self.xmin.iter().zip(self.xmax).enumerate() {
            if !lo.is_finite()
                || !hi.is_finite()
                || lo > hi
                || !(hi - lo).is_finite()
                || !(hi + lo).is_finite()
            {
                return Err(CubatureError::InvalidBounds { index });
            }
        }
        u32::try_from(self.xmin.len()).map_err(|_| CubatureError::DimensionOverflow)
    }

    /// Number of coordinates, without a narrowing cast.
    pub fn dim(self) -> usize {
        self.xmin.len()
    }
}

/// Integration controls. `max_eval = 0` means no evaluation limit in C.
/// Reaching a nonzero budget is not itself a convergence certificate; inspect `err`.
#[derive(Debug, Clone, Copy)]
pub struct Options {
    pub max_eval: usize,
    pub req_abs_error: f64,
    pub req_rel_error: f64,
    pub norm: ErrorNorm,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            max_eval: 0,
            req_abs_error: 1e-8,
            req_rel_error: 1e-8,
            norm: ErrorNorm::L2,
        }
    }
}

fn validate(fdim: u32, bounds: Bounds<'_>, opts: Options) -> CubatureResult<u32> {
    let dim = bounds.validate()?;
    if !opts.req_abs_error.is_finite()
        || !opts.req_rel_error.is_finite()
        || opts.req_abs_error < 0.0
        || opts.req_rel_error < 0.0
        || (opts.req_abs_error == 0.0 && opts.req_rel_error == 0.0)
    {
        return Err(CubatureError::InvalidTolerance);
    }
    // Genz-Malik uses 1U << dim. Serial hcubature evaluates at most two regions
    // together; alloc_rule_pts doubles that capacity. Its product is unsigned C
    // arithmetic BEFORE multiplication by sizeof(double). Guard all those indices.
    let d = u64::from(dim);
    let points = match dim {
        0 => 1,
        1 => 15,
        2..32 => 1 + 4 * d + 2 * d * (d - 1) + (1_u64 << dim),
        _ => return Err(CubatureError::DimensionOverflow),
    };
    let elements = (4 * points)
        .checked_mul(d + u64::from(fdim))
        .ok_or(CubatureError::DimensionOverflow)?;
    if elements > u64::from(u32::MAX)
        || elements > (isize::MAX as u64) / 8
        || (opts.max_eval != 0 && opts.max_eval > usize::MAX - (2 * points) as usize)
    {
        return Err(CubatureError::DimensionOverflow);
    }
    Ok(dim)
}

/// Pointer-level integration with checked bounds, sizes, and options.
/// Output contents are unspecified on backend/callback error (preflight errors leave them alone).
///
/// # Errors
/// Returns input validation errors or a nonzero backend return code.
///
/// # Safety
/// `f` must obey the cubature callback ABI: read exactly `ndim` initialized input
/// doubles, initialize every one of the `fdim` output doubles before returning zero,
/// and never unwind. Output memory may initially be uninitialized: do not read it
/// or form references to its values before initializing it. `fdata` must satisfy
/// the callback's type, alignment, lifetime and aliasing requirements. Calls are
/// synchronous, serial, and may repeat; neither callback nor C may retain pointers.
pub unsafe fn hcubature_ptr(
    fdim: u32,
    f: Integrand,
    fdata: *mut c_void,
    bounds: Bounds<'_>,
    opts: Options,
    val: &mut [f64],
    err: &mut [f64],
) -> CubatureResult<()> {
    let dim = validate(fdim, bounds, opts)?;
    if val.len() != fdim as usize || err.len() != fdim as usize {
        return Err(CubatureError::OutputLength {
            expected: fdim as usize,
            value: val.len(),
            error: err.len(),
        });
    }
    if f.is_none() {
        return Err(CubatureError::MissingCallback);
    }
    if fdim == 0 {
        return Ok(());
    }
    // SAFETY: input/output slices are valid, sized, and disjoint Rust borrows.
    // Validation protects the C indexing/size arithmetic. The caller guarantees
    // callback/userdata validity and no unwinding. hcubature retains no pointers.
    let code = unsafe {
        crate::cubature_raw::hcubature(
            fdim,
            f,
            fdata,
            dim,
            bounds.xmin.as_ptr(),
            bounds.xmax.as_ptr(),
            opts.max_eval,
            opts.req_abs_error,
            opts.req_rel_error,
            opts.norm,
            val.as_mut_ptr(),
            err.as_mut_ptr(),
        )
    };
    if code == 0 {
        Ok(())
    } else {
        Err(CubatureError::Backend { code })
    }
}

/// Function-pointer integration with typed userdata and allocated result buffers.
///
/// # Errors
/// Returns the same validation/backend errors as [`hcubature_ptr`].
/// # Safety
/// All [`hcubature_ptr`] callback requirements apply. In particular, `f` must
/// interpret its userdata as this exact `T`; a typed reference alone cannot prove that.
pub unsafe fn hcubature_fn<T>(
    fdim: u32,
    bounds: Bounds<'_>,
    opts: Options,
    f: Integrand,
    userdata: &mut T,
) -> CubatureResult<(Vec<f64>, Vec<f64>)> {
    validate(fdim, bounds, opts)?;
    if f.is_none() {
        return Err(CubatureError::MissingCallback);
    }
    let mut val = vec![0.0; fdim as usize];
    let mut err = vec![0.0; fdim as usize];
    // SAFETY: the caller guarantees the callback contract and matching userdata.
    unsafe {
        hcubature_fn_into(fdim, bounds, opts, f, userdata, &mut val, &mut err)?;
    }
    Ok((val, err))
}

/// Function-pointer integration reusing result buffers. C still allocates scratch.
/// # Errors
/// Returns the same validation/backend errors as [`hcubature_ptr`].
/// # Safety
/// All [`hcubature_fn`] callback and userdata requirements apply.
pub unsafe fn hcubature_fn_into<T>(
    fdim: u32,
    bounds: Bounds<'_>,
    opts: Options,
    f: Integrand,
    userdata: &mut T,
    val: &mut [f64],
    err: &mut [f64],
) -> CubatureResult<()> {
    // SAFETY: userdata stays exclusively borrowed and alive until synchronous C
    // returns. The caller guarantees that f understands T and does not unwind.
    unsafe {
        hcubature_ptr(
            fdim,
            f,
            std::ptr::from_mut(userdata).cast(),
            bounds,
            opts,
            val,
            err,
        )
    }
}

struct CallbackState<F> {
    f: F,
    dim: u32,
    fdim: u32,
    failure: Option<CubatureError>,
    panic: Option<Box<dyn Any + Send>>,
}

/// # Safety
/// `fdata` points to a live, exclusively accessible CallbackState<F>. C provides
/// valid, non-null, aligned, disjoint ranges x[ndim] (initialized) and fval[fdim]
/// (writable, possibly uninitialized). No pointers outlive the synchronous call.
unsafe extern "C" fn trampoline<F: FnMut(&[f64], &mut [f64]) -> c_int>(
    ndim: u32,
    x: *const f64,
    fdata: *mut c_void,
    fdim: u32,
    fval: *mut f64,
) -> c_int {
    // SAFETY: guaranteed by the caller, including exact monomorphized F and lifetime.
    let state = unsafe { &mut *fdata.cast::<CallbackState<F>>() };
    if state.failure.is_some() || state.panic.is_some() {
        return 1;
    }
    if ndim != state.dim || fdim != state.fdim {
        state.failure = Some(CubatureError::CallbackDimension);
        return 1;
    }
    for i in 0..fdim as usize {
        // SAFETY: C provides fdim writable doubles. Initialize BEFORE references
        // are formed, and poison unwritten components for partial-output detection.
        unsafe {
            fval.add(i).write(f64::NAN);
        }
    }
    // SAFETY: x contains ndim initialized doubles; output was initialized above.
    // These ranges are valid and disjoint, exclusively borrowed for this callback.
    let (x, output) = unsafe {
        (
            std::slice::from_raw_parts(x, ndim as usize),
            std::slice::from_raw_parts_mut(fval, fdim as usize),
        )
    };
    match catch_unwind(AssertUnwindSafe(|| (state.f)(x, output))) {
        Ok(0) => {
            if let Some(index) = output.iter().position(|value| !value.is_finite()) {
                state.failure = Some(CubatureError::NonFiniteOutput { index });
                1
            } else {
                0
            }
        }
        Ok(code) => {
            state.failure = Some(CubatureError::Callback { code });
            1
        }
        // Keep the payload alive: even a payload's Drop could panic. Resume only
        // after returning through C, never drop/replace a pending payload here.
        Err(payload) => {
            state.panic = Some(payload);
            1
        }
    }
}

/// Integrates a Rust closure. Every output must be finite and written on success.
/// Return zero on success or a nonzero application error code to stop integration.
///
/// # Errors
/// Rejects invalid inputs, incomplete/non-finite outputs and callback/backend errors.
/// # Panics
/// A closure panic is caught at the callback boundary and resumed after C returns.
/// With `panic = "abort"`, Rust panics abort as usual.
///
/// ```
/// use ffi::{Bounds, Options, hcubature};
/// let (value, _) = hcubature(1, Bounds::try_new(&[0.0], &[1.0])?, Options::default(),
///     |x, out| { out[0] = x[0] * x[0]; 0 })?;
/// assert!((value[0] - 1.0 / 3.0).abs() < 1e-8);
/// # Ok::<(), ffi::CubatureError>(())
/// ```
pub fn hcubature<F: FnMut(&[f64], &mut [f64]) -> c_int>(
    fdim: u32,
    bounds: Bounds<'_>,
    opts: Options,
    f: F,
) -> CubatureResult<(Vec<f64>, Vec<f64>)> {
    validate(fdim, bounds, opts)?;
    let mut val = vec![0.0; fdim as usize];
    let mut err = vec![0.0; fdim as usize];
    hcubature_into(fdim, bounds, opts, &mut val, &mut err, f)?;
    Ok((val, err))
}

/// Safe monomorphized closure boundary, reusing caller-owned result buffers.
/// Rust wrapper success paths allocate nothing; the C integrator allocates scratch.
/// Outputs are unspecified after a callback/backend failure or panic.
/// # Errors
/// See [`hcubature`]. Input validation errors do not touch buffers or invoke `f`.
/// # Panics
/// See [`hcubature`]; unwinding never crosses C.
pub fn hcubature_into<F: FnMut(&[f64], &mut [f64]) -> c_int>(
    fdim: u32,
    bounds: Bounds<'_>,
    opts: Options,
    val: &mut [f64],
    err: &mut [f64],
    f: F,
) -> CubatureResult<()> {
    let dim = validate(fdim, bounds, opts)?;
    let mut state = CallbackState {
        f,
        dim,
        fdim,
        failure: None,
        panic: None,
    };
    // SAFETY: exact F trampoline and live unique state pointer; trampoline
    // initializes output, contains panics and retains no pointers. Inputs checked
    // by hcubature_ptr; C calls synchronously and serially.
    let result = unsafe {
        hcubature_ptr(
            fdim,
            Some(trampoline::<F>),
            std::ptr::from_mut(&mut state).cast(),
            bounds,
            opts,
            val,
            err,
        )
    };
    if let Some(payload) = state.panic {
        resume_unwind(payload);
    }
    if let Some(failure) = state.failure {
        return Err(failure);
    }
    result?;
    if let Some(index) = val
        .iter()
        .zip(err.iter())
        .position(|(v, e)| !v.is_finite() || !e.is_finite())
    {
        return Err(CubatureError::NonFiniteResult { index });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
