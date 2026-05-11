pub use crate::cubature::*;
pub use crate::cubature_raw::{ErrorNorm, Integrand};

/// Safe wrappers for the `cubature` C library.
///
/// Design:
/// - `cubature_raw`: raw `extern "C"` FFI declarations (unsafe, pointer-based).
/// - `cubature`: safe/idiomatic Rust façade:
///   - accepts slices instead of raw pointers,
///   - bundles common parameters into `Bounds` and `Options`,
///   - provides a closure-based integrand API,
///   - provides a function-pointer API (for `extern "C"` callbacks).
///
/// Safety model:
/// - The *safe* APIs (`hcubature`, `hcubature_fn`) do not accept raw pointers.
/// - The pointer-level wrapper (`hcubature_ptr`) is `unsafe` and documents its preconditions.
///
/// Performance:
/// - The wrappers add a small constant overhead per callback (one extra indirection for the closure case),
///   but cubature integrators are typically dominated by the integrand cost and integration algorithm.
/// - No heap allocation occurs during integration besides returning the output `Vec`s (unless you choose the
///   `_into` variants with caller-provided buffers).
mod cubature {
    use crate::cubature_raw::{ErrorNorm, Integrand};
    use std::os::raw::{c_int, c_void};

    // ------------------------- errors/results -------------------------------

    /// Error returned from cubature functions (non-zero C return code)
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CubatureError(pub c_int);

    pub type CubatureResult<T> = Result<T, CubatureError>;

    impl CubatureError {
        #[inline]
        fn from_code(code: c_int) -> CubatureResult<()> {
            if code == 0 {
                Ok(())
            } else {
                Err(CubatureError(code))
            }
        }
    }

    // ------------------------- configuration -------------------------

    /// Integration bounds: integrates over the hyper-rectangle `[xmin, xmax]`.
    ///
    /// `xmin` and `xmax` must have the same length; that length is the integration dimension.
    #[derive(Debug, Clone, Copy)]
    pub struct Bounds<'a> {
        pub xmin: &'a [f64],
        pub xmax: &'a [f64],
    }

    impl<'a> Bounds<'a> {
        #[inline]
        pub fn new(xmin: &'a [f64], xmax: &'a [f64]) -> Self {
            assert_eq!(
                xmin.len(),
                xmax.len(),
                "xmin and xmax must have same length"
            );
            Self { xmin, xmax }
        }

        #[inline]
        pub fn dim(&self) -> u32 {
            self.xmin.len() as u32
        }
    }

    /// Options passed to hcubature/pcubature.
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
                max_eval: 0, // library-dependent meaning; often "no limit"
                req_abs_error: 1e-8,
                req_rel_error: 1e-8,
                norm: ErrorNorm::L2,
            }
        }
    }

    // ------------------------- unsafe pointer-level wrapper -------------------------

    /// Low-level wrapper around `cubature_raw::hcubature` with Rust slice checks.
    ///
    /// This exists for users who already have an `extern "C"` integrand and a `void*` userdata pointer.
    ///
    /// # Safety
    /// - `fdata` must be valid for reads/writes for the duration of the call (as required by the C library).
    /// - `f` must obey the C callback contract and may be called many times.
    pub unsafe fn hcubature_ptr(
        fdim: u32,
        f: Integrand,
        fdata: *mut c_void,
        bounds: Bounds<'_>,
        opts: Options,
        val: &mut [f64],
        err: &mut [f64],
    ) -> CubatureResult<()> {
        assert_eq!(val.len(), fdim as usize, "val length must equal fdim");
        assert_eq!(err.len(), fdim as usize, "err length must equal fdim");

        let dim = bounds.dim();

        unsafe {
            let code = crate::cubature_raw::hcubature(
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
            );

            CubatureError::from_code(code)
        }
    }

    // ------------------------- safe function-pointer API (no closure) -------------------------

    /// Safe wrapper for an `extern "C"` integrand with typed user data.
    ///
    /// This is the “no closure” safe option: you pass an `extern "C"` function pointer and `&mut T`
    /// as userdata, and we handle the `void*` casting internally.
    ///
    /// # Example
    /// ```no_run
    /// # use ffi::{Bounds, Options, ErrorNorm, hcubature_fn};
    /// # use std::os::raw::{c_int, c_uint, c_void};
    /// extern "C" fn my_integrand(
    ///     ndim: c_uint,
    ///     x: *const f64,
    ///     userdata: *mut c_void,
    ///     fdim: c_uint,
    ///     fval: *mut f64,
    /// ) -> c_int {
    ///     unsafe {
    ///         let x = std::slice::from_raw_parts(x, ndim as usize);
    ///         let f = std::slice::from_raw_parts_mut(fval, fdim as usize);
    ///         let counter = &mut *userdata.cast::<usize>();
    ///         *counter += 1;
    ///         f[0] = x[0];
    ///     }
    ///     0
    /// }
    ///
    /// let xmin = [0.0];
    /// let xmax = [1.0];
    /// let bounds = Bounds::new(&xmin, &xmax);
    /// let opts = Options { max_eval: 10000, req_abs_error: 1e-8, req_rel_error: 1e-8, norm: ErrorNorm::L2 };
    ///
    /// let mut evals: usize = 0;
    /// let (val, err) = hcubature_fn(1, bounds, opts, Some(my_integrand), &mut evals).unwrap();
    /// ```
    pub fn hcubature_fn<T>(
        fdim: u32,
        bounds: Bounds<'_>,
        opts: Options,
        f: Integrand,
        userdata: &mut T,
    ) -> CubatureResult<(Vec<f64>, Vec<f64>)> {
        let mut val = vec![0.0; fdim as usize];
        let mut err = vec![0.0; fdim as usize];

        // SAFETY: we pass a valid pointer to `userdata` for the duration of the call.
        unsafe {
            hcubature_ptr(
                fdim,
                f,
                (userdata as *mut T).cast::<c_void>(),
                bounds,
                opts,
                &mut val,
                &mut err,
            )?;
        }

        Ok((val, err))
    }

    /// Function-pointer based integration that writes into caller-provided output buffers.
    ///
    /// This is the allocation-free variant of [`hcubature_fn`]. It is useful when:
    /// - you want to avoid dynamic dispatch (closure trait-object call),
    /// - you already have an `extern "C"` integrand,
    /// - you want to reuse output buffers across many calls.
    ///
    /// - `val` must have length `fdim`
    /// - `err` must have length `fdim`
    ///
    /// # Example (extern "C" integrand, typed userdata, reuse buffers)
    /// ```no_run
    /// # use ffi::{Bounds, Options, ErrorNorm, hcubature_fn_into};
    /// # use std::os::raw::{c_int, c_uint, c_void};
    /// extern "C" fn my_integrand(
    ///     ndim: c_uint,
    ///     x: *const f64,
    ///     userdata: *mut c_void,
    ///     fdim: c_uint,
    ///     fval: *mut f64,
    /// ) -> c_int {
    ///     unsafe {
    ///         let x = std::slice::from_raw_parts(x, ndim as usize);
    ///         let f = std::slice::from_raw_parts_mut(fval, fdim as usize);
    ///         let counter = &mut *userdata.cast::<usize>();
    ///         *counter += 1;
    ///
    ///         // Example: return a 2-vector [x0, x1]
    ///         f[0] = x[0];
    ///         f[1] = x[1];
    ///     }
    ///     0
    /// }
    ///
    /// let xmin = [0.0, 0.0];
    /// let xmax = [1.0, 1.0];
    /// let bounds = Bounds::new(&xmin, &xmax);
    ///
    /// let opts = Options {
    ///     max_eval: 100000,
    ///     req_abs_error: 1e-8,
    ///     req_rel_error: 1e-8,
    ///     norm: ErrorNorm::L2,
    /// };
    ///
    /// let fdim = 2;
    /// let mut val = [0.0_f64; 2];
    /// let mut err = [0.0_f64; 2];
    ///
    /// let mut evals: usize = 0;
    ///
    /// hcubature_fn_into(fdim, bounds, opts, Some(my_integrand), &mut evals, &mut val, &mut err)
    ///     .unwrap();
    ///
    /// // `val`/`err` updated in-place.
    /// # let _ = (val, err, evals);
    /// ```
    pub fn hcubature_fn_into<T>(
        fdim: u32,
        bounds: Bounds<'_>,
        opts: Options,
        f: Integrand,
        userdata: &mut T,
        val: &mut [f64],
        err: &mut [f64],
    ) -> CubatureResult<()> {
        unsafe {
            hcubature_ptr(
                fdim,
                f,
                (userdata as *mut T).cast::<c_void>(),
                bounds,
                opts,
                val,
                err,
            )
        }
    }

    // ------------------------- safe closure API -------------------------

    /// Closure type used by the safe wrapper.
    pub type IntegrandFn<'a> = dyn FnMut(&[f64], &mut [f64]) -> c_int + 'a;

    /// Trampoline used by cubature to call our Rust closure.
    ///
    /// # Safety:
    /// `fdata` must be a valid pointer to `&mut IntegrandFn` for the duration of the call.
    unsafe extern "C" fn integrand_trampoline(
        ndim: u32,
        x: *const f64,
        fdata: *mut c_void,
        fdim: u32,
        fval: *mut f64,
    ) -> c_int {
        unsafe {
            let x = std::slice::from_raw_parts(x, ndim as usize);
            let fval = std::slice::from_raw_parts_mut(fval, fdim as usize);

            // Important: the lifetime is erased at FFI boundary; we just need it to be valid for this call.
            let func = &mut *(fdata as *mut &mut IntegrandFn<'_>);
            (func)(x, fval)
        }
    }

    /// Safe API: pass a Rust closure that computes `fval` from `x`.
    ///
    /// The closure may capture and mutate local variables (e.g. count evaluations).
    ///
    /// Example:
    /// ```no_run
    /// # use ffi::{Bounds, Options, ErrorNorm, hcubature};
    /// let xmin = [0.0, 0.0];
    /// let xmax = [1.5, 1.5];
    /// let bounds = Bounds::new(&xmin, &xmax);
    /// let mut evals: usize = 0;
    /// let opts = Options { max_eval: 100000, req_abs_error: 1e-8, req_rel_error: 1e-8, norm: ErrorNorm::L2 };
    ///
    /// let (val, err) = hcubature(3, bounds, opts, |x, f| {
    ///     f[0] = x[0] - x[0].floor();
    ///     f[1] = x[1] - x[1].floor();
    ///     f[2] = f[0] * f[1];
    ///     evals += 1;
    ///     0
    /// }).unwrap();
    /// ```
    pub fn hcubature<'a, F>(
        fdim: u32,
        bounds: Bounds<'_>,
        opts: Options,
        mut f: F,
    ) -> CubatureResult<(Vec<f64>, Vec<f64>)>
    where
        F: FnMut(&[f64], &mut [f64]) -> c_int + 'a,
    {
        let dim = bounds.dim();

        let mut val = vec![0.0; fdim as usize];
        let mut err = vec![0.0; fdim as usize];

        // Convert the closure to a trait object reference with non-'static lifetime.
        let mut trait_obj: &mut IntegrandFn<'a> = &mut f;

        let code = unsafe {
            crate::cubature_raw::hcubature(
                fdim,
                Some(integrand_trampoline),
                (&mut trait_obj as *mut &mut IntegrandFn).cast::<c_void>(),
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

        CubatureError::from_code(code)?;
        Ok((val, err))
    }

    /// Closure-based variant that writes into caller-provided buffers (avoids allocating output Vecs).
    ///
    /// This is the allocation-free variant of [`hcubature`]. It is useful in hot loops where you want
    /// to reuse `val`/`err` buffers across many integrations.
    ///
    /// - `val` must have length `fdim`
    /// - `err` must have length `fdim`
    ///
    /// # Returns
    /// - `Ok(())` on success (C library returned 0)
    /// - `Err(CubatureError(code))` on failure
    ///
    /// # Example (reuse buffers across multiple calls)
    /// ```no_run
    /// # use ffi::{Bounds, Options, ErrorNorm, hcubature_into};
    /// let xmin = [0.0, 0.0];
    /// let xmax = [1.5, 1.5];
    /// let bounds = Bounds::new(&xmin, &xmax);
    ///
    /// let opts = Options {
    ///     max_eval: 100000,
    ///     req_abs_error: 1e-8,
    ///     req_rel_error: 1e-8,
    ///     norm: ErrorNorm::L2,
    /// };
    ///
    /// let fdim = 3;
    /// let mut val = vec![0.0; fdim as usize];
    /// let mut err = vec![0.0; fdim as usize];
    ///
    /// // You can capture and mutate local state safely.
    /// let mut evals: usize = 0;
    ///
    /// // First run
    /// hcubature_into(fdim, bounds, opts, &mut val, &mut err, |x, f| {
    ///     f[0] = x[0] - x[0].floor();
    ///     f[1] = x[1] - x[1].floor();
    ///     f[2] = f[0] * f[1];
    ///     evals += 1;
    ///     0
    /// }).unwrap();
    ///
    /// // Reuse the same buffers for a second run (perhaps with different opts)
    /// let opts2 = Options { req_abs_error: 1e-10, req_rel_error: 1e-10, ..opts };
    /// hcubature_into(fdim, bounds, opts2, &mut val, &mut err, |x, f| {
    ///     f[0] = x[0];
    ///     f[1] = x[1];
    ///     f[2] = x[0] * x[1];
    ///     0
    /// }).unwrap();
    ///
    /// // `val` and `err` now contain results from the second run.
    /// # let _ = (evals, val, err);
    /// ```
    pub fn hcubature_into<'a, F>(
        fdim: u32,
        bounds: Bounds<'_>,
        opts: Options,
        val: &mut [f64],
        err: &mut [f64],
        mut f: F,
    ) -> CubatureResult<()>
    where
        F: FnMut(&[f64], &mut [f64]) -> c_int + 'a,
    {
        assert_eq!(val.len(), fdim as usize, "val length must equal fdim");
        assert_eq!(err.len(), fdim as usize, "err length must equal fdim");

        let dim = bounds.dim();
        let mut trait_obj: &mut IntegrandFn<'a> = &mut f;

        let code = unsafe {
            crate::cubature_raw::hcubature(
                fdim,
                Some(integrand_trampoline),
                (&mut trait_obj as *mut &mut IntegrandFn<'a>).cast::<c_void>(),
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

        CubatureError::from_code(code)
    }
}

mod cubature_raw {
    use std::os::raw::{c_int, c_uint, c_void};

    #[derive(Clone, Copy, Debug)]
    #[repr(C)]
    pub enum ErrorNorm {
        Individual = 0,
        Paired = 1,
        L2 = 2,
        L1 = 3,
        LInf = 4,
    }

    pub type Integrand = ::std::option::Option<
        unsafe extern "C" fn(
            ndim: c_uint,
            x: *const f64,
            arg1: *mut c_void,
            fdim: c_uint,
            fval: *mut f64,
        ) -> c_int,
    >;

    pub type IntegrandV = ::std::option::Option<
        unsafe extern "C" fn(
            ndim: c_uint,
            npt: usize,
            x: *const f64,
            arg1: *mut c_void,
            fdim: c_uint,
            fval: *mut f64,
        ) -> c_int,
    >;

    unsafe extern "C" {

        pub fn hcubature(
            fdim: c_uint,
            f: Integrand,
            fdata: *mut c_void,
            dim: c_uint,
            xmin: *const f64,
            xmax: *const f64,
            maxEval: usize,
            reqAbsError: f64,
            reqRelError: f64,
            norm: ErrorNorm,
            val: *mut f64,
            err: *mut f64,
        ) -> c_int;

        pub fn hcubature_v(
            fdim: c_uint,
            f: IntegrandV,
            fdata: *mut c_void,
            dim: c_uint,
            xmin: *const f64,
            xmax: *const f64,
            maxEval: usize,
            reqAbsError: f64,
            reqRelError: f64,
            norm: ErrorNorm,
            val: *mut f64,
            err: *mut f64,
        ) -> c_int;

        pub fn pcubature(
            fdim: c_uint,
            f: Integrand,
            fdata: *mut c_void,
            dim: c_uint,
            xmin: *const f64,
            xmax: *const f64,
            maxEval: usize,
            reqAbsError: f64,
            reqRelError: f64,
            norm: ErrorNorm,
            val: *mut f64,
            err: *mut f64,
        ) -> c_int;

        pub fn pcubature_v(
            fdim: c_uint,
            f: IntegrandV,
            fdata: *mut c_void,
            dim: c_uint,
            xmin: *const f64,
            xmax: *const f64,
            maxEval: usize,
            reqAbsError: f64,
            reqRelError: f64,
            norm: ErrorNorm,
            val: *mut f64,
            err: *mut f64,
        ) -> c_int;

        pub fn pcubature_v_buf(
            fdim: c_uint,
            f: IntegrandV,
            fdata: *mut c_void,
            dim: c_uint,
            xmin: *const f64,
            xmax: *const f64,
            maxEval: usize,
            reqAbsError: f64,
            reqRelError: f64,
            norm: ErrorNorm,
            m: *mut c_uint,
            buf: *mut *mut f64,
            nbuf: *mut usize,
            max_nbuf: usize,
            val: *mut f64,
            err: *mut f64,
        ) -> c_int;
    }
}

#[cfg(test)]
mod tests {
    use super::cubature as safe;
    use super::cubature_raw as raw;
    use std::ptr::addr_of_mut;

    #[test]
    fn h_integrate_sawtooth() {
        extern "C" fn my_integrand(
            ndim: ::std::os::raw::c_uint,
            x: *const f64,
            arg1: *mut ::std::os::raw::c_void,
            fdim: ::std::os::raw::c_uint,
            fval: *mut f64,
        ) -> ::std::os::raw::c_int {
            unsafe {
                let xv = std::slice::from_raw_parts(x, ndim as usize);
                let fvalv = std::slice::from_raw_parts_mut(fval, fdim as usize);
                fvalv[0] = xv[0] - xv[0].floor();
                fvalv[1] = xv[1] - xv[1].floor();
                fvalv[2] = (xv[0] - xv[0].floor()) * (xv[1] - xv[1].floor());
                *arg1.cast::<usize>() += 1;
            }
            0
        }

        let xmin = [0.0, 0.0];
        let xmax = [1.5, 1.5];
        let mut num_eval: usize = 0;
        let max_eval = 100000;
        let req_abs_error = 1.0e-8;
        let req_rel_error = 1.0e-8;
        let mut valv: [f64; 3] = Default::default();
        let mut errv: [f64; 3] = Default::default();
        unsafe {
            raw::hcubature(
                3,
                Some(my_integrand),
                addr_of_mut!(num_eval) as *mut _,
                2,
                xmin.as_ptr(),
                xmax.as_ptr(),
                max_eval,
                req_abs_error,
                req_rel_error,
                raw::ErrorNorm::L2,
                valv.as_mut_ptr(),
                errv.as_mut_ptr(),
            );
        }
        assert!(num_eval < max_eval);

        let expected_vals = [0.9375, 0.9375, 0.390625];
        for ((&val, &err), &expected_val) in valv.iter().zip(errv.iter()).zip(expected_vals.iter())
        {
            assert!((val - expected_val).abs() < req_abs_error);
            assert!(err < req_abs_error);
        }
    }

    #[test]
    fn h_integrate_sawtooth_safe_matches_raw() {
        // shared setup
        let xmin = [0.0, 0.0];
        let xmax = [1.5, 1.5];

        let fdim = 3_u32;
        let dim = 2_u32;

        let max_eval = 100000_usize;
        let req_abs_error = 1.0e-8;
        let req_rel_error = 1.0e-8;

        let expected_vals = [0.9375, 0.9375, 0.390625];

        // raw integrand (extern "C") used for the raw call
        extern "C" fn my_integrand_raw(
            ndim: ::std::os::raw::c_uint,
            x: *const f64,
            arg1: *mut ::std::os::raw::c_void,
            fdim: ::std::os::raw::c_uint,
            fval: *mut f64,
        ) -> ::std::os::raw::c_int {
            unsafe {
                let xv = std::slice::from_raw_parts(x, ndim as usize);
                let fvalv = std::slice::from_raw_parts_mut(fval, fdim as usize);
                fvalv[0] = xv[0] - xv[0].floor();
                fvalv[1] = xv[1] - xv[1].floor();
                fvalv[2] = (xv[0] - xv[0].floor()) * (xv[1] - xv[1].floor());
                *arg1.cast::<usize>() += 1;
            }
            0
        }

        // run raw
        let mut num_eval_raw: usize = 0;
        let mut val_raw: [f64; 3] = Default::default();
        let mut err_raw: [f64; 3] = Default::default();

        unsafe {
            raw::hcubature(
                fdim,
                Some(my_integrand_raw),
                addr_of_mut!(num_eval_raw) as *mut _,
                dim,
                xmin.as_ptr(),
                xmax.as_ptr(),
                max_eval,
                req_abs_error,
                req_rel_error,
                raw::ErrorNorm::L2,
                val_raw.as_mut_ptr(),
                err_raw.as_mut_ptr(),
            );
        }

        assert!(num_eval_raw < max_eval);

        for i in 0..3 {
            assert!((val_raw[i] - expected_vals[i]).abs() < req_abs_error);
            assert!(err_raw[i] < req_abs_error);
        }

        // run safe closure wrapper
        let bounds = safe::Bounds::new(&xmin, &xmax);
        let opts = safe::Options {
            max_eval,
            req_abs_error,
            req_rel_error,
            norm: raw::ErrorNorm::L2, // re-exported raw enum is fine here
        };

        let mut num_eval_safe: usize = 0;
        let (val_safe, err_safe) = safe::hcubature(fdim, bounds, opts, |x, f| {
            f[0] = x[0] - x[0].floor();
            f[1] = x[1] - x[1].floor();
            f[2] = f[0] * f[1];
            num_eval_safe += 1;
            0
        })
        .expect("safe hcubature failed");

        assert!(num_eval_safe < max_eval);

        for i in 0..3 {
            assert!((val_safe[i] - expected_vals[i]).abs() < req_abs_error);
            assert!(err_safe[i] < req_abs_error);
        }

        // safe matches raw (within tolerance)
        for i in 0..3 {
            assert!(
                (val_safe[i] - val_raw[i]).abs() < req_abs_error,
                "val mismatch at i={i}: safe={} raw={}",
                val_safe[i],
                val_raw[i]
            );
            assert!(
                (err_safe[i] - err_raw[i]).abs() < req_abs_error,
                "err mismatch at i={i}: safe={} raw={}",
                err_safe[i],
                err_raw[i]
            );
        }
    }
}
