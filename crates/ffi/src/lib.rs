//! Checked synchronous cubature bindings. Rust callback panics resume only after C returns.
mod cubature;
pub use cubature::*;
pub use cubature_raw::{ErrorNorm, Integrand};

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

    }
}

#[cfg(all(test, not(miri)))]
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
            // SAFETY: synchronous C calls supply ndim initialized inputs, fdim writable
            // outputs, and the live, uniquely borrowed usize passed below.
            unsafe {
                let xv = std::slice::from_raw_parts(x, ndim as usize);
                assert_eq!(fdim, 3);
                for i in 0..3 {
                    fval.add(i).write(0.0);
                }
                let fvalv = std::slice::from_raw_parts_mut(fval, 3);
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
        // SAFETY: arrays have the stated lengths; callback matches userdata,
        // initializes every output, never unwinds, and retains no pointers.
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
            // SAFETY: synchronous C calls supply ndim initialized inputs, fdim writable
            // outputs, and the live, uniquely borrowed usize passed below.
            unsafe {
                let xv = std::slice::from_raw_parts(x, ndim as usize);
                assert_eq!(fdim, 3);
                for i in 0..3 {
                    fval.add(i).write(0.0);
                }
                let fvalv = std::slice::from_raw_parts_mut(fval, 3);
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

        // SAFETY: arrays have the stated lengths; callback matches userdata,
        // initializes every output, never unwinds, and retains no pointers.
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
