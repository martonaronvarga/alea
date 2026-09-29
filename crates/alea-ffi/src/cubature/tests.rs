use super::*;
use std::mem::MaybeUninit;

fn invoke<F: FnMut(&[f64], &mut [f64]) -> c_int>(state: &mut CallbackState<F>) -> c_int {
    let input = [0.25];
    let mut output = [MaybeUninit::<f64>::uninit(); 2];
    // SAFETY: exact live state type, one initialized input, two disjoint writable
    // aligned outputs. Uninitialized output deliberately models C malloc scratch.
    unsafe {
        trampoline::<F>(
            1,
            input.as_ptr(),
            std::ptr::from_mut(state).cast(),
            2,
            output.as_mut_ptr().cast(),
        )
    }
}

fn state<F: FnMut(&[f64], &mut [f64]) -> c_int>(f: F) -> CallbackState<F> {
    CallbackState {
        f,
        dim: 1,
        fdim: 2,
        failure: None,
        panic: None,
    }
}

#[test]
fn callback_initializes_foreign_scratch_before_exposing_slices() {
    let mut calls = 0;
    let mut s = state(|x, out| {
        calls += 1;
        assert!(out.iter().all(|v| v.is_nan()));
        out.copy_from_slice(&[x[0], x[0] * 2.0]);
        0
    });
    assert_eq!(invoke(&mut s), 0);
    assert_eq!(invoke(&mut s), 0);
    assert!(s.failure.is_none());
    assert_eq!(calls, 2);
}

#[test]
fn partial_nonfinite_and_error_outputs_are_latched() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut s = state(|_, out| {
            out.copy_from_slice(&[1.0, value]);
            0
        });
        assert_eq!(invoke(&mut s), 1);
        assert_eq!(s.failure, Some(CubatureError::NonFiniteOutput { index: 1 }));
    }
    let mut calls = 0;
    let mut s = state(|_, out| {
        calls += 1;
        out[0] = 1.0;
        0
    });
    assert_eq!(invoke(&mut s), 1);
    assert_eq!(invoke(&mut s), 1);
    assert_eq!(s.failure, Some(CubatureError::NonFiniteOutput { index: 1 }));
    assert_eq!(calls, 1);
    let mut s = state(|_, _| -7);
    assert_eq!(invoke(&mut s), 1);
    assert_eq!(s.failure, Some(CubatureError::Callback { code: -7 }));
}

#[test]
fn panic_is_retained_until_rust_regains_control() {
    let mut calls = 0;
    let mut s = state(|_, _| {
        calls += 1;
        panic!("integrand failed")
    });
    assert_eq!(invoke(&mut s), 1);
    assert_eq!(invoke(&mut s), 1);
    let payload = s.panic.take().unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| resume_unwind(payload))).is_err());
    assert_eq!(calls, 1);
}

#[test]
fn wrong_callback_dimension_never_calls_closure() {
    let mut s = state(|_, _| panic!("must not run"));
    s.dim = 2;
    assert_eq!(invoke(&mut s), 1);
    assert_eq!(s.failure, Some(CubatureError::CallbackDimension));
}

#[test]
fn invalid_inputs_fail_before_foreign_calls_or_output_mutation() {
    let bounds = Bounds::new(&[0.0], &[1.0]);
    let opts = Options::default();
    let mut value = [42.0];
    let mut error = [43.0];
    let bad = Bounds {
        xmin: &[0.0, 0.0],
        xmax: &[1.0],
    };
    assert_eq!(
        hcubature_into(1, bad, opts, &mut value, &mut error, |_, _| panic!()),
        Err(CubatureError::BoundsLength)
    );
    assert!(matches!(
        hcubature_into(2, bounds, opts, &mut value, &mut error, |_, _| panic!()),
        Err(CubatureError::OutputLength { .. })
    ));
    for v in [-1.0, f64::NAN, f64::INFINITY] {
        let invalid = Options {
            req_abs_error: v,
            ..opts
        };
        assert_eq!(
            hcubature_into(1, bounds, invalid, &mut value, &mut error, |_, _| panic!()),
            Err(CubatureError::InvalidTolerance)
        );
    }
    let both_zero = Options {
        req_abs_error: 0.0,
        req_rel_error: 0.0,
        ..opts
    };
    assert_eq!(
        validate(1, bounds, both_zero),
        Err(CubatureError::InvalidTolerance)
    );
    assert_eq!(
        // SAFETY: no callback exists, so validation rejects before accessing userdata.
        unsafe {
            hcubature_ptr(
                1,
                None,
                std::ptr::null_mut(),
                bounds,
                opts,
                &mut value,
                &mut error,
            )
        },
        Err(CubatureError::MissingCallback)
    );
    assert_eq!(value, [42.0]);
    assert_eq!(error, [43.0]);
    hcubature_into(0, bounds, opts, &mut [], &mut [], |_, _| panic!()).unwrap();
}

#[test]
fn dimensions_and_bounds_cover_backend_integer_limits() {
    for dim in 0..=65 {
        let lo = vec![0.0; dim];
        let hi = vec![1.0; dim];
        let bounds = Bounds::new(&lo, &hi);
        let result = validate(1, bounds, Options::default());
        if dim <= 20 {
            assert!(result.is_ok(), "dimension {dim}");
        }
        if dim >= 26 {
            assert_eq!(result, Err(CubatureError::DimensionOverflow));
        }
        assert_eq!(
            validate(u32::MAX, bounds, Options::default()),
            Err(CubatureError::DimensionOverflow)
        );
    }
    for (lo, hi) in [
        (1.0, 0.0),
        (f64::NAN, 1.0),
        (0.0, f64::INFINITY),
        (-f64::MAX, f64::MAX),
        (f64::MAX, f64::MAX),
    ] {
        assert!(matches!(
            Bounds::try_new(&[lo], &[hi]),
            Err(CubatureError::InvalidBounds { index: 0 })
        ));
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "native cubature C execution; trampoline tested separately under Miri"
)]
fn native_panics_errors_and_partial_outputs_return_through_c() {
    let bounds = Bounds::new(&[0.0], &[1.0]);
    let opts = Options::default();
    assert!(
        catch_unwind(|| hcubature(1, bounds, opts, |_, _| panic!("native roundtrip"))).is_err()
    );
    assert_eq!(
        hcubature(1, bounds, opts, |_, _| -19),
        Err(CubatureError::Callback { code: -19 })
    );
    assert_eq!(
        hcubature(2, bounds, opts, |_, out| {
            out[0] = 1.0;
            0
        }),
        Err(CubatureError::NonFiniteOutput { index: 1 })
    );
    let (value, error) = hcubature(1, bounds, opts, |x, out| {
        out[0] = x[0] * x[0];
        0
    })
    .unwrap();
    assert!((value[0] - 1.0 / 3.0).abs() < 1e-12);
    assert!(error[0] < 1e-8);
    let (value, _) = hcubature(1, Bounds::new(&[], &[]), opts, |x, out| {
        assert!(x.is_empty());
        out[0] = 7.0;
        0
    })
    .unwrap();
    assert_eq!(value, [7.0]);
}
