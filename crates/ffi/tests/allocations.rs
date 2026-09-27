#![cfg(not(miri))]
use ffi::{Bounds, Options, hcubature_into};

#[test]
fn successful_closure_boundary_has_no_rust_heap_allocations() {
    let bounds = Bounds::new(&[0.0], &[1.0]);
    let mut val = [0.0; 1];
    let mut err = [0.0; 1];
    let mut calls = 0;
    let count = allocation_counter::measure(|| {
        for _ in 0..16 {
            hcubature_into(
                1,
                bounds,
                Options::default(),
                &mut val,
                &mut err,
                |x, out| {
                    calls += 1;
                    out[0] = x[0] * x[0];
                    0
                },
            )
            .unwrap();
            assert!((val[0] - 1.0 / 3.0).abs() < 1e-12);
        }
    });
    assert!(calls >= 16);
    assert_eq!(count.count_total, 0);
    assert_eq!(count.bytes_total, 0);
    // This measures the Rust wrapper only. The C harness separately counts and
    // verifies cleanup of foreign malloc/calloc/realloc allocations.
}
