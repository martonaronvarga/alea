#![cfg(not(miri))]

#[test]
fn foreign_allocations_are_freed_on_callback_and_allocation_failures() {
    let status = std::process::Command::new(env!("ALEA_CUBATURE_AUDIT"))
        .status()
        .expect("run the native C audit binary built alongside cubature");
    assert!(status.success(), "C ownership/failure regression: {status}");
}
