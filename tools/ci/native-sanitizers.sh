#!/usr/bin/env bash
set -euo pipefail
# cc-rs compiles the exact patched source and standalone allocation-failure
# harness. ASan/UBSan instrument both C objects and the independently run harness.
export CFLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer'
export RUSTFLAGS='-C link-arg=-fsanitize=address -C link-arg=-fsanitize=undefined'
export ASAN_OPTIONS='detect_leaks=1:halt_on_error=1'
export UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1'
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target/sanitizers}"
cargo test --manifest-path crates/Cargo.toml -p alea-ffi --locked --test native_cleanup
