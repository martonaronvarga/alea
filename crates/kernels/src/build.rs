use cxx_build::CFG;
use std::path::Path;

fn main() {
    link_cxx(Path::new("src/dist/mod.rs"));
}

fn link_cxx(lib_dir: &Path) {
    CFG.include_prefix = "";

    cxx_build::bridge(lib_dir)
        .include("../../ffi/cpp/include")
        .file("../../ffi/cpp/src/gaussian.cpp")
        .flag_if_supported("-std=c++23")
        .flag("-Wno-system-headers")
        .flag_if_supported("-Wno-unused-parameter")
        .flag("-O3")
        .flag("-march=native")
        .compile("matmod_cpp");

    println!("cargo:rerun-if-changed=src/dist/mod.rs");
    println!("cargo:rerun-if-changed=../../ffi/cpp/src/gaussian.cpp");
    println!("cargo:rerun-if-changed=../../ffi/cpp/include/gaussian.hpp");
}
