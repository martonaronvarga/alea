fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "openblas")]
    {
        println!("cargo:rerun-if-changed=src/openblas_abi.c");
        let library = pkg_config::Config::new().probe("openblas").expect(
            "the openblas feature requires LP64 OpenBLAS and pkg-config; use nix develop .#blas",
        );
        for include in &library.include_paths {
            let header = include.join("openblas_config.h");
            if header.is_file() {
                println!("cargo:rerun-if-changed={}", header.display());
            }
        }
        // Do not accept headers from an ILP64 installation with i32 Rust bindings.
        cc::Build::new()
            .opt_level(1)
            .file("src/openblas_abi.c")
            .includes(&library.include_paths)
            .flag_if_supported("-std=c11")
            .compile("alea_openblas_abi");
    }
}
