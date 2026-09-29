fn main() {
    compile_cubature_static();
}

fn compile_cubature_static() {
    println!("cargo:rerun-if-changed=vendor/cubature");
    println!("cargo:rerun-if-changed=cubature_patch.rs");
    println!("cargo:include=vendor/cubature");
    let upstream = std::fs::read_to_string("vendor/cubature/hcubature.c")
        .expect("cubature submodule must be initialized");
    let checked = cubature_patch::checked_source(upstream);
    let source = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo OUT_DIR"))
        .join("hcubature_checked.c");
    std::fs::write(&source, checked).expect("write checked cubature source");
    // An independent executable tracks malloc/free and injects failures in the
    // exact generated C source. It is run by native tests, never linked into Rust.
    println!("cargo:rerun-if-changed=tests/cubature_cleanup.c");
    let audit = source.with_file_name(
        if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
            "cubature_audit.exe"
        } else {
            "cubature_audit"
        },
    );
    let compiler = cc::Build::new().get_compiler();
    let mut command = compiler.to_command();
    command
        .arg("tests/cubature_cleanup.c")
        .arg("-Ivendor/cubature")
        .arg("-I")
        .arg(source.parent().expect("OUT_DIR parent"))
        .arg("-std=c11")
        .arg("-O1")
        .arg("-UNDEBUG")
        .arg("-o")
        .arg(&audit);
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        command.arg("-lm");
    }
    assert!(
        command
            .status()
            .expect("compile C audit executable")
            .success(),
        "C audit build failed"
    );
    println!("cargo:rustc-env=ALEA_CUBATURE_AUDIT={}", audit.display());
    let mut build = cc::Build::new();
    build
        .cargo_metadata(true)
        .include("vendor/cubature")
        .file(source)
        // Cubature is a numerical kernel; even debug Rust builds need a
        // minimally optimized C build for the toolchain's fortify checks.
        .opt_level(1);

    build.flag_if_supported("-std=c11");
    build.warnings(true);

    build.compile("cubature");

    // cc emits the link-search and -l: we just ensure libm on Unix for math functions
    if cfg!(not(target_os = "windows")) {
        println!("cargo:rustc-link-lib=m");
    }
}
mod cubature_patch;
