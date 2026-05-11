fn main() {
    compile_cubature_static();
}

fn compile_cubature_static() {
    println!("cargo:include=vendor/cubature");
    let mut build = cc::Build::new();
    build
        .cargo_metadata(true)
        .include("vendor/cubature")
        .file("vendor/cubature/hcubature.c")
        .file("vendor/cubature/pcubature.c");

    build.flag_if_supported("-std=c11");
    build.warnings(true);

    build.compile("cubature");

    // cc emits the link-search and -l: we just ensure libm on Unix for math functions
    if cfg!(not(target_os = "windows")) {
        println!("cargo:rustc-link-lib=m");
    }
}
