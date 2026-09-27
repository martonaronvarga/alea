{
  pkgs,
  llvmPkgs,
  bootstrap,
}:
pkgs.clangStdenv.mkDerivation {
  pname = "rust-base";
  version = "nightly-2026-04-17";

  src = pkgs.fetchurl {
    url = "https://static.rust-lang.org/dist/2026-04-17/rustc-nightly-src.tar.xz";
    sha256 = "e23864cbd25298df55ee9abacd34d7b7330b867826a7b694af9e753d5cfd04a2";
  };

  dontUpdateAutotoolsGnuConfigScripts = true;
  stripDebugList = ["bin"];

  nativeBuildInputs = with pkgs; [
    libffi
    cmake
    ninja
    rustc
    cargo
    perl
    curl
    libiconv
    python3
    file
    which
    xz
    pkg-config
    git
    sccache
  ];

  buildInputs = [
    llvmPkgs.llvm
    llvmPkgs.clang
    llvmPkgs.libcxx
    llvmPkgs.libunwind
    llvmPkgs.lld
    pkgs.zlib
    pkgs.openssl
    pkgs.xz
    bootstrap
  ];

  env = {
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      llvmPkgs.llvm
      llvmPkgs.lld
      llvmPkgs.libcxx
      pkgs.xz
      pkgs.zlib
      pkgs.openssl
    ];
  };

  postPatch = ''
    patchShebangs src/etc
  '';

  configurePhase = ''
    mkdir -p $out

      cat > bootstrap.toml <<EOF
      change-id = 154587

    [build]
    build = "${pkgs.stdenv.buildPlatform.config}"
    host = ["${pkgs.stdenv.buildPlatform.config}"]
    target = ["${pkgs.stdenv.buildPlatform.config}"]
    build-dir = "build"
    rustc = "${bootstrap}/bin/rustc"
    cargo = "${bootstrap}/bin/cargo"
    rustfmt = "${pkgs.rustfmt}/bin/rustfmt"
    docs = false
    extended = true
    tools = ["cargo", "clippy", "miri", "rust-analyzer-proc-macro-srv", "rustdoc", "rustfmt"]

    [rust]
    channel = "nightly"
    download-rustc = false
    llvm-libunwind = "system"
    rpath = true
    lld = false

    [llvm]
    download-ci-llvm = false
    link-shared = true
    use-libcxx = true
    assertions = true
    ninja = true

    [target.${pkgs.stdenv.hostPlatform.config}]
    cc = "${llvmPkgs.clang}/bin/clang"
    cxx = "${llvmPkgs.clang}/bin/clang++"
    linker = "${llvmPkgs.clang}/bin/clang"
    llvm-config = "${llvmPkgs.llvm.dev}/bin/llvm-config"

    [install]
    prefix = "$out"
    sysconfdir = "$out/etc"
    EOF
  '';

  buildPhase = ''
    python3 x.py build --stage 1 \
      library/std \
      src/tools/cargo \
      src/tools/clippy \
      src/tools/rustfmt \
      src/tools/miri
  '';

  doCheck = false;
  checkPhase = ''
    python3 x.py test --stage 1 tests/codegen-llvm/autodiff
    python3 x.py test --stage 1 tests/pretty/autodiff
    python3 x.py test --stage 1 tests/ui/autodiff
    python3 x.py test --stage 1 tests/run-make/autodiff
    python3 x.py test --stage 1 tests/ui/feature-gates/feature-gate-autodiff.rs
  '';

  installPhase = ''
    python3 x.py install --stage 1
  '';

  postInstall = ''
    # Miri and rust-analyzer need sources from exactly this compiler revision.
    mkdir -p $out/lib/rustlib/src/rust
    cp -R library $out/lib/rustlib/src/rust/library
    rm $out/lib/rustlib/install.log
    for m in $out/lib/rustlib/manifest-rust*
    do
      sort --output=$m < $m
    done

    # remove uninstall script that doesn't really make sense for Nix.
    rm $out/lib/rustlib/uninstall.sh
  '';

  passthru = {
    # Actual autodiff execution validation remains an M2 gate. The old passthru
    # test referred to an unset $src and was not a runnable check.
    isRustToolchain = true;
  };
}
