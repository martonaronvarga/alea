{
  description = "Alea: reproducible Rust, SIMD, Miri, and experimental autodiff environments";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    naersk.url = "github:nix-community/naersk";
    naersk.inputs.nixpkgs.follows = "nixpkgs";
    fenix.url = "github:nix-community/fenix";
    fenix.inputs.nixpkgs.follows = "nixpkgs";
    treefmt-nix.url = "github:numtide/treefmt-nix";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs @ {
    self,
    nixpkgs,
    flake-utils,
    naersk,
    fenix,
    treefmt-nix,
    flake-parts,
    ...
  }:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = flake-utils.lib.defaultSystems;
      perSystem = {system, ...}: let
        pkgs = import nixpkgs {inherit system;};
        # cblas 0.4 uses i32 BLAS integers, not the nixpkgs ILP64 default.
        openblasLp64 = pkgs.openblas.override {blas64 = false;};
        stdenv = pkgs.clangStdenv;
        fenixPkgs = fenix.packages.${system};
        # flake.lock pins both the date and hashes of every component. Do not use
        # "latest": its components can come from different compiler dates.
        components = ["cargo" "rustc" "rust-std" "rust-src" "rustfmt" "clippy" "rust-analyzer"];
        stableToolchain = fenixPkgs.stable.withComponents components;
        rustToolchain = fenixPkgs.complete.withComponents (components ++ ["miri" "llvm-tools"]);

        # Autodiff has its own content-pinned upstream toolchain: the compiler
        # and Enzyme plugin must come from exactly the same distribution.
        autodiffRelease = fenixPkgs.fromToolchainName {
          name = "nightly-2026-06-23";
          sha256 = "sha256-/yl/30nnnHU8U//+i5usZZDCcBZ+QUMeQ+3uWw9lN0g=";
        };
        llvmPkgs = pkgs.llvmPackages_22;
        autodiffToolchain = autodiffRelease.withComponents (components ++ ["enzyme"]);

        cmdStan = stdenv.mkDerivation rec {
          pname = "cmdStan";
          version = "2.38.0";
          src = pkgs.fetchFromGitHub {
            owner = "stan-dev";
            repo = "cmdstan";
            tag = "v${version}";
            fetchSubmodules = true;
            hash = "sha256-4Mx4LvXW2lYOSSOgNT0f+unry6mBobgGTDLwtiypHBU=";
          };
          postPatch = ''
            substituteInPlace stan/lib/stan_math/make/libraries \
              --replace "/usr/bin/env/bash" "bash"
          '';

          nativeBuildInputs = with pkgs; [
            python3
            stanc
            gcc
            gnumake
            pkg-config
            openssl
          ];

          preConfigure =
            ''
              patchShebangs test-all.sh runCmdStanTests.py stan/
            ''
            + ''
              mkdir -p $out/opt
              cp -R . $out/opt/cmdstan
              cd $out/opt/cmdstan
              mkdir -p bin
              ln -s ${pkgs.stanc}/bin/stanc bin/stanc
            '';

          makeFlags =
            [
              "build"
            ]
            ++ pkgs.lib.optionals stdenv.hostPlatform.isDarwin [
              "arch=${stdenv.hostPlatform.darwinArch}"
            ];
          env.CXXFLAGS = pkgs.lib.optionalString stdenv.cc.isClang "-Xclang -fno-pch-timestamp";
          enableParallelBuilding = true;
          installPhase = ''
            runHook preInstall

            mkdir -p $out/bin
            ln -s $out/opt/cmdstan/bin/stanc $out/bin/stanc
            ln -s $out/opt/cmdstan/bin/stansummary $out/bin/stansummary
            ln -s $out/opt/cmdstan/bin/diagnose $out/bin/diagnose
            cat > $out/bin/stan <<EOF
            #! ${pkgs.runtimeShell}
            make -C $out/opt/cmdstan "\$(realpath "\$1")"
            EOF
            chmod a+x $out/bin/stan

            runHook postInstall
          '';

          passthru.tests = {
            test = pkgs.runCommand "cmdstan-test" {nativeBuildInputs = [pkgs.python3 pkgs.gnumake stdenv.cc];} ''
              cp -R ${cmdStan}/opt/cmdstan cmdstan
              chmod -R +w cmdstan
              cd cmdstan
              ./runCmdStanTests.py -j$NIX_BUILD_CORES src/test/interface
              touch $out
            '';
          };

          meta = {
            description = "Command-line interface to Stan";
            longDescription = ''
              Stan is a probabilistic programming language implementing full Bayesian
              statistical inference with MCMC sampling (NUTS, HMC), approximate Bayesian
              inference with Variational inference (ADVI) and penalized maximum
              likelihood estimation with Optimization (L-BFGS).
            '';
            homepage = "https://mc-stan.org/interfaces/cmdstan.html";
            license = pkgs.lib.licenses.bsd3;
          };
        };

        naersk-lib = pkgs.callPackage naersk {
          rustc = stableToolchain;
          cargo = stableToolchain;
        };
        rustPackage = naersk-lib.buildPackage {
          # naersk interpolates root while discovering manifests. Avoid copying
          # this subdirectory to an unrealised store path in read-only evaluation
          # (notably nix flake check --no-build).
          root = builtins.toString ./crates;
          src = pkgs.lib.cleanSource ./crates;
          nativeBuildInputs = [pkgs.pkg-config];
          buildInputs = [pkgs.openssl];
          # No nonexistent ffi-backend feature, nightly flags, or global BLAS linking.
        };
        treefmtEval = treefmt-nix.lib.evalModule pkgs ./treefmt.nix;

        mkRustShell = toolchain: extra:
          pkgs.mkShell ({
              packages = [toolchain pkgs.python3 pkgs.pkg-config pkgs.cmake pkgs.eigen pkgs.openssl];
              RUSTC = "${toolchain}/bin/rustc";
              RUSTDOC = "${toolchain}/bin/rustdoc";
              RUST_SRC_PATH = "${toolchain}/lib/rustlib/src/rust/library";
              MIRI_LIB_SRC = "${toolchain}/lib/rustlib/src/rust/library";
              OPENBLAS_NUM_THREADS = "1";
              # No nested interactive shell, local symlink lookup, RUSTC_BOOTSTRAP,
              # CPU-specific flags, or global LD_LIBRARY_PATH. --command works in CI.
            }
            // extra);
      in {
        packages = {
          inherit rustToolchain stableToolchain autodiffToolchain cmdStan rustPackage;
          default = rustPackage;
        };
        apps.default = {
          type = "app";
          program = "${rustPackage}/bin/alea";
          meta.description = "Run the Alea experiment application";
        };
        formatter = treefmtEval.config.build.wrapper;
        checks = {
          formatting = treefmtEval.config.build.check self;
          toolchain = pkgs.runCommand "alea-toolchain-check" {nativeBuildInputs = [rustToolchain];} ''
            rustc --version
            cargo --version
            cargo clippy --version
            cargo miri --version
            rustfmt --version
            test -f ${rustToolchain}/lib/rustlib/src/rust/library/Cargo.toml
            touch "$out"
          '';
        };
        devShells = {
          # Default: one pinned nightly with matching Clippy + Miri + rust-src.
          default = mkRustShell rustToolchain {};
          nightly = self.devShells.${system}.default;
          stable = mkRustShell stableToolchain {};
          # Focused profiling tools without the full research/Python/R environment.
          profiling = mkRustShell stableToolchain {
            nativeBuildInputs =
              (with pkgs; [hyperfine time])
              ++ pkgs.lib.optionals pkgs.stdenv.isLinux (with pkgs; [perf valgrind]);
          };
          blas = mkRustShell rustToolchain {
            buildInputs = [openblasLp64];
            # Fenix's upstream linker does not embed Nix library rpaths.
            # Scope runtime lookup to this opt-in shell, never the default shell.
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [openblasLp64];
          };
          # Upstream prebuilt Enzyme; opt in explicitly. Use release mode for AD.
          autodiff = mkRustShell autodiffToolchain {
            # Fat LTO is set by the workspace release profile, not globally:
            # applying -Clto=fat to proc-macro build dependencies is invalid.
            RUSTFLAGS = "-Zautodiff=Enable -Cembed-bitcode=yes";
          };
          full = mkRustShell rustToolchain {
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [openblasLp64];
            packages =
              [rustToolchain cmdStan openblasLp64]
              ++ (with pkgs; [
                pkg-config
                cmake
                clang
                eigen
                boost
                fmt
                openssl
                openmpi
                opencl-headers
                ocl-icd
                clang-tools
                cppcheck
                clang-analyzer
                llvmPkgs.libcxx
                llvmPkgs.libunwind
                zig
                zls
                futhark
                gdb
                lldb
                hyperfine
                lcov
                ccache
                bear
                doxygen
              ])
              ++ pkgs.lib.optionals pkgs.stdenv.isLinux (with pkgs; [valgrind perf hotspot massif-visualizer])
              ++ [
                (pkgs.python3.withPackages (ps: with ps; [numpy pandas matplotlib seaborn pyarrow torch]))
                (pkgs.rWrapper.override {packages = with pkgs.rPackages; [tidyverse lme4 lmerTest brms rmarkdown];})
              ];
          };
        };
      };
    };
}
