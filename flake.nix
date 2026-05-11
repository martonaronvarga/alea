{
  description = "CSE Modeling Environment";
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
      imports = [];
      perSystem = {system, ...}: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [];
        };
        stdenv = pkgs.clangStdenv;
        llvmPkgs = pkgs.llvmPackages_22;

        enzyme = pkgs.callPackage ./nix/enzyme.nix {
          inherit llvmPkgs;
        };

        fenixPkgs = fenix.packages.${system};
        bootstrapRust = fenixPkgs.latest.toolchain;

        rustToolchainBase = pkgs.callPackage ./nix/rust-toolchain.nix {
          version = "nightly";
          bootstrap = bootstrapRust;
          inherit pkgs llvmPkgs;
        };
        rustToolchain = pkgs.symlinkJoin {
          name = "rust-toolchain";
          paths = [
            rustToolchainBase
          ];
          nativeBuildInputs = [pkgs.makeWrapper];

          postBuild = ''
            libdir=$out/lib/rustlib/x86_64-unknown-linux-gnu/lib/
            mkdir -p $libdir
            ln -s ${enzyme}/lib/LLVMEnzyme-22.so $libdir/
            if [ -f ${enzyme}/lib/LLVMEnzyme-22.so ]; then
              ln -s ${enzyme}/lib/LLVMEnzyme-22.so $libdir/libEnzyme-22.so
            else
              echo "WARNING: Could not find LLVMEnzyme-22.so in ${enzyme}/lib/"
            fi

            # wrapProgram $out/bin/rustc --add-flags "--sysroot $out"
            wrapProgram $out/bin/cargo --set RUSTC $out/bin/rustc
            # wrapProgram $out/bin/rustdoc --add-flags "--sysroot $out"

            # # Wrap the compiler binaries to enforce the custom sysroot
            # makeWrapper ${rustToolchainBase}/bin/rustc $out/bin/rustc --add-flags "--sysroot $out"
            # makeWrapper ${rustToolchainBase}/bin/rustdoc $out/bin/rustdoc --add-flags "--sysroot $out"

            # # Wrap cargo
            # makeWrapper ${rustToolchainBase}/bin/cargo $out/bin/cargo \
            #   --set RUSTC "$out/bin/rustc" \
            #   --set RUSTDOC "$out/bin/rustdoc" \
            #   --set RUSTC_WORKSPACE_WRAPPER "$out/bin/rustc"

            # # # Wrap cargo-clippy to ensure it inherits our settings
            # makeWrapper ${rustToolchainBase}/bin/clippy-driver $out/bin/clippy-driver \
            #   --set CARGO "$out/bin/cargo" \
            #   --set RUSTC "$out/bin/rustc" \
            #   --set RUSTDOC "$out/bin/rustdoc"


            # makeWrapper ${rustToolchainBase}/bin/cargo-miri $out/bin/cargo-miri \
            #   --set CARGO "$out/bin/cargo" \
            #   --set RUSTC "$out/bin/rustc" \
            #   --set RUSTDOC "$out/bin/rustdoc"

          '';
        };

        coreBuild = with pkgs; [gcc cmake gnumake pkg-config clang clang-tools cppcheck clang-analyzer llvmPackages_22.libcxx llvmPackages_22.libunwind openssl openblas openmpi opencl-headers ocl-icd];
        cppPkgs = with pkgs; [boost eigen fmt];
        zigPkgs = with pkgs; [zig zls];
        futharkPkgs = with pkgs; [futhark];

        utilities = with pkgs; [
          valgrind
          lcov
          ccache
          bear
          doxygen
          perf
          hotspot
          gdb
          lldb
          cppcheck
          massif-visualizer
          hyperfine
        ];

        pythonEnv = pkgs.python3.withPackages (ps:
          with ps; [
            numpy
            pandas
            matplotlib
            seaborn
            pyarrow
            torch
          ]);

        rEnv = pkgs.rWrapper.override {
          packages = with pkgs.rPackages; [
            tidyverse
            lme4
            lmerTest
            brms
            rmarkdown
          ];
        };

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
            test = stdenv.runCommandCC "cmdstan-test" {inherit (self) cmdStan;} ''
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
          rustc = rustToolchain;
          cargo = rustToolchain;
        };

        treefmtEval = treefmt-nix.lib.evalModule pkgs ./treefmt.nix;

        rustPackage = naersk-lib.buildPackage {
          src = ./crates;
          cargoBuildOptions = opts:
            opts
            ++ [
              "--features"
              "ffi-backend"
              "--config=build.rustflags=[\"-Zautodiff=Enable\"]"
            ];
          nativeBuildInputs = [
            pkgs.pkg-config
          ];
          buildInputs = [
            pkgs.openblas
          ];
          cargoEnv = {
            "CXXFLAGS" = "-I${pkgs.eigen}/include/eigen3";
            "OPENBLAS_NUM_THREADS" = "1";
            "RUSTC_BOOTSTRAP" = "1";
          };
        };

        matmod = stdenv.mkDerivation {
          pname = "matmod";
          version = "0.1";
          dontUnpack = true;

          nativeBuildInputs = [pkgs.makeWrapper];

          installPhase = ''
            mkdir -p $out/bin

            makeWrapper ${rustPackage}/bin/app $out/bin/matmod \
              --set-default MATMOD_BACKEND dynamic
          '';
        };
      in {
        packages = {
          rustToolchain = rustToolchain;
          cmdStan = cmdStan;
          rustPackage = rustPackage;
          default = matmod;
        };
        apps.default = flake-utils.lib.mkApp {drv = self.packages.${system}.default;};

        formatter = treefmtEval.config.build.wrapper;

        checks = {
          cmdStan-test = cmdStan.passthru.tests.test;
          formatting = treefmtEval.config.build.check self;
        };

        # lazy devshell
        devShells = {
          default = pkgs.mkShell rec {
            hardeningDisable = ["all"];
            packages =
              [pythonEnv rEnv]
              ++ coreBuild
              ++ zigPkgs
              ++ cppPkgs
              ++ futharkPkgs
              ++ utilities;

            shellHook = ''
              echo "---------------------------------------"
              echo "you are in the flake's default devshell"

              export DCMAKE_EXPORT_COMPILE_COMMANDS=1

              if [ -x "$PWD/.nix-rust/bin/rustc" ]; then
                export PATH="$PWD/.nix-rust/bin:$PATH"
                export RUSTFLAGS="-L${pkgs.openblas}/lib -lopenblas -C target-cpu=native -C target-feature=+avx2,+fma -C link-arg=-Wl,-rpath,${pkgs.gcc.cc.lib}/lib -C link-arg=-Wl,-rpath,$PWD/.nix-rust/lib -Zautodiff=Enable --sysroot $PWD/.nix-rust"
                echo "local rust toolchain detected and loaded"
              else
                echo "rust toolchain is missing or not built"
                echo "   run:  nix build .#rustToolchain -o .nix-rust"
                echo "   then: exit and re-enter 'nix develop'"
              fi

              if [ -x "$PWD/.nix-cmdstan/bin/stanc" ]; then
                export PATH="$PWD/.nix-cmdstan/bin:$PATH"
                echo "local cmdStan detected and loaded"
              else
                echo "cmdStan is missing or not built."
                echo "run:  nix build .#cmdStan -o .nix-cmdstan"
              fi

              echo "---------------------------------------"
              exec zsh
            '';

            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath packages;
            LLDB_DEBUGSERVER_PATH = "${pkgs.lldb}/bin/lldb-server";
            CPLUS_INCLUDE_PATH = "crates/target/cxxbridge";

            # RUSTFLAGS = [
            #   "-L${pkgs.openblas}/lib"
            #   "-lopenblas"
            #   "-C target-cpu=native"
            #   "-C target-feature=+avx2,+fma"
            #   "-C link-arg=-Wl,-rpath,${pkgs.gcc.cc.lib}/lib"
            #   "-C link-arg=-Wl,-rpath,${enzyme}/lib"
            #   "-Zautodiff=Enable"
            # ];
          };

          full = pkgs.mkShell {
            inputsFrom = [self.devShells.${system}.default];
            packages = [rustToolchain cmdStan];
            # RUSTFLAGS = "-L${pkgs.openblas}/lib -lopenblas -C target-cpu=native -C target-feature=+avx2,+fma -C link-arg=-Wl,-rpath,${pkgs.gcc.cc.lib}/lib -C link-arg=-Wl,-rpath,${rustToolchain}/lib -Zautodiff=Enable";
          };
        };
      };
    };
}
