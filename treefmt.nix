{pkgs, ...}: {
  projectRootFile = "flake.nix";
  # Preserve archived, historical reference and vendored bytes verbatim.
  settings.global.excludes = [
    "archive/**"
    "crates/alea-ffi/vendor/**"
    "research/ddm-prototypes/**"
    "research/cxx-prototype/**"
    "research/inference-prototypes/**"
    "tools/stan-reference/**"
  ];

  programs = {
    alejandra.enable = true;
    rustfmt.enable = true;
    clang-format.enable = true;
    zig.enable = true;
  };

  settings.formatter = {
    futhark = {
      command = "${pkgs.futhark}/bin/futhark";
      options = ["fmt"];
      includes = ["*.fut"];
    };
  };
}
