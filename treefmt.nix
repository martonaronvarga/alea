{pkgs, ...}: {
  projectRootFile = "flake.nix";
  # Preserve archived experiments and vendored submodule bytes verbatim.
  settings.global.excludes = ["archive/**" "crates/ffi/vendor/**"];

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
