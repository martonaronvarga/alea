# Deliberately independent of future workspace flake.lock updates: fixture pin.
let
  source = builtins.fetchTree {
    type = "github";
    owner = "NixOS";
    repo = "nixpkgs";
    rev = "68d8aa3d661f0e6bd5862291b5bb263b2a6595c9";
    narHash = "sha256-vPKLpjhIVWdDrfiUM8atW6YkIggCEKdSAlJPzzhkQlw=";
  };
  pkgs = import source {system = builtins.currentSystem;};
in
  # Both packages ship a top-level docs/ tree. Prefer JAX's documentation files
  # when linking the environment; the blackjax/ implementation is unaffected.
  pkgs.python3.withPackages (ps: [(pkgs.lib.lowPrio ps.blackjax)])
