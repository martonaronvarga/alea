# Reuse nixpkgs' release-pinned Enzyme source, with the same LLVM as rustc.
# Compatibility with std::autodiff still needs an execution test (roadmap M2).
{
  pkgs,
  llvmPkgs,
}:
pkgs.enzyme.override {llvmPackages = llvmPkgs;}
