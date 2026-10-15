# Executed WALNUTS reference

This probe executes the upstream Walnutpie sampler at commit
`5710b0334ba69cf0caa35fc02a2338b61d756953` on a two-dimensional standard
Gaussian. Two fixed `std::mt19937_64` streams must reproduce the checked trace
byte for byte on the pinned toolchain. The runner also verifies the existing
nuts-rs checkout at `a762aae513bf7bb5ccb27ffc9a938a923fc837ec`, the version used
by Alea's M4F and M5 reference work.

From the repository root:

```sh
git clone https://github.com/flatironinstitute/walnutpie.git \
  target/walnutpie-reference-upstream
git -C target/walnutpie-reference-upstream checkout \
  5710b0334ba69cf0caa35fc02a2338b61d756953
# tools/nuts-reference/run_upstream.py can provision the pinned nuts-rs checkout.
nix develop .#stable --command \
  python3 tools/walnuts-reference/run.py --output target/walnutpie-trace.csv
```

This fixture proves that the cited upstream source was built and run; Alea does
not claim bitwise equality with a C++ sampler using a different RNG/distribution
implementation. Alea's independent scalar leapfrog tests cover the local map,
while sampling and diagnostic tests cover stationary output.

There is one deliberate mathematical difference. The JMLR construction bounds
the Hamiltonian range over every microstep in a macroblock. The pinned Walnutpie
code checks the two endpoint energies and describes the full proof as expected.
Alea implements the paper's stronger full-range condition and the deterministic
forward/reverse support correction, so this upstream trace is provenance rather
than an Alea golden trajectory.
