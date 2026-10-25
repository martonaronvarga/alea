# Alea licensing

Alea separates the reusable Rust inference libraries from its native cubature
integration. Each Cargo manifest states the license for that distributed package.

| Packages | License |
|---|---|
| `alea-math`, `alea-core`, `alea-mcmc`, `alea-runtime`, `alea-autodiff`, `alea-smc` | `MIT OR Apache-2.0` |
| `alea`, `alea-ffi`, `alea-distributions`, `alea-cli` | `GPL-3.0-or-later` |

Original Alea code outside the GPL package trees is available under either
the [MIT license](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at the recipient's
option. That grant covers only work Alea's contributors are authorized to license.
Existing third-party notices and explicit file/directory licenses take precedence
for those materials. The GPL package trees use [GPLv3 or later](LICENSE-GPL).
The public `alea` facade is distributed under GPL-3.0-or-later; its optional
`distributions` feature includes the cubature dependency.

## Native dependency boundary

`alea-ffi` statically compiles Steven G. Johnson's cubature. Its upstream license
is **GPL-2.0-or-later**, including identified HIntLib/GSL portions. Alea's local
patches retain the original notices in the generated C source. Upstream remains
GPL-2.0-or-later; Alea selects the permitted GPLv3 route for the combined native
packages. `alea-distributions` depends on this boundary, and `alea-cli` depends on
distributions. Those packages are therefore not offered as wholly permissive
distributions.

The six permissive libraries have no production dependency on cubature. Their
integration tests can use GPL-covered distribution fixtures; a compiled test or
benchmark executable has its own dependency closure. A permissive source-file
license does not waive the conditions of linked dependencies. This split follows
the [upstream cubature terms](https://github.com/stevengj/cubature/blob/master/COPYING.md)
and the [Apache/GPLv3 compatibility guidance](https://www.apache.org/licenses/GPL-compatibility).

## Reused work and attribution

Retain [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and every upstream notice
that accompanies redistributed source or binaries. In particular, the adapted
Stan Math Wiener test fixtures retain their BSD-3-Clause attribution. PosteriorDB
models/data/reference files, BridgeStan and nuts-rs retain their own notices.
Algorithm citations in method documents are separate from copyright permission
for reused source. This repository license does not grant rights over papers,
external implementations, data or trademarks.

Cargo dependencies keep their upstream licenses. Source `.crate` packages do not
bundle their registry dependencies. A binary distributor must include notices
and corresponding source obligations for the **actual linked dependency closure**,
including any selected optional backend. Fetch/build dependencies from the pinned
lock when reproducing a source release; do not interpret the Alea package license
as an inventory of every dependency license.

The development-only `pprof` flamegraph path includes `inferno` under CDDL-1.0.
The reference inference libraries do not use it in production. The prepared release
artifacts contain source packages and a separately tested reference consumer;
they do not ship profiling executables. Profiling executables that also link the
GPL native path need a separate compatibility review before binary distribution.

## Packaging checks

Every package includes its full license texts and third-party notices. The release
metadata check verifies the declared split against the production dependency
graph. The package verifier checks the **actual `.crate` archive contents** against
the canonical texts, so a missing notice or a permissively labelled cubature
package fails preparation. Source archives include all root licensing documents.

Source verification and maintainer signing apply to distributed artifacts.
