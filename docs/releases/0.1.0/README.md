# Alea 0.1.0 release review

Status: preparing; not published and not yet S4. The first stable release covers
`alea-math`, `alea-core`, `alea-mcmc` and `alea-runtime`, under MIT OR Apache-2.0.
All public items and features enter the compatibility baseline at publication.
The numerical maturity ledger and exactness ledger remain authoritative: research
methods stay experimental, approximate methods stay approximate, and RMHMC's
exactness obligations remain open. No default algorithm changes.

## Changes since the previous candidate

- Ordinary owning modules and curated preludes replace `reference` namespaces.
  No aliases or deprecated compatibility paths are retained for unpublished APIs.
- Wiener7 gains locally log-scaled quadrature for extreme tails and visible
  convergence failures. Wiener4/5 gain reusable contiguous batch storage and
  chain-local cached models. Direct batching remains the default. These changes
  are in the companion distributions source; that crate is not promoted to S4.
- The [release policy](../../releases.md) establishes a 0.1.x compatibility window,
  12-month maintenance windows, MSRV and deprecation rules. Experimental labels
  grant no exception to the public Rust API promise.

The complete source includes the other workspace crates for reproducibility;
including their sources does not graduate or publish those crates. Optional
Enzyme/native backends and design-only particle contracts retain their limitations.

## Required evidence

A completed review must bind all results to the exact candidate source and retain:

1. Passing native Linux x86_64/aarch64, macOS x86_64/aarch64 and Windows x86_64
   MSVC workflows at MSRV 1.95 and stable, plus pinned feature, Miri, sanitizer,
   property and operational-soak results. Older-revision CI is historical only.
2. Independent fresh-project modeling/calibration and durable recovery from the
   source archive and the actual unpacked `.crate` files.
3. Actual public-API extraction and a reviewed breaking diff from the unpublished
   candidate. The first published source becomes the future semver baseline.
4. Two byte-identical source archives, reproduction from an unpacked archive,
   complete licenses/notices, package hashes and a verified maintainer signature.
5. Successful artifact publication and a public compatibility/security policy.
   Before this step, report preparation rather than S4 graduation.

Registry publication has an additional gate: current integration fixtures form
cyclic dev dependencies through unpublished distributions/core. The package verifier
uses patches only in its independent consumer and records that fact; it is not a
successful registry dry run. Registry names, dependency availability and a real
registry-backed consumer must be verified before advertising crates.io installation.
A signed source release can be published independently of registry packages.

## Verification and publication

From the exact reviewed checkout, with an independently trusted signer:

```sh
nix develop .#stable --command python3 tools/release/metadata.py --candidate
nix develop .#stable --command python3 tools/release/packages.py /new/packages
nix develop .#stable --command python3 tools/release/source.py /new/alea-0.1.0.tar.gz --signing-key /path/to/maintainer-key.pub
nix develop .#stable --command python3 tools/release/verify.py /new/alea-0.1.0.tar.gz --allowed-signers /path/to/trusted-source-signers --identity git@martonaronvarga.dev
```

Publish the accepted archive, digest, detached signature, package archives and
review evidence together. The archive verifier must use a trust record obtained
independently of the download. Keep the reviewed source digest unchanged; fixes
require a new candidate and the relevant gates. Retain unsuccessful attempts.

For the first publication's intentional break, run
`bash tools/release/semver.sh --initial /accepted/unpublished/source`.
This checks the actual reviewed endpoints and does not claim compatibility with
the unpublished API. Subsequent releases omit `--initial` and compare with the
published source. The first-publication exception is restricted to version 0.1.0.

The final evidence record will identify the candidate commit, signed source and
package digests, same-source CI URLs, reviewed API scope, signing identity and
publication location. No placeholder URL or earlier candidate is publication proof.
