# Reporting security issues

For a suspected memory-safety problem, unsafe foreign callback, malicious input
issue or sensitive data disclosure, contact the maintainer privately at
martonaronvarga@gmail.com. Include the pinned source revision, platform/toolchain,
enabled features and a minimal reproducer without secrets or private datasets.
Do not post sensitive details in a public issue before coordinating disclosure.

Alea is currently unpublished and pre-alpha, with no supported stable version.
Security fixes target the current source; future stable maintenance windows are
defined in [the release policy](docs/releases.md). No response-time guarantee or
security audit is implied. Accepted reports receive a tracked fix and coordinated
advisory when appropriate. A release candidate with unresolved unsoundness,
correctness, data-loss or schema-compatibility defects cannot graduate.

Ordinary numerical failures, diagnostics and feature requests can use the public
[issue tracker](https://github.com/martonaronvarga/alea/issues). Preserve failing
inputs, seed identities and full diagnostic outcomes rather than filtering them.
