# Reporting security issues

For a suspected memory-safety problem, unsafe foreign callback, malicious input
issue or sensitive data disclosure, contact the maintainer privately at
martonaronvarga@gmail.com. Include the pinned source revision, platform/toolchain,
enabled features and a minimal reproducer without secrets or private datasets.
Do not post sensitive details in a public issue before coordinating disclosure.

The first stable release is in preparation; no stable artifact has been published.
Until publication, fixes target the current source. Publication activates the
12-month maintenance windows in [the release policy](docs/releases.md), including
correctness/security fixes for the current compatible line and its predecessor. No response-time guarantee or
security audit is implied. Accepted reports receive a tracked fix and coordinated
advisory when appropriate. A release candidate with unresolved unsoundness,
correctness, data-loss or schema-compatibility defects cannot graduate.

Ordinary numerical failures, diagnostics and feature requests can use the public
[issue tracker](https://github.com/martonaronvarga/alea/issues). Preserve failing
inputs, seed identities and full diagnostic outcomes rather than filtering them.
