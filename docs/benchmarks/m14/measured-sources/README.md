# Preserved M14 inputs

These are the original, hash-verified M14 workspace manifest and parameter-layout
source. M17 adds repository/registry-version metadata and missing-documentation
enforcement to their current counterparts. The measured bytes are retained here
so historical M14 measurements continue to identify their actual inputs.

`tools/test_m14.py` checks the original digests from the unchanged `evidence.json`,
then separately verifies that current manifest changes are precisely the reviewed
release metadata and that current layout changes contain only documentation and
formatting. Numerical code, dependency versions and historical results are not
rewritten. Further changes require a new explicit compatibility review.
