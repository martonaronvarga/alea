# Version 3 smoke descriptive benchmark

56 planned jobs: {'completed': 51, 'failed': 5}. Every failure, timeout and unsupported configuration is retained. No promotion score or success gate is computed.

Read [the findings](findings.md), [per-target observations](per-target-findings.md), [interactive report](report.html), [paired baseline observations](baseline-pairs.csv), [failures](failures.csv), and [verification](verification.json). The exact manifest, raw draws, partial outputs, sources and preparation artifacts are archived in the checksummed bundle segments.

The collection reader, current worktree reader and archived reader reproduced the identical report offline. The two earlier interrupted/diagnostic v3 attempts remain separate on dusk and are not pooled. The short [smoke experiment](../posteriordb-complete-v3-smoke/README.md) is separate pipeline evidence. RMHMC remains exactness-pending; MCLMC remains approximate.

Reproduce the supplementary tables with `python3 analysis.py RUN_DIRECTORY OUTPUT_DIRECTORY --legacy-report ../posteriordb-v2/report.json`. The output directory must exist with no prior table files.
