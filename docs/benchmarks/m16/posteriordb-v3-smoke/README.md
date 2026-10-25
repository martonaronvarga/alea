# Version 3 pipeline smoke evidence

56 jobs on dusk: **51 completed, 5 failed**. Two posteriors, 28 configurations,
four chains, one repetition, 128 warmup plus 64 retained draws. This short run
checks the repaired workflows and evidence pipeline; it does not establish
convergence or competitive performance. The broad and longer focused experiments
use separate source-sealed protocols and records.

Both ChEES cases and all four Alea raw/PSIS Pathfinder cases complete. Four RMHMC
jobs fail the unchanged strict symmetry contract despite using AD third
derivatives. The remaining failure is CmdStan Pathfinder initialization on
earnings/height. All outcomes, partial outputs, actual work, seeds, compiled
artifacts, preparation records and source snapshots remain in the bundle.

[Interactive report](report.html), [summary](summary.csv), [failures](failures.csv),
[verification](verification.json), [method rationale](../../../native-warmup-and-research-workflows.md).

The bundle segments and every archived job/source hash were checked after transfer.
Both current and archived report readers reproduced the identical report offline.
The manifest is `3da0df67e3ff077501163f4fffea2fc0f0c04a076f2eb512e91b789f7e79f134`.
The two earlier v3 attempts are preserved on dusk as interrupted diagnostic runs;
they are not pooled with this experiment. No score or promotion decision is made.
