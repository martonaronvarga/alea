# Executed Stan adaptation reference

`trace.cpp` is an original harness calling the pinned Stan 2.38 public
`stepsize_adaptation`, `var_adaptation` and `covar_adaptation` classes. It reads no
Alea code, binary or output. The source store, exact compiler command, binary and
header hashes are in `provenance.json`. Stan's sources and notices remain at their
pinned upstream locations; see the repository's third-party notices.

Run `python3 tools/stan-adaptation-reference/check.py` for offline fixture checks.
The 400 diagonal/dense recurrence observations use the same declared deterministic
inputs as the independently executed BlackJAX fixture consumed by Rust tests.
Their largest scaled difference is approximately 3.44e-14. The harness supplies a
shared smoothed-step reset at covariance boundaries to isolate the recurrences;
full Stan NUTS instead performs a new metric-dependent step search there. This is
not whole-workflow equivalence.

The 1711 schedule observations retain four differences from BlackJAX/Alea: fresh
Stan class instances do not emit the terminal window flag at lengths 20, 21, 149,
and emit an additional flag at index 99 of length 151. No rows were changed to
force agreement. The harness calls public setup/learning methods without adding a
restart after setup; this describes that bounded class experiment. It does not
establish behavior for every Stan service entry point. The broad 512 and focused
1000 budgets have matching schedules in these fixtures. Alea retains its existing
BlackJAX-compatible short-budget schedule.

To rebuild, run the compiler command in `provenance.json` from the repository
root, then invoke `target/stan-adaptation-trace schedule` and `trace` into fresh
files. Compare the output with the frozen fixtures. Normal CI needs no C++ build
or Stan installation for these checks.
