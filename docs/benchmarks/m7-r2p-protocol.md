# M7 randomized two-point completion protocol

This follow-up closes the algorithmic gap between Alea's original deterministic
WALNUTS-D evidence and the randomized two-point WALNUTS-R2P profile used in the
paper's funnel experiments. It does not alter or overwrite the locked October 14
deterministic result.

The benchmark uses the same implementation, funnel and 50-dimensional stochastic-
volatility targets, four seeds (`9001..9004`), 1,000 Fisher warmup transitions,
4,096 retained draws, tree depth eight, ten critical dyadic attempts, one initial
microstep and range tolerance 0.21 as the locked M7 comparison. It adds one method:
WALNUTS-R2P with probabilities 2/3 at the critical level and 1/3 at its next dyadic
successor, exact cumulative forward/reverse selection weights, and one uniform
±20% macro-interval jitter draw per orbit. Time and fused target calls include
construction, warmup and retained sampling; no seed or stopped extension is removed.

R2P is eligible for a later promotion study only if every observable has rank/folded
R-hat at most 1.01 and bulk/tail ESS at least 400, and if minimum bulk ESS/second is
no worse than Fisher NUTS on both targets. One run can reject promotion but cannot
establish the roadmap's reproducibility requirement. Covariance warmup and
multinomial NUTS remain the defaults regardless of this follow-up.

## Result

All 30 target/method/observable rows pass the diagnostic thresholds. The aggregate
minimums are:

| target/method | max R-hat | min bulk ESS | min tail ESS | total seconds | calls | min bulk ESS/s | min bulk ESS/call |
|---|---:|---:|---:|---:|---:|---:|---:|
| funnel / Fisher NUTS | 1.00242 | 3,800.5 | 5,055.1 | 0.0912 | 106,300 | 41,664 | 0.0358 |
| funnel / Fisher WALNUTS-D | 1.00050 | 5,393.3 | 6,837.6 | 0.0738 | 620,768 | 73,097 | 0.00869 |
| funnel / Fisher WALNUTS-R2P | 1.00085 | 5,989.5 | 7,623.0 | 0.0858 | 730,734 | 69,799 | 0.00820 |
| stochastic volatility / Fisher NUTS | 1.00087 | 9,247.9 | 10,557.7 | 3.1226 | 297,344 | 2,962 | 0.0311 |
| stochastic volatility / Fisher WALNUTS-D | 1.00041 | 9,887.3 | 9,839.4 | 6.0750 | 1,672,054 | 1,628 | 0.00591 |
| stochastic volatility / Fisher WALNUTS-R2P | 1.00049 | 10,539.2 | 10,730.4 | 7.5415 | 2,142,040 | 1,398 | 0.00492 |

R2P reduces reverse/refinement stops from 3,094 to 1,797 on the funnel and from
1,795 to 1,007 on stochastic volatility, but increases calls by 18% and 28%
relative to WALNUTS-D. It clears the funnel timing threshold in this run but is
about 53% below Fisher NUTS minimum ESS/second on stochastic volatility. It
therefore rejects promotion under the predeclared rule.

Raw artifact: [`m7-r2p-2026-10-15.csv`](m7-r2p-2026-10-15.csv), SHA-256
`253696a66cd2be9a803aa1f20c38e1aab5c8b43a0a8a5f03fa985e55cc4279b5`.
