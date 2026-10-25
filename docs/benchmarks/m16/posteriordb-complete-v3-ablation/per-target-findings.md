# Per-target workflow observations

Before/after entries compare sealed version2 with this experiment. They describe whole workflows, not an isolated cutoff effect. The separate regression/AR-K ablation and identical-phase one-step probes isolate those questions more closely.

Times are per-job medians. R-hat and ESS use unconstrained coordinates; mean/spread discrepancies use available empirical PosteriorDB reference parameters. A completed run is not a convergence certificate. Undefined quantities stay absent.

## arK-arK

AR-K separates startup geometry adaptation from retained trajectory cost; its starts also appear in the controlled ablation.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 1.007 → 1.005; minimum bulk ESS 989.2 → 1114; maximum mean error/reference SD 0.04259 → 0.04387; maximum spread-ratio error 0.05467 → 0.04399.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_legacy_warmup | 3/3 | 3.876 | 3.248 | 1.007 | 989.2 | 0.04259 | 0.05467 |
| nuts_cov_diag | 3/3 | 5.62 | 2.652 | 1.005 | 1114 | 0.04387 | 0.04399 |

## earnings-earn_height

The pilot must learn coordinate scales before retention; difficult regression starts are directly checked in the ablation.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 4.567 → 1.007; minimum bulk ESS 4.272 → 534.5; maximum mean error/reference SD 48.97 → 0.04146; maximum spread-ratio error 0.9999 → 0.05819.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_legacy_warmup | 3/3 | 1.049 | 0.4974 | 4.567 | 4.272 | 48.97 | 0.9999 |
| nuts_cov_diag | 3/3 | 8.462 | 3.84 | 1.007 | 534.5 | 0.04146 | 0.05819 |

## sblri-blr

The pilot must learn coordinate scales before retention; difficult regression starts are directly checked in the ablation.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 1.019 → 1.017; minimum bulk ESS 240.9 → 331.1; maximum mean error/reference SD 16.26 → 0.1215; maximum spread-ratio error 157.1 → 0.04322.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_legacy_warmup | 3/3 | 0.119 | 0.06464 | 1.019 | 240.9 | 16.26 | 157.1 |
| nuts_cov_diag | 3/3 | 0.2001 | 0.0533 | 1.017 | 331.1 | 0.1215 | 0.04322 |
