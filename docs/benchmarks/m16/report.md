# Manifest-driven validation report

Manifest: `b44a67966ce24ac0ceff8ca5c6e9c25e7895aaa41754ffed846be1188167dad7`

Promotion decision: **not_eligible**. Numerical defaults and maturity ledgers were not changed.

| Case | Engine | Completed jobs | Quality passes | Median process s | Peak RSS KiB |
| --- | --- | ---: | ---: | ---: | ---: |
| banana_tune | alea | 3/3 | 3/3 | 0.03261 | 3372 |
| banana_tune | blackjax | 3/3 | 3/3 | 8.42785 | 408788 |
| banana_tune | nuts-rs | 3/3 | 3/3 | 0.03238 | 3156 |
| banana_tune | stan | 3/3 | 3/3 | 0.41545 | 16060 |
| correlated_heldout | alea | 3/3 | 3/3 | 0.06472 | 3316 |
| correlated_heldout | blackjax | 3/3 | 3/3 | 8.98039 | 415680 |
| correlated_heldout | nuts-rs | 3/3 | 3/3 | 0.06464 | 3152 |
| correlated_heldout | stan | 3/3 | 3/3 | 0.51550 | 16296 |
| eight_schools_centered | alea | 3/3 | 0/3 | 0.36521 | 3200 |
| eight_schools_centered | blackjax | 3/3 | 0/3 | 13.68266 | 436276 |
| eight_schools_centered | nuts-rs | 0/3 | 0/3 | 0.00842 | 3040 |
| eight_schools_centered | stan | 3/3 | 0/3 | 1.51739 | 17420 |
| eight_schools_noncentered | alea | 3/3 | 3/3 | 0.16480 | 3316 |
| eight_schools_noncentered | blackjax | 3/3 | 3/3 | 12.08424 | 433460 |
| eight_schools_noncentered | nuts-rs | 3/3 | 3/3 | 0.11458 | 3168 |
| eight_schools_noncentered | stan | 3/3 | 3/3 | 0.86605 | 17560 |
| funnel_heldout | alea | 3/3 | 0/3 | 0.21483 | 3316 |
| funnel_heldout | blackjax | 3/3 | 0/3 | 9.88120 | 420964 |
| funnel_heldout | nuts-rs | 3/3 | 0/3 | 0.16492 | 3140 |
| funnel_heldout | stan | 3/3 | 0/3 | 0.86588 | 16572 |
| gaussian_tune | alea | 3/3 | 3/3 | 0.03261 | 3356 |
| gaussian_tune | blackjax | 3/3 | 3/3 | 8.42793 | 403032 |
| gaussian_tune | nuts-rs | 3/3 | 3/3 | 0.03235 | 3136 |
| gaussian_tune | stan | 3/3 | 3/3 | 0.36514 | 15728 |
| hierarchical_ddm_boundary | alea | 3/3 | 3/3 | 0.11469 | 3272 |
| hierarchical_ddm_boundary | blackjax | 3/3 | 3/3 | 11.58162 | 438916 |
| hierarchical_ddm_boundary | nuts-rs | 3/3 | 3/3 | 0.11458 | 3124 |
| hierarchical_ddm_boundary | stan | 3/3 | 3/3 | 0.66569 | 16440 |
| hierarchical_normal | alea | 3/3 | 3/3 | 0.06474 | 3336 |
| hierarchical_normal | blackjax | 3/3 | 3/3 | 10.47989 | 422616 |
| hierarchical_normal | nuts-rs | 3/3 | 3/3 | 0.11489 | 3140 |
| hierarchical_normal | stan | 3/3 | 3/3 | 0.61522 | 16188 |
| ill_conditioned_heldout | alea | 3/3 | 0/3 | 0.31552 | 3276 |
| ill_conditioned_heldout | blackjax | 3/3 | 0/3 | 9.93160 | 409748 |
| ill_conditioned_heldout | nuts-rs | 3/3 | 0/3 | 0.31480 | 3176 |
| ill_conditioned_heldout | stan | 3/3 | 0/3 | 1.06673 | 16180 |
| latent_gaussian | alea | 3/3 | 3/3 | 0.03256 | 3212 |
| latent_gaussian | blackjax | 3/3 | 3/3 | 8.97884 | 411604 |
| latent_gaussian | nuts-rs | 3/3 | 3/3 | 0.06458 | 3156 |
| latent_gaussian | stan | 3/3 | 3/3 | 0.46556 | 16312 |
| logistic_heldout | alea | 3/3 | 3/3 | 0.11468 | 3244 |
| logistic_heldout | blackjax | 3/3 | 3/3 | 13.43629 | 453404 |
| logistic_heldout | nuts-rs | 3/3 | 3/3 | 0.11475 | 3132 |
| logistic_heldout | stan | 3/3 | 3/3 | 0.56546 | 16060 |
| stochastic_volatility | alea | 3/3 | 3/3 | 0.11459 | 3196 |
| stochastic_volatility | blackjax | 3/3 | 3/3 | 11.48461 | 439604 |
| stochastic_volatility | nuts-rs | 3/3 | 3/3 | 0.16485 | 3152 |
| stochastic_volatility | stan | 3/3 | 3/3 | 0.81544 | 17580 |
| wiener_drift | alea | 3/3 | 3/3 | 0.06463 | 3340 |
| wiener_drift | blackjax | 3/3 | 3/3 | 8.87798 | 397524 |
| wiener_drift | nuts-rs | 3/3 | 3/3 | 0.03242 | 3108 |
| wiener_drift | stan | 3/3 | 3/3 | 0.61566 | 15416 |

All failures, exclusions, phase/counter availability, uncertainty and calibration details are retained in `report.json`. Failed quality gates never enter a favorable speed ratio.

## Excluded claims

- No new sampler/default or S3/S4 promotion
- Fixed-discard identity policy is not tuned adaptive performance
- One physical host does not establish cross-host speed
- Finite SBC is a regression gate, not universal calibration
- Wiener infers drift only; DDM prediction covers boundary events
- Unavailable upstream allocation/gradient counters are null with reasons
- Stan native phase timings include CSV output; process-wall efficiency is the common timing scope
