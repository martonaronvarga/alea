# Pre-migration contract sketches

Original declarations and advisory dispatch from `50b66bb`, retained as design
inputs rather than exported placeholder algorithms. `NUTS` had configuration but
no transitions; particle-filter and PMCMC files were empty. `policy.rs` selected
descriptive text, not a running algorithm (its RMHMC/ChEES conflation is not a
validated recommendation). Other files contain unused extension/proposal/family
contracts. The supported dual-averaging implementation now lives in
`alea-mcmc::adapt`; these old trait declarations never drove its implementation.

No legacy package names or compatibility APIs are compiled from this directory.
