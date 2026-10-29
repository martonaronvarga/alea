//! Actual fits -> named artifacts -> an independent standard-library Python reader.
use alea::{
    analysis::{
        AnalysisSpec, AnalysisWriter, ArrayValues, ChainStatus, DataArray, DataAxis, ObservedData,
        PosteriorPredictive, Provenance,
    },
    distribution::Sample,
    distributions::{
        modeling::{bernoulli_logit, normal},
        scalar::{Bernoulli, Normal},
    },
    math::buffer::OwnedBuffer,
    model::assembly::{ModelBuilder, PointwiseLikelihood, PredictiveUnit},
    output::{ChainId, DiagnosticTransition, Draw, Phase, SCHEMA_VERSION},
    sampling::{
        HmcOptions, Nuts,
        adapt::{HmcWarmup, WarmupOptions},
        config::TreeDepth,
    },
};
use rand::{SeedableRng, rngs::SmallRng};
use std::{collections::BTreeMap, fs::File, process::Command};

#[test]
fn gaussian_and_logistic_fits_round_trip_physical_draws_and_joint_units() {
    for logistic in [false, true] {
        let mut writer = None;
        for chain in 0..2 {
            let mut b = ModelBuilder::new();
            let mu = b.real("mu").unwrap();
            let tau = b.positive("tau").unwrap();
            let z = [b.real("z0").unwrap(), b.real("z1").unwrap()];
            let zero = b.constant(0.).unwrap();
            let one = b.constant(1.).unwrap();
            let prior = normal(&mut b, mu.expr(), zero, one).unwrap();
            b.prior("mu prior", prior).unwrap();
            let minus = b.constant(-1.).unwrap();
            let prior = b.multiply(minus, tau.expr()).unwrap();
            b.prior("tau prior", prior).unwrap();
            let mut units = Vec::new();
            for (group, z) in z.iter().enumerate() {
                let prior = normal(&mut b, z.expr(), zero, one).unwrap();
                b.prior(&format!("z{group} prior"), prior).unwrap();
                let dev = b.multiply(tau.expr(), z.expr()).unwrap();
                let eta = b.add(mu.expr(), dev).unwrap();
                let mut factors = Vec::new();
                for row in 0..3 {
                    let term = if logistic {
                        bernoulli_logit(&mut b, row > group, eta).unwrap()
                    } else {
                        let y = b.constant((row as f64 - group as f64) * 0.3).unwrap();
                        normal(&mut b, y, eta, one).unwrap()
                    };
                    let name = format!("group {group}, row {row}");
                    b.likelihood(&name, term).unwrap();
                    factors.push(name);
                }
                units.push(PredictiveUnit {
                    name: format!("group \"{group}\"\nλ"),
                    observations: factors.clone(),
                    factors,
                    conditioned_on: vec!["mu".into(), "tau".into(), format!("z{group}")],
                    integrated_out: vec![],
                });
            }
            let target = b
                .compile()
                .unwrap()
                .select_posterior()
                .unwrap()
                .into_target()
                .unwrap();
            let mut rng = SmallRng::seed_from_u64(777 + chain);
            let mut rows = Vec::new();
            {
                let (hmc, _) = HmcWarmup::new(
                    &target,
                    OwnedBuffer::from_fn(4, |_| 0.),
                    HmcOptions::new(0.08, 1).unwrap(),
                    WarmupOptions::new(128),
                )
                .unwrap()
                .run(&mut rng)
                .unwrap();
                let mut nuts = Nuts::from_hmc(hmc, TreeDepth::try_from(6).unwrap()).unwrap();
                for _ in 0..32 {
                    let event = nuts.step(&mut rng).unwrap();
                    rows.push((nuts.point().position().to_vec(), event.diagnostic_stats()));
                }
            }
            if writer.is_none() {
                let pointwise = PointwiseLikelihood::new(target.into_model(), units).unwrap();
                let provenance = Provenance {
                    model: format!(
                        "hierarchical-{}:inline-v1",
                        if logistic { "logistic" } else { "gaussian" }
                    ),
                    data: "inline-six-observations:v1".into(),
                    source: "conformance-source:analysis.rs".into(),
                    method: "NUTS".into(),
                    settings: BTreeMap::from([
                        ("warmup".into(), "128".into()),
                        ("seed".into(), "777+chain".into()),
                        ("prediction_seed".into(), "13+chain".into()),
                        ("retained_draws".into(), "32".into()),
                    ]),
                };
                let mut spec = AnalysisSpec::from_likelihood(
                    pointwise,
                    provenance,
                    vec![ChainId(0), ChainId(1)],
                )
                .unwrap();
                spec.save_unconstrained = true;
                spec.predictions = vec!["mean,\"λ\"".into()];
                spec.predictions
                    .extend((0..6).map(|i| format!("replicate{i}")));
                let axes = vec![DataAxis {
                    name: "observation".into(),
                    coordinates: (0..2)
                        .flat_map(|group| {
                            (0..3).map(move |row| format!("group {group}, row {row}"))
                        })
                        .collect(),
                }];
                let values = if logistic {
                    ArrayValues::Boolean(
                        (0..2)
                            .flat_map(|group| (0..3).map(move |row| row > group))
                            .collect(),
                    )
                } else {
                    ArrayValues::Real(
                        (0..2)
                            .flat_map(|group| {
                                (0..3).map(move |row| (row as f64 - group as f64) * 0.3)
                            })
                            .collect(),
                    )
                };
                spec.observed_data = vec![ObservedData {
                    array: DataArray {
                        name: "y".into(),
                        axes: axes.clone(),
                        values,
                        unit: None,
                    },
                    observation_dimension: "observation".into(),
                }];
                spec.constant_data = vec![DataArray {
                    name: "subject".into(),
                    axes,
                    values: ArrayValues::Unsigned(vec![0, 0, 0, 1, 1, 1]),
                    unit: None,
                }];
                spec.posterior_predictive = vec![PosteriorPredictive {
                    variable: "y".into(),
                    columns: (0..6).map(|i| format!("replicate{i}")).collect(),
                }];
                writer = Some(AnalysisWriter::new(Vec::new(), spec).unwrap());
            }
            let mut prediction_rng = SmallRng::seed_from_u64(13 + chain);
            for (i, (q, stats)) in rows.into_iter().enumerate() {
                let draw = Draw {
                    schema_version: SCHEMA_VERSION,
                    chain: ChainId(chain),
                    phase: Phase::Sampling,
                    index: 500 + i,
                    position: &q,
                    transition: &(),
                    stats,
                };
                let mut predictions = vec![q[0]];
                for group in 0..2 {
                    let eta = q[0] + q[1].exp() * q[2 + group];
                    for _ in 0..3 {
                        let value = if logistic {
                            f64::from(
                                Bernoulli::new(1. / (1. + (-eta).exp()))
                                    .unwrap()
                                    .sample(&mut prediction_rng)
                                    .unwrap(),
                            )
                        } else {
                            Normal::new(eta, 1.)
                                .unwrap()
                                .sample(&mut prediction_rng)
                                .unwrap()
                        };
                        predictions.push(value);
                    }
                }
                writer
                    .as_mut()
                    .unwrap()
                    .write_draw(&draw, &predictions)
                    .unwrap();
            }
        }
        let dir =
            std::env::temp_dir().join(format!("alea-analysis-{}-{logistic}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let manifest = dir.join("manifest.json");
        let csv = dir.join("draws.csv");
        let bytes = writer
            .unwrap()
            .finish(
                File::create(&manifest).unwrap(),
                &[
                    (ChainId(0), ChainStatus::Completed),
                    (ChainId(1), ChainStatus::Completed),
                ],
            )
            .unwrap();
        std::fs::write(&csv, bytes).unwrap();
        let code = r#"import sys, struct
sys.path.insert(0, sys.argv[1])
import analysis
a = analysis.load(sys.argv[2], sys.argv[3])
chains, indices, rows = a.rectangular()
assert chains == [0, 1] and indices == list(range(500, 532))
assert len(a.records) == 64
for r in a.records:
    v = r.values
    assert struct.pack('>d', v['parameter:mu']) == struct.pack('>d', v['unconstrained:0'])
    assert v['parameter:mu'] == v['prediction:mean,"λ"']
    assert v['parameter:tau'] > 0
    assert v['stat:divergent'] in (True, False)
    assert len([k for k in v if k.startswith('log_likelihood:')]) == 2
assert a.manifest['predictive_units'][1]['observations'][2] == 'group 1, row 2'
assert a.manifest['observed_data'][0]['array']['axes'][0]['coordinates'][5] == 'group 1, row 2'
import os
if os.environ.get('ALEA_ANALYSIS_EXTERNAL') == '1':
    import conversion, xarray as xr, numpy as np
    from pathlib import Path
    tree = conversion.to_datatree(a)
    assert 'energy' not in tree['sample_stats'].data_vars
    assert 'initial_energy' in tree['sample_stats'].data_vars
    assert tree['posterior_predictive']['y'].shape == (2,32,6)
    assert tree['log_likelihood']['likelihood'].shape == (2,32,2)
    assert tree['constant_data']['subject'].dtype == np.dtype('uint64')
    path = Path(sys.argv[2]).with_suffix('.nc')
    conversion.write_netcdf(tree, path)
    with xr.open_datatree(path, engine='h5netcdf') as restored:
        np.testing.assert_array_equal(restored['posterior']['mu'].values, tree['posterior']['mu'].values)
        np.testing.assert_array_equal(restored['log_likelihood']['likelihood'].values, tree['log_likelihood']['likelihood'].values)
    report = conversion.loo_report(tree)
    assert report['pareto_k'].shape == (2,)
    assert report.attrs['n_samples'] == 64
    report.to_netcdf(path.with_name('loo.nc'), engine='h5netcdf')
    import subprocess
    r_code = r'''args <- commandArgs(TRUE)
source(args[1]); x <- read_alea(args[2], args[3], args[4])
grid <- alea_rectangular(x)
stopifnot(nrow(x$posterior) == 64L, identical(grid$draw, as.character(500:531)))
stopifnot(identical(names(x$posterior), c("mu","tau","z0","z1")))
stopifnot(length(x$manifest$observed_data[[1]]$array$values) == 6L)
stopifnot(all(!is.na(x$log_likelihood[[1]])))
'''
    r = subprocess.run(['Rscript', '-e', r_code, str(Path(sys.argv[1]).parent/'r/analysis.R'), sys.argv[2], sys.argv[3], str(Path(sys.argv[1])/'analysis.py')], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
"#;
        let python = std::env::var_os("ALEA_PYTHON").unwrap_or_else(|| "python3".into());
        let result = Command::new(python)
            .args(["-c", code])
            .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/python"))
            .arg(&manifest)
            .arg(&csv)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
