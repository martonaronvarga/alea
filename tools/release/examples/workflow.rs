//! Independent consumer: native models, diagnostics, bounded output and restart.
use alea_core::{
    model::{
        ModelError,
        assembly::{AssemblyError, CompiledModel, ModelBuilder},
    },
    target::{EvaluationError, LogDensityGradient},
};
use alea_mcmc::{NutsOptions, adapt::MetricKind};
use alea_runtime::{
    ensemble::Execution,
    session::{
        Checkpoint, Diagnostics, Kernel, SessionBuilder, Status, Warmup,
        storage::{Discard, DrawReader, DurableSink, Observer},
    },
};
use serde_json::json;
use std::{fs::File, path::Path, time::Instant};
use tokio_util::sync::CancellationToken;

type Error = ModelError<AssemblyError>;

// y = intercept + slope*x + Normal(0,1), independent Normal(0,2) priors.
// Centered x makes the two analytic posterior coordinates independent.
fn regression() -> Result<CompiledModel, AssemblyError> {
    regression_data(&[
        (-2.0, -1.7),
        (-1.0, -0.9),
        (0.0, 0.1),
        (1.0, 1.1),
        (2.0, 2.2),
    ])
}

fn regression_data(data: &[(f64, f64)]) -> Result<CompiledModel, AssemblyError> {
    let mut b = ModelBuilder::new();
    let intercept = b.real("intercept")?.expr();
    let slope = b.real("slope")?.expr();
    for parameter in [intercept, slope] {
        let square = b.multiply(parameter, parameter)?;
        let scale = b.constant(-0.125)?;
        let prior = b.multiply(scale, square)?;
        b.factor(prior)?;
    }
    b.plate(data, |b, _, &(x, y)| {
        let x = b.constant(x)?;
        let linear = b.multiply(x, slope)?;
        let mean = b.add(intercept, linear)?;
        let minus = b.constant(-1.0)?;
        let negative = b.multiply(minus, mean)?;
        let y = b.constant(y)?;
        let residual = b.add(y, negative)?;
        let square = b.multiply(residual, residual)?;
        let half = b.constant(-0.5)?;
        b.multiply(half, square)
    })?;
    b.compile()
}

// Gamma(shape=3,rate=2), built in constrained precision coordinates. The
// parameter layout supplies the exponential-transform Jacobian exactly once.
fn precision() -> Result<CompiledModel, AssemblyError> {
    let mut b = ModelBuilder::new();
    let tau = b.positive("precision")?.expr();
    let log = b.log(tau)?;
    let two = b.constant(2.0)?;
    let shape = b.multiply(two, log)?;
    let minus_two = b.constant(-2.0)?;
    let rate = b.multiply(minus_two, tau)?;
    let posterior = b.add(shape, rate)?;
    b.factor(posterior)?;
    b.compile()
}

fn factory() -> Result<impl LogDensityGradient<Error = Error>, Error> {
    regression().map_err(model_error)?.into_target()
}

fn model_error(error: AssemblyError) -> Error {
    ModelError::Evaluation(EvaluationError::Model(error))
}

fn options() -> NutsOptions {
    NutsOptions::new(0.2, 8)
        .unwrap()
        .with_step_size_jitter(0.1)
        .unwrap()
        .with_position_turning(true)
}

fn warmup(fast: bool) -> Warmup {
    if fast {
        Warmup::GradientDiagonal {
            iterations: 512,
            target: 0.8.try_into().unwrap(),
            search: None,
            final_restart: false,
        }
    } else {
        Warmup::Covariance {
            iterations: 512,
            metric: MetricKind::Diagonal,
            target: 0.8.try_into().unwrap(),
            search: None,
        }
    }
}

fn spec(draws: usize, chains: usize) -> SessionBuilder {
    SessionBuilder::new(
        "candidate-regression-data-v1",
        2,
        vec![vec![0.0; 2]; chains],
    )
    .seed([73; 32])
    .draws(draws)
    .kernel(Kernel::Nuts(options()))
    .diagnostics(Diagnostics {
        window: 2048,
        maximum_rhat: 1.03,
        minimum_ess: 200.0,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = std::env::args()
        .nth(1)
        .ok_or("usage: workflow NEW_OUTPUT_DIRECTORY")?;
    let output = Path::new(&output);
    std::fs::create_dir(output)?;
    let mut models = Vec::new();
    for fast in [false, true] {
        let start = Instant::now();
        let kernel = if fast {
            options()
        } else {
            NutsOptions::new(0.2, 8)?
        };
        let result = spec(2048, 4)
            .kernel(Kernel::Nuts(kernel))
            .warmup(warmup(fast))
            .build()?
            .run(|_| factory(), Execution::Sequential, &Discard)?;
        assert_eq!(result.status, Status::Success, "regression diagnostics");
        let mut coordinates = Vec::new();
        for (coordinate, (mean, variance)) in [(0.8 / 5.25, 1.0 / 5.25), (9.8 / 10.25, 1.0 / 10.25)]
            .into_iter()
            .enumerate()
        {
            let draws: Vec<_> = result
                .chains
                .iter()
                .flat_map(|c| c.draws.iter().map(|d| d.position[coordinate]))
                .collect();
            let observed = draws.iter().sum::<f64>() / draws.len() as f64;
            let second = draws.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / draws.len() as f64;
            let diagnostics = result.diagnostics.diagnostics[coordinate].as_ref().unwrap();
            assert!((observed - mean).abs() < 0.05);
            assert!((second / variance - 1.0).abs() < 0.15);
            coordinates.push(json!({"mean":observed,"expected_mean":mean,"variance_about_expected_mean":second,
                "expected_variance":variance,"rhat":diagnostics.rhat,"bulk_ess":diagnostics.bulk_ess,"tail_ess":diagnostics.tail_ess}));
        }
        models.push(json!({"case":"regression","profile":if fast {"gradient-position-jitter"} else {"covariance"},
            "elapsed_ns":start.elapsed().as_nanos(),"chains":4,"warmup":512,"draws_per_chain":2048,"coordinates":coordinates}));
    }

    let gamma = SessionBuilder::new("candidate-gamma-3-2-v1", 1, vec![vec![0.0]; 4])
        .seed([81; 32])
        .draws(2048)
        .kernel(Kernel::Nuts(NutsOptions::new(0.2, 8)?))
        .warmup(warmup(false))
        .diagnostics(Diagnostics {
            window: 2048,
            maximum_rhat: 1.03,
            minimum_ess: 200.0,
        })
        .build()?
        .run(
            |_| precision().map_err(model_error)?.into_target(),
            Execution::Sequential,
            &Discard,
        )?;
    assert_eq!(gamma.status, Status::Success, "precision diagnostics");
    let draws: Vec<_> = gamma
        .chains
        .iter()
        .flat_map(|c| c.draws.iter().map(|d| d.position[0].exp()))
        .collect();
    let mean = draws.iter().sum::<f64>() / draws.len() as f64;
    let variance = draws.iter().map(|x| (x - 1.5).powi(2)).sum::<f64>() / draws.len() as f64;
    assert!((mean - 1.5).abs() < 0.10);
    assert!((variance / 0.75 - 1.0).abs() < 0.25);
    models.push(
        json!({"case":"gamma_precision","constrained_mean":mean,"expected_mean":1.5,
        "variance_about_expected_mean":variance,"expected_variance":0.75,"jacobian":"once"}),
    );

    let uninterrupted = spec(256, 1).warmup(warmup(true)).build()?.run(
        |_| factory(),
        Execution::Sequential,
        &Discard,
    )?;
    let cancel = CancellationToken::new();
    let partial = spec(256, 1)
        .warmup(warmup(true))
        .cancellation(cancel.clone())
        .build()?
        .run(
            |_| factory(),
            Execution::Sequential,
            &Observer(|draw: &alea_runtime::session::OwnedDraw| {
                if draw.index == 63 {
                    cancel.cancel();
                }
                Ok(())
            }),
        )?;
    assert_eq!(partial.status, Status::Cancelled);
    let path = output.join("chain.checkpoint");
    partial.chains[0]
        .checkpoint
        .as_ref()
        .ok_or("no checkpoint")?
        .write_new(&path, 65536)?;
    let saved = Checkpoint::read(&path, 65536)?;
    let session = spec(256, 1).restart(vec![saved]).collect(false).build()?;
    let path = output.join("continuation.alea");
    let sink = DurableSink::create_new(&path, session.provenance(), 1_000_000)?;
    let rest = session.run(|_| factory(), Execution::Sequential, &sink)?;
    assert!(
        rest.chains[0].draws.is_empty(),
        "durable streaming does not collect history"
    );
    let mut reader = DrawReader::new(File::open(&path)?, 65536)?;
    let mut durable = Vec::new();
    while let Some(draw) = reader.next_draw()? {
        durable.push(draw);
    }
    assert_eq!(durable, uninterrupted.chains[0].draws[64..]);
    assert_eq!(
        rest.chains[0].checkpoint,
        uninterrupted.chains[0].checkpoint
    );
    assert!(DurableSink::create_new(&path, session.provenance(), 1_000_000).is_err());

    // Predeclared bounded SBC sanity check: 32 prior-predictive datasets, 512
    // covariance warmup and 512 draws, four rank bins, chi-square limit 16.27.
    // One-chain diagnostics do not authorize a convergence claim.
    let mut data_rng =
        alea_runtime::ensemble::chain_rng([91; 32], alea_runtime::output::ChainId(0));
    let mut normal = alea_math::random::NormalGenerator::default();
    let mut ranks = Vec::new();
    let mut bins = [[0_usize; 4]; 2];
    for dataset in 0..32 {
        let truth = [
            2.0 * normal.sample(&mut data_rng),
            2.0 * normal.sample(&mut data_rng),
        ];
        let data: Vec<_> = [-2.0, -1.0, 0.0, 1.0, 2.0]
            .into_iter()
            .map(|x| (x, truth[0] + x * truth[1] + normal.sample(&mut data_rng)))
            .collect();
        let result = SessionBuilder::new(
            format!("regression-sbc-data-{dataset}-v1"),
            2,
            vec![vec![0.0; 2]],
        )
        .seed([dataset + 3; 32])
        .draws(512)
        .kernel(Kernel::Nuts(NutsOptions::new(0.2, 8)?))
        .warmup(warmup(false))
        .build()?
        .run(
            |_| regression_data(&data).map_err(model_error)?.into_target(),
            Execution::Sequential,
            &Discard,
        )?;
        assert_eq!(result.chains[0].completed, 512);
        assert!(result.chains[0].failure.is_none());
        let rank: Vec<_> = (0..2)
            .map(|c| {
                result.chains[0]
                    .draws
                    .iter()
                    .filter(|d| d.position[c] < truth[c])
                    .count()
            })
            .collect();
        for c in 0..2 {
            bins[c][rank[c] * 4 / 513] += 1;
        }
        ranks.push(json!({"dataset":dataset,"rank":rank,"truth":truth,"data":data}));
    }
    let chi: Vec<_> = bins
        .iter()
        .map(|b| {
            b.iter()
                .map(|&n| (n as f64 - 8.0).powi(2) / 8.0)
                .sum::<f64>()
        })
        .collect();
    let calibration = json!({"datasets":32,"warmup":512,"draws":512,"bins":bins,
                            "chi_square":chi,"limit":16.27,"raw_ranks":ranks});
    // Retain the complete experiment even when a future platform fails the gate.
    std::fs::write(
        output.join("calibration.json"),
        serde_json::to_vec_pretty(&calibration)?,
    )?;
    assert!(
        chi.iter().all(|&v| v < 16.27),
        "bounded SBC rank check: {bins:?}"
    );
    let report = json!({"schema_version":1,"models":models,"recovery":{
        "cancelled_prefix":64,"resumed_draws":durable.len(),"exact_draw_replay":true,
        "exact_checkpoint_replay":true,"existing_output_preserved":true,"checkpoint_format":2},
        "sbc":{"datasets":32,"warmup":512,"draws":512,"bins":bins,"chi_square":chi,"limit":16.27,"raw_ranks":ranks},
        "scope":"agent-executed independent consumer; no external-human feedback claimed"});
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{report}");
    Ok(())
}
