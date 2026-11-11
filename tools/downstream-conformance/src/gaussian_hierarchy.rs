//! Application qualification consumer: unknown-scale Gaussian hierarchy.
//! Uses native laws, session NUTS, diagnostics and sealed analysis interchange.
use alea::{
    analysis::{
        AnalysisSpec, AnalysisWriter, ArrayValues, ChainStatus, DataArray, DataAxis, ObservedData,
        Provenance,
    },
    diagnostics::modern,
    distribution::Sample,
    distributions::{modeling::normal, scalar::Normal},
    ensemble::Execution,
    model::{
        ModelError,
        assembly::{
            AssemblyError, NativeModel, ModelBuilder, PointwiseLikelihood, PredictiveUnit,
        },
    },
    output::{ChainId, Draw},
    sampling::{NutsOptions, adapt::MetricKind},
    session::{Diagnostics, SamplerConfig, SessionBuilder, Status, Warmup, storage::Discard},
    target::EvaluationError,
};
use rand::{SeedableRng, rngs::SmallRng};
use serde_json::{Value, json};
use std::{collections::BTreeMap, error::Error};
use tokio_util::sync::CancellationToken;

/// All planned workflow outcomes, including empty cancelled/failed chains.
pub const CASES: [&str; 3] = ["complete", "cancelled", "initial_transform"];
/// Fixed observations; group identity and row order are explicit.
pub const DATA: [[f64; 3]; 2] = [[-1., -0.5, 0.], [0.5, 1., 1.5]];
const WARMUP: usize = 1000;
const DRAWS: usize = 2000;
const PREDICTIONS: [&str; 6] = [
    "theta0",
    "theta1",
    "existing_y",
    "new_a_y0",
    "new_a_y1",
    "new_b_y0",
];
const OBSERVABLES: [&str; 15] = [
    "mu",
    "tau",
    "tau2",
    "theta0",
    "theta1",
    "theta0_2",
    "theta1_2",
    "mu2",
    "theta01",
    "existing_y",
    "existing_y2",
    "new_y",
    "new_y2",
    "same_group_diff2",
    "different_group_diff2",
];

fn units() -> Vec<PredictiveUnit> {
    (0..2)
        .map(|g| {
            let observations: Vec<_> = (0..3).map(|i| format!("group{g}/row{i}")).collect();
            PredictiveUnit {
                name: format!("existing_group{g}"),
                factors: observations.clone(),
                observations,
                conditioned_on: vec!["mu".into(), "tau".into(), format!("z{g}")],
                integrated_out: vec![],
            }
        })
        .collect()
}

/// Noncentered hierarchy: mu~N(0,1), tau~half-N(0,1), z_g~N(0,1),
/// y_gi~N(mu+tau*z_g,1). All factors have explicit statistical roles.
/// # Errors
/// Preserves native model assembly failures.
pub fn model() -> Result<NativeModel, AssemblyError> {
    let mut b = ModelBuilder::new();
    let mu = b.real("mu")?;
    let tau = b.positive("tau")?;
    let z = [b.real("z0")?, b.real("z1")?];
    let zero = b.constant(0.)?;
    let one = b.constant(1.)?;
    let prior = normal(&mut b, mu.expr(), zero, one)?;
    b.prior("mu prior", prior)?;
    let prior = normal(&mut b, tau.expr(), zero, one)?;
    b.prior("tau prior", prior)?;
    let half = b.constant(std::f64::consts::LN_2)?;
    b.prior("tau half-normal normalization", half)?;
    for (g, z) in z.iter().enumerate() {
        let prior = normal(&mut b, z.expr(), zero, one)?;
        b.prior(&format!("z{g} prior"), prior)?;
        let deviation = b.multiply(tau.expr(), z.expr())?;
        let mean = b.add(mu.expr(), deviation)?;
        for (i, y) in DATA[g].into_iter().enumerate() {
            let y = b.constant(y)?;
            let term = normal(&mut b, y, mean, one)?;
            b.likelihood(&format!("group{g}/row{i}"), term)?;
        }
    }
    b.compile()?.select_posterior()
}

fn error_chain(error: &(dyn Error + 'static)) -> Vec<String> {
    let mut result = vec![error.to_string()];
    let mut source = error.source();
    while let Some(error) = source {
        result.push(error.to_string());
        source = error.source();
    }
    result
}

/// Executes the declared four-chain plan and retains its actual analysis artifact,
/// failures and diagnostics. Completion is not statistical qualification: the
/// independent Python consumer applies the predeclared reference/diagnostic gates.
/// # Errors
/// Rejects unknown cases, setup/admission, generated-quantity or export failures.
/// Native chain failures remain in the report, including chains with no draws.
pub fn report(case: &str) -> Result<Value, Box<dyn Error>> {
    if !CASES.contains(&case) {
        return Err("unknown Gaussian hierarchy case".into());
    }
    let initial = vec![
        vec![-0.5, -0.7, -0.2, 0.3],
        vec![0., 0., 0., 0.],
        vec![0.5, 0.4, 0.3, -0.2],
        vec![1., -0.3, -0.3, 0.2],
    ];
    let initial: Vec<_> = initial
        .into_iter()
        .map(|mut q| {
            if case == "initial_transform" {
                q[1] = -1000.;
            }
            q
        })
        .collect();
    let cancel = CancellationToken::new();
    if case == "cancelled" {
        cancel.cancel();
    }
    let result = SessionBuilder::new(
        "gaussian-hierarchy:half-normal-tau:fixed-residual-scale:v1",
        4,
        initial.clone(),
    )
    .seed([83; 32])
    .draws(DRAWS)
    .sampler(SamplerConfig::Nuts(NutsOptions::new(0.1, 8)?))
    .warmup(Warmup::Covariance {
        iterations: WARMUP,
        metric: MetricKind::Diagonal,
        target: 0.9.try_into()?,
        search: None,
    })
    .diagnostics(Diagnostics {
        window: DRAWS,
        maximum_rhat: 1.05,
        minimum_ess: 200.,
    })
    .cancellation(cancel)
    .build()?
    .run(
        |_| {
            model()
                .map_err(|e| ModelError::Evaluation(EvaluationError::Model(e)))?
                .into_target()
        },
        Execution::Sequential,
        &Discard,
    )?;
    let status = match result.status {
        Status::Success => "success",
        Status::DiagnosticWarning => "diagnostic_warning",
        Status::Cancelled => "cancelled",
        Status::NumericalExhaustion => "numerical_exhaustion",
        Status::BackendFailure => "backend_failure",
        _ => return Err("unknown session status".into()),
    };
    let mut spec = AnalysisSpec::from_likelihood(
        PointwiseLikelihood::new(model()?, units())?,
        Provenance {
            model: "gaussian-hierarchy:half-normal-tau:fixed-residual-scale:v1".into(),
            data: "two-groups:[-1,-.5,0],[.5,1,1.5]:v1".into(),
            source: "public-native-gaussian-hierarchy:v1".into(),
            method: "session NUTS; diagonal covariance warmup".into(),
            settings: BTreeMap::from([
                ("seed".into(), "[83;32]; logical chain streams".into()),
                ("prediction_seed".into(), "83001+chain; SmallRng".into()),
                ("case".into(), case.into()),
            ]),
        },
        (0..4).map(ChainId).collect(),
    )?;
    spec.save_unconstrained = true;
    spec.predictions = PREDICTIONS.map(String::from).to_vec();
    let axes = vec![DataAxis {
        name: "observation".into(),
        coordinates: (0..2)
            .flat_map(|g| (0..3).map(move |i| format!("group{g}/row{i}")))
            .collect(),
    }];
    spec.observed_data = vec![ObservedData {
        array: DataArray {
            name: "y".into(),
            axes: axes.clone(),
            values: ArrayValues::Real(DATA.into_iter().flatten().collect()),
            unit: None,
        },
        observation_dimension: "observation".into(),
    }];
    spec.constant_data = vec![DataArray {
        name: "group".into(),
        axes,
        values: ArrayValues::Unsigned(vec![0, 0, 0, 1, 1, 1]),
        unit: None,
    }];
    let mut writer = AnalysisWriter::new(Vec::new(), spec)?;
    let standard = Normal::new(0., 1.)?;
    let mut chains = Vec::new();
    let mut statuses = Vec::new();
    let mut values = Vec::new();
    for c in &result.chains {
        let mut rng = SmallRng::seed_from_u64(83001 + c.chain.0);
        let mut chain_values = Vec::new();
        for d in &c.draws {
            let q = &d.position;
            let tau = q[1].exp();
            let theta = [q[0] + tau * q[2], q[0] + tau * q[3]];
            let existing = Normal::new(theta[0], 1.)?.sample(&mut rng)?;
            // Two observations share ONE new random effect, but independent
            // observation errors. A different new group gets its own effect.
            let new_a = q[0] + tau * standard.sample(&mut rng)?;
            let new_b = q[0] + tau * standard.sample(&mut rng)?;
            let a0 = Normal::new(new_a, 1.)?.sample(&mut rng)?;
            let a1 = Normal::new(new_a, 1.)?.sample(&mut rng)?;
            let b0 = Normal::new(new_b, 1.)?.sample(&mut rng)?;
            writer.write_draw(
                &Draw {
                    schema_version: d.schema_version,
                    chain: d.chain,
                    phase: d.phase,
                    index: d.index,
                    position: q,
                    transition: &(),
                    stats: d.stats,
                },
                &[theta[0], theta[1], existing, a0, a1, b0],
            )?;
            chain_values.push([
                q[0],
                tau,
                tau * tau,
                theta[0],
                theta[1],
                theta[0] * theta[0],
                theta[1] * theta[1],
                q[0] * q[0],
                theta[0] * theta[1],
                existing,
                existing * existing,
                a0,
                a0 * a0,
                (a0 - a1).powi(2),
                (a0 - b0).powi(2),
            ]);
        }
        let chain_status = if let Some(e) = &c.failure {
            ChainStatus::Failed(e.to_string())
        } else if c.completed == DRAWS {
            ChainStatus::Completed
        } else {
            ChainStatus::Interrupted(
                if c.cancelled {
                    "cancelled"
                } else {
                    "incomplete"
                }
                .into(),
            )
        };
        let step = match c.checkpoint.as_ref().map(|p| p.sampler()) {
            Some(SamplerConfig::Nuts(o)) => Some(o.step_size().value()),
            None => None,
            _ => return Err("unexpected hierarchy sampler".into()),
        };
        chains.push(json!({"id":c.chain.0,"warmup_completed":c.warmup_completed,"completed":c.completed,"cancelled":c.cancelled,
            "error_chain":c.failure.as_ref().map(|e| error_chain(e)),"numerical_rejections":c.numerical_rejections,"final_step_size":step}));
        statuses.push((c.chain, chain_status));
        values.push(chain_values);
    }
    let summaries = if values.iter().all(|v| v.len() == DRAWS) {
        let mut summaries = serde_json::Map::new();
        for (i, name) in OBSERVABLES.into_iter().enumerate() {
            let data: Vec<Vec<_>> = values
                .iter()
                .map(|c| c.iter().map(|v| v[i]).collect())
                .collect();
            let refs: Vec<_> = data.iter().map(Vec::as_slice).collect();
            let summary = match modern::summarize(&refs) {
                Ok(s) => {
                    json!({"mean":s.mean,"rhat":s.rhat,"bulk_ess":s.bulk_ess,"tail_ess":s.tail_ess,"mean_ess":s.mean_ess,"mcse_mean":s.mcse_mean})
                }
                Err(e) => json!({"error_chain":error_chain(&e)}),
            };
            summaries.insert(name.into(), summary);
        }
        Value::Object(summaries)
    } else {
        Value::Null
    };
    let mut manifest = Vec::new();
    let csv = writer.finish(&mut manifest, &statuses)?;
    Ok(
        json!({"format":"gaussian-hierarchy-workflow","schema_version":1,"case":case,
        "plan":{"chains":4,"warmup":WARMUP,"draws":DRAWS,"initial":initial,"seed":vec![83;32],"step_size":0.1,"max_depth":8,"target_acceptance":0.9,"metric":"diagonal","search":null},
        "session_status":status,"sink_error":result.sink_failure.as_ref().map(|e|error_chain(e)),
        "chains":chains,"summaries":summaries,"manifest":serde_json::from_slice::<Value>(&manifest)?,"csv":String::from_utf8(csv)?}),
    )
}
