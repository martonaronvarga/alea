//! Public-facade consumers: native batches reach sampling/analysis and curvature.
use alea::{
    analysis::{AnalysisSpec, AnalysisWriter, ChainStatus, Provenance},
    distributions::modeling::{normal, regression::*},
    math::buffer::OwnedBuffer,
    model::assembly::{NativeModel, ModelBuilder, PointwiseLikelihood, PredictiveUnit},
    output::{ChainId, DiagnosticTransition, Draw, Phase, SCHEMA_VERSION},
    sampling::{
        HmcOptions, Nuts,
        adapt::{HmcWarmup, WarmupOptions},
        config::TreeDepth,
    },
};
use rand::{SeedableRng, rngs::SmallRng};
use std::{
    collections::BTreeMap,
    io::Write,
    process::{Command, Stdio},
};

fn model(family: usize) -> NativeModel {
    let mut b = ModelBuilder::new();
    let beta = b.real("beta").unwrap();
    let prior = normal(&mut b, beta, 0., 2.).unwrap();
    b.prior("beta prior", prior).unwrap();
    let data = RegressionData::new(1, vec![1.; 4], vec![0., 0.5, 1., 1.5]).unwrap();
    let likelihood = match family {
        0 => {
            let sigma = b.positive("sigma").unwrap();
            let prior = normal(&mut b, sigma, 0., 1.).unwrap();
            b.prior("sigma prior", prior).unwrap();
            let ln2 = b.constant(std::f64::consts::LN_2).unwrap();
            b.prior("half normal", ln2).unwrap();
            data.linear_map(&[beta.expr()])
                .unwrap()
                .normal(&sigma)
                .unwrap()
                .sum_log_prob(&mut b)
        }
        1 => data
            .with_observations(vec![false, true, false, true])
            .unwrap()
            .linear_map(&[beta.expr()])
            .unwrap()
            .bernoulli_logit()
            .unwrap()
            .sum_log_prob(&mut b),
        _ => data
            .with_observations(vec![0_u64, 1, 2, 3])
            .unwrap()
            .linear_map(&[beta.expr()])
            .unwrap()
            .poisson_log()
            .unwrap()
            .sum_log_prob(&mut b),
    }
    .unwrap();
    b.likelihood("responses", likelihood).unwrap();
    b.compile().unwrap()
}
#[test]
fn four_chain_native_batches_export_grouped_rows_and_pass_independent_reader_density_checks() {
    for family in 0..3 {
        let batch = model(family).batch_likelihood().unwrap();
        let assignments = vec![batch.units()[..2].to_vec(), batch.units()[2..].to_vec()];
        let units = (0..2)
            .map(|i| PredictiveUnit {
                name: format!("group{i}"),
                factors: vec!["responses".into()],
                observations: (2 * i..2 * i + 2).map(|j| format!("y{j}")).collect(),
                conditioned_on: vec!["beta".into()],
                integrated_out: vec![],
            })
            .collect();
        let pointwise = PointwiseLikelihood::from_batch(batch, units, assignments).unwrap();
        let spec = AnalysisSpec::from_likelihood(
            pointwise,
            Provenance {
                model: format!("native-{family}"),
                data: "four-rows".into(),
                source: "native_verticals.rs".into(),
                method: "NUTS".into(),
                settings: BTreeMap::new(),
            },
            (0..4).map(ChainId).collect(),
        )
        .unwrap();
        let mut writer = AnalysisWriter::new(Vec::new(), spec).unwrap();
        for chain in 0..4 {
            let target = model(family)
                .select_posterior()
                .unwrap()
                .into_target()
                .unwrap();
            let mut rng = SmallRng::seed_from_u64(90210 + chain);
            let (hmc, _) = HmcWarmup::new(
                &target,
                OwnedBuffer::from_fn(if family == 0 { 2 } else { 1 }, |_| 0.),
                HmcOptions::new(0.08, 1).unwrap(),
                WarmupOptions::new(128),
            )
            .unwrap()
            .run(&mut rng)
            .unwrap();
            let mut nuts = Nuts::from_hmc(hmc, TreeDepth::try_from(6).unwrap()).unwrap();
            for index in 0..16 {
                let transition = nuts.step(&mut rng).unwrap();
                let draw = Draw {
                    schema_version: SCHEMA_VERSION,
                    chain: ChainId(chain),
                    phase: Phase::Sampling,
                    index,
                    position: nuts.point().position(),
                    stats: transition.diagnostic_stats(),
                    transition: &(),
                };
                writer.write_draw(&draw, &[]).unwrap();
            }
        }
        let mut manifest = Vec::new();
        let csv = writer
            .finish(
                &mut manifest,
                &(0..4)
                    .map(|i| (ChainId(i), ChainStatus::Completed))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let payload = serde_json::json!({"family":family,"manifest":serde_json::from_slice::<serde_json::Value>(&manifest).unwrap(),"csv":String::from_utf8(csv).unwrap()});
        let code = r#"import sys,json,io,math
sys.path.insert(0,sys.argv[1]); import analysis
x=json.load(sys.stdin); a=analysis.load_stream(x['manifest'],io.BytesIO(x['csv'].encode()))
chains,indices,rows=a.rectangular(); assert chains==[0,1,2,3] and indices==list(range(16))
assert len(a.records)==64
for record in a.records:
 v=record.values; beta=v['parameter:beta']; values=[]
 for i in range(4):
  if x['family']==0:
   sigma=v['parameter:sigma']; lp=-math.log(sigma)-.5*math.log(2*math.pi)-.5*((i*.5-beta)/sigma)**2
  elif x['family']==1:
   z=-beta if i%2 else beta; lp=-(max(z,0)+math.log1p(math.exp(-abs(z))))
  else: lp=i*beta-math.exp(beta)-math.lgamma(i+1)
  values.append(lp)
 for j in range(2):
  actual=v['log_likelihood:group'+str(j)]; expected=sum(values[2*j:2*j+2])
  assert abs(actual-expected)<1e-11*(1+abs(expected)),(actual,expected)
assert a.manifest['predictive_units'][1]['observations']==['y2','y3']
"#;
        let mut child = Command::new("python3")
            .args(["-c", code, concat!(env!("CARGO_MANIFEST_DIR"), "/python")])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(payload.to_string().as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[cfg(feature = "faer")]
#[test]
fn native_glm_fit_replays_exactly_into_information_and_exports_explicit_covariance() {
    use alea::{
        analysis::write_fit,
        math::information::InformationOptions,
        model::prediction::NamedOutput,
        optimize::{
            LbfgsOptions, NamedFit, ObjectiveDescription, ObjectiveKind, OptimizationTermination,
            information::observed_information, maximize,
        },
    };
    for poisson in [false, true] {
        // One unconstrained coefficient, finite attained MLE and identified design.
        let target = if poisson {
            GlmLikelihood::poisson_log(
                RegressionData::new(1, vec![1.; 4], vec![1, 2, 3, 2]).unwrap(),
            )
            .unwrap()
        } else {
            GlmLikelihood::bernoulli_logit(
                RegressionData::new(1, vec![1.; 4], vec![false, true, true, true]).unwrap(),
            )
            .unwrap()
        };
        let provenance = Provenance {
            model: format!("glm-{poisson}"),
            data: "saturated-four".into(),
            source: "native_verticals.rs".into(),
            method: "MLE".into(),
            settings: BTreeMap::new(),
        };
        let result = maximize(
            &target,
            &[0.],
            ObjectiveDescription {
                kind: ObjectiveKind::Likelihood,
                measure: "real coefficient likelihood without chart Jacobian".into(),
                model: provenance.model.clone(),
                data: provenance.data.clone(),
                source: provenance.source.clone(),
            },
            LbfgsOptions::new(100, 1000).unwrap(),
        )
        .unwrap();
        assert_eq!(result.termination, OptimizationTermination::ScoreTolerance);
        let mut builder = ModelBuilder::new();
        let coefficient = builder.real("beta").unwrap();
        builder.likelihood("metadata", coefficient.expr()).unwrap();
        let model = builder.compile().unwrap();
        let metadata = NamedOutput::new(model.coordinates(), model.layout().clone(), ()).unwrap();
        let fit = NamedFit::new(metadata, result).unwrap();
        let info = observed_information(
            &target,
            &fit,
            InformationOptions {
                relative_eigenvalue_cutoff: 1e-12,
                relative_symmetry_tolerance: 64. * f64::EPSILON,
            },
        )
        .unwrap();
        let covariance = info.constrained_covariance().values[0];
        let expected = if poisson { 1. / 8. } else { 4. / 3. };
        assert!((covariance - expected).abs() < 1e-8);
        let mut bytes = Vec::new();
        write_fit(&mut bytes,&fit.with_assessment(serde_json::json!({"kind":"observed_likelihood_information", "covariance":[covariance],
            "assumptions":"identified fixed design, finite attained MLE; no interval coverage claim"})),&provenance,|a|Ok(a.clone())).unwrap();
        let report: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(
            report
                .to_string()
                .contains("observed_likelihood_information")
        );
    }
}
