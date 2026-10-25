//! Run with cargo run --manifest-path tools/downstream-conformance/Cargo.toml --example modeling.
use alea::diagnostics::modern;
use alea::distributions::{
    modeling::ddm_upper_probability,
    scalar::{Bernoulli, Normal},
    wiener::{Boundary, WienerObservation},
};
use alea::math::buffer::OwnedBuffer;
use alea::sampling::{
    HmcOptions, Nuts,
    adapt::{HmcWarmup, WarmupOptions},
    config::TreeDepth,
};
use alea::{
    distribution::Sample,
    model::{
        assembly::CompiledModel,
        prediction::{NamedOutput, posterior_predict},
    },
};
use alea_downstream_conformance::modeling::{Observation, hierarchy, regression};
use rand::{SeedableRng, rngs::SmallRng};
fn run(name: &str, make: impl Fn() -> CompiledModel) -> Result<(), Box<dyn std::error::Error>> {
    let model = make();
    let coordinates = model.coordinates();
    let d = coordinates.len();
    let layout = model.layout().clone();
    let mut chains = Vec::new();
    for id in 0..4 {
        let target = make().into_target()?;
        let mut rng = SmallRng::seed_from_u64(14300 + id);
        let initial = OwnedBuffer::from_fn(d, |i| if d == 4 && i == 1 { -0.7 } else { 0.0 });
        let (chain, _) = HmcWarmup::new(
            &target,
            initial,
            HmcOptions::new(0.08, 1)?,
            WarmupOptions::new(512),
        )?
        .run(&mut rng)?;
        let mut chain = Nuts::from_hmc(chain, TreeDepth::try_from(7)?)?;
        let mut draws = Vec::new();
        for _ in 0..512 {
            let _ = chain.step(&mut rng)?;
            draws.push(chain.point().position().to_vec());
        }
        chains.push(draws);
    }
    for i in 0..d {
        let values: Vec<Vec<f64>> = chains
            .iter()
            .map(|chain| chain.iter().map(|q| q[i]).collect())
            .collect();
        let refs: Vec<&[f64]> = values.iter().map(Vec::as_slice).collect();
        let summary = modern::summarize(&refs)?;
        println!(
            "{name} unconstrained({}) mean={:.4} Rhat={:.4} bulk_ESS={:.0}",
            coordinates[i].name, summary.mean, summary.rhat, summary.bulk_ess
        );
    }
    let mut rng = SmallRng::seed_from_u64(14350);
    let generated = posterior_predict(
        &layout,
        chains.iter().flatten().map(Vec::as_slice),
        &mut rng,
        |p, rng| {
            if name == "normal regression" {
                Normal::new(p[0] + p[1], 0.7)?.sample(rng)
            } else {
                let effect = p[0] + p[1] * p[2];
                let probability = if name == "hierarchical DDM" {
                    ddm_upper_probability(1.2, 0.5, effect)?
                } else {
                    1.0 / (1.0 + (-effect).exp())
                };
                Ok(f64::from(Bernoulli::new(probability)?.sample(rng)?))
            }
        },
    )?;
    let output = NamedOutput::new(coordinates, layout, generated)?;
    println!(
        "{name}: {} posterior predictions, geometry blocks={:?}",
        output.values.generated.len(),
        output.blocks()
    );
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    run("normal regression", || {
        regression(&[(-1.0, -0.2), (0.0, 0.4), (1.0, 1.3)])
    })?;
    run("hierarchical logistic", || {
        hierarchy(&[
            Observation::Logistic {
                group: 0,
                event: true,
            },
            Observation::Logistic {
                group: 1,
                event: false,
            },
        ])
    })?;
    run("hierarchical DDM", || {
        hierarchy(&[
            Observation::Ddm {
                group: 0,
                event: WienerObservation {
                    rt: 0.6,
                    boundary: Boundary::Upper,
                },
            },
            Observation::Ddm {
                group: 1,
                event: WienerObservation {
                    rt: 0.9,
                    boundary: Boundary::Lower,
                },
            },
        ])
    })?;
    Ok(())
}
