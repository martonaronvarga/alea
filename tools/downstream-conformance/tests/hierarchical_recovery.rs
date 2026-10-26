//! A downstream model survives durable frozen-state recovery without new warmup.
use alea::{
    ensemble::Execution,
    sampling::{NutsOptions, adapt::MetricKind},
    session::{
        Checkpoint, Kernel, OwnedDraw, SessionBuilder, Warmup,
        storage::{Discard, DrawReader, DurableSink, Observer, Sink},
    },
};
use alea_downstream_conformance::modeling::{Observation, hierarchy};
use std::{fs::File, path::PathBuf};
use tokio_util::sync::CancellationToken;

struct Directory(PathBuf);
impl Directory {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let path = std::env::temp_dir().join(format!(
            "alea-hierarchical-recovery-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        std::fs::create_dir(&path)?;
        Ok(Self(path))
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn settings() -> Result<SessionBuilder, Box<dyn std::error::Error>> {
    Ok(SessionBuilder::new(
        "hierarchical-logistic;events=2,6;trials=8,8;priors=1;build=test",
        4,
        vec![vec![0., -0.7, 0., 0.]],
    )
    .seed([73; 32])
    .draws(96)
    .kernel(Kernel::Nuts(NutsOptions::new(0.12, 7)?)))
}

#[test]
fn hierarchical_checkpoint_replays_draws_and_preserves_absolute_indices()
-> Result<(), Box<dyn std::error::Error>> {
    let data: Vec<_> = [2, 6]
        .into_iter()
        .enumerate()
        .flat_map(|(group, events)| {
            (0..8).map(move |i| Observation::Logistic {
                group,
                event: i < events,
            })
        })
        .collect();
    let directory = Directory::new()?;
    for (case, metric) in [MetricKind::Diagonal, MetricKind::Dense]
        .into_iter()
        .enumerate()
    {
        let warmup = Warmup::Covariance {
            iterations: 120,
            metric,
            target: 0.9.try_into()?,
            search: None,
        };
        let full = settings()?.warmup(warmup).build()?.run(
            |_| hierarchy(&data).into_target(),
            Execution::Sequential,
            &Discard,
        )?;
        assert!(full.chains[0].failure.is_none());
        assert_eq!(full.chains[0].completed, 96);

        let cancel = CancellationToken::new();
        let partial = settings()?
            .warmup(warmup)
            .cancellation(cancel.clone())
            .build()?
            .run(
                |_| hierarchy(&data).into_target(),
                Execution::Sequential,
                &Observer(|draw: &OwnedDraw| {
                    if draw.index == 31 {
                        cancel.cancel();
                    }
                    Ok(())
                }),
            )?;
        let prefix = &partial.chains[0];
        assert!(prefix.failure.is_none());
        assert!(prefix.cancelled);
        assert_eq!(prefix.completed, 32);
        assert_eq!(prefix.draws, full.chains[0].draws[..32]);
        let checkpoint = prefix.checkpoint.as_ref().ok_or("missing checkpoint")?;
        let checkpoint_path = directory.0.join(format!("{case}.checkpoint"));
        checkpoint.write_new(&checkpoint_path, 65_536)?;
        let decoded = Checkpoint::read(&checkpoint_path, 65_536)?;
        assert_eq!(&decoded, checkpoint);
        assert!(checkpoint.write_new(&checkpoint_path, 65_536).is_err());

        // Rebuild the model from data, not from retained graph or target scratch.
        // The builder has no warmup: restart restores the frozen geometry and step.
        let resumed = settings()?.restart(vec![decoded]).collect(false).build()?;
        let draws_path = directory.0.join(format!("{case}.draws"));
        let sink = DurableSink::create_new(&draws_path, resumed.provenance(), 1_000_000)?;
        let rest = resumed.run(
            |_| hierarchy(&data).into_target(),
            Execution::Sequential,
            &sink,
        )?;
        assert!(rest.chains[0].failure.is_none());
        assert_eq!(rest.chains[0].completed, 96);
        assert_eq!(rest.chains[0].warmup_completed, 0);
        assert!(rest.chains[0].draws.is_empty());
        assert_eq!(rest.chains[0].checkpoint, full.chains[0].checkpoint);
        let final_path = directory.0.join(format!("{case}.final"));
        let expected_final_path = directory.0.join(format!("{case}.expected-final"));
        rest.chains[0]
            .checkpoint
            .as_ref()
            .ok_or("missing resumed checkpoint")?
            .write_new(&final_path, 65_536)?;
        full.chains[0]
            .checkpoint
            .as_ref()
            .ok_or("missing full checkpoint")?
            .write_new(&expected_final_path, 65_536)?;
        assert_eq!(
            std::fs::read(final_path)?,
            std::fs::read(expected_final_path)?
        );

        // Compare serialized frames as well as f64 equality, preserving signed zeros.
        let expected_path = directory.0.join(format!("{case}.expected-draws"));
        let expected_sink =
            DurableSink::create_new(&expected_path, resumed.provenance(), 1_000_000)?;
        for draw in &full.chains[0].draws[32..] {
            expected_sink.draw(draw)?;
        }
        expected_sink.finish()?;
        assert_eq!(std::fs::read(&draws_path)?, std::fs::read(expected_path)?);
        assert_eq!(rest.diagnostics.chains[0].range, 32..96);

        let mut reader = DrawReader::new(File::open(&draws_path)?, 65_536)?;
        assert_eq!(reader.provenance(), resumed.provenance());
        let mut decoded_draws = Vec::new();
        while let Some(draw) = reader.next_draw()? {
            decoded_draws.push(draw);
        }
        assert_eq!(decoded_draws, full.chains[0].draws[32..]);
        let mut joined = prefix.draws.clone();
        joined.extend(decoded_draws);
        assert_eq!(joined, full.chains[0].draws);
    }
    Ok(())
}
