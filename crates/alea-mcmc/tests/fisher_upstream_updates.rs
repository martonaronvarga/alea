//! Executed pinned nuts-rs update traces, not whole-sampler equivalence.
#![cfg(feature = "faer")]
use alea_math::{fisher::fit_low_rank, metric::EuclideanMetric};

fn numbers(s: &str) -> Vec<f64> {
    s.split(';').map(|v| v.parse().unwrap()).collect()
}
fn close(a: f64, b: f64) {
    assert!(
        a.is_finite() && (a - b).abs() <= 1e-8 * (1.0 + b.abs()),
        "{a} != {b}"
    );
}

#[test]
fn pinned_collectors_switches_and_installed_metric_actions_are_replayed() {
    let text = include_str!("fixtures/fisher-nuts-updates.csv");
    assert!(text.starts_with("# nuts-rs a762aae513bf7bb5ccb27ffc9a938a923fc837ec"));
    let mut rows = text.lines().filter(|line| !line.starts_with('#')).skip(1);
    let mut cases = 0;
    for dim in [2, 8] {
        let mut observations = Vec::<(Vec<f64>, Vec<f64>)>::new();
        let mut split = 0;
        let mut version = -1;
        for event in 1..=16 {
            let q: Vec<_> = (0..dim)
                .map(|j| (((event + 1) * (j + 3) + event * event) % 17) as f64 / 4.0 - 2.0)
                .collect();
            let score: Vec<_> = (0..dim)
                .map(|j| {
                    -0.7 * q[j] + 0.3 * q[(j + 1) % dim] + ((event * (j + 1) + 3) % 7) as f64 / 10.0
                })
                .collect();
            let good = event % 5 != 0;
            if good {
                observations.push((q, score));
            }
            let switched = event == 6 || event == 12;
            if switched {
                observations.drain(..split);
                split = observations.len();
            }
            if observations.len() < 3 {
                continue;
            }
            version += 1;
            let row: Vec<_> = rows.next().expect("complete trace").split(',').collect();
            assert_eq!(row.len(), 15);
            assert_eq!(row[0].parse::<usize>().unwrap(), event);
            assert_eq!(row[1].parse::<usize>().unwrap(), dim);
            assert_eq!(row[2].parse::<bool>().unwrap(), good);
            assert_eq!(row[3].parse::<bool>().unwrap(), switched);
            assert_eq!(row[4].parse::<usize>().unwrap(), observations.len());
            assert_eq!(row[5].parse::<usize>().unwrap(), observations.len() - split);
            assert_eq!(row[14].parse::<i32>().unwrap(), version);
            let positions = numbers(row[6]);
            let scores = numbers(row[7]);
            assert_eq!(
                positions,
                observations
                    .iter()
                    .flat_map(|x| x.0.iter().copied())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                scores,
                observations
                    .iter()
                    .flat_map(|x| x.1.iter().copied())
                    .collect::<Vec<_>>()
            );
            let metric = fit_low_rank(
                &positions,
                &scores,
                &numbers(row[8]),
                row[9].parse().unwrap(),
                row[10].parse().unwrap(),
                dim,
            )
            .unwrap();
            assert_eq!(metric.rank(), row[11].parse::<usize>().unwrap());
            close(metric.log_det(), row[13].parse().unwrap());
            let expected = numbers(row[12]);
            assert_eq!(expected.len(), dim * dim);
            for j in 0..dim {
                let mut input = vec![0.0; dim];
                input[j] = 1.0;
                let mut out = vec![0.0; dim];
                metric.velocity(&input, &mut out).unwrap();
                for i in 0..dim {
                    close(out[i], expected[i * dim + j]);
                }
            }
            cases += 1;
        }
    }
    assert_eq!(cases, 28);
    assert!(rows.next().is_none());
}
