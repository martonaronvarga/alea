//! Executed, version-pinned BlackJAX evidence; no Python needed in ordinary CI.
use alea_math::metric::EuclideanMetric;
use alea_mcmc::adapt::{DualAveraging, MetricKind, OnlineCovariance, WarmupStage, WindowSchedule};

#[test]
fn schedules_match_executed_blackjax_1_5() {
    let mut rows = include_str!("fixtures/blackjax-schedule.csv")
        .lines()
        .filter(|s| !s.starts_with('#'));
    assert_eq!(rows.next(), Some("length,index,slow,end"));
    let mut count = 0;
    let mut schedule = WindowSchedule::new(0);
    let mut length = 0;
    for row in rows {
        let f: Vec<usize> = row.split(',').map(|x| x.parse().unwrap()).collect();
        if f[0] != length {
            assert_eq!(schedule.len(), 0);
            length = f[0];
            schedule = WindowSchedule::new(length);
        }
        assert_eq!(f[1], length - schedule.len());
        let expected = if f[2] == 1 {
            WarmupStage::Slow {
                window_end: f[3] == 1,
            }
        } else if f[1]
            < if f[0] < 20 {
                f[0]
            } else if f[0] < 150 {
                f[0] * 15 / 100
            } else {
                75
            }
        {
            WarmupStage::InitialFast
        } else {
            WarmupStage::FinalFast
        };
        assert_eq!(schedule.next(), Some(expected), "{row}");
        count += 1;
    }
    assert_eq!(schedule.len(), 0);
    assert_eq!(count, 1711);
}

#[test]
fn covariance_dual_averaging_and_window_resets_match_blackjax_trace() {
    let rows: Vec<_> = include_str!("fixtures/blackjax-adaptation.csv")
        .lines()
        .filter(|s| !s.starts_with('#'))
        .collect();
    assert_eq!(rows[0], "kind,index,x,y,acceptance,step,c00,c01,c11");
    assert_eq!(rows.len(), 401);
    for (case, kind) in [MetricKind::Diagonal, MetricKind::Dense]
        .into_iter()
        .enumerate()
    {
        let mut covariance = OnlineCovariance::new(2, kind).unwrap();
        let mut adapter = DualAveraging::new(0.3.try_into().unwrap(), 0.8.try_into().unwrap());
        let mut matrix = [1.0, 0.0, 1.0];
        for (i, stage) in WindowSchedule::new(200).enumerate() {
            let f: Vec<_> = rows[1 + case * 200 + i].split(',').collect();
            assert_eq!(f[0], if case == 0 { "diagonal" } else { "dense" });
            assert_eq!(f[1].parse::<usize>().unwrap(), i);
            let value: Vec<f64> = f[2..].iter().map(|s| s.parse().unwrap()).collect();
            let mut step = adapter.update(value[2]).unwrap();
            if let WarmupStage::Slow { window_end } = stage {
                covariance.update(&value[..2]).unwrap();
                if window_end {
                    let metric = covariance.metric().unwrap();
                    let mut a = [0.0; 2];
                    let mut b = [0.0; 2];
                    metric.velocity(&[1.0, 0.0], &mut a).unwrap();
                    metric.velocity(&[0.0, 1.0], &mut b).unwrap();
                    matrix = [a[0], a[1], b[1]];
                    step = adapter.finish();
                    adapter = DualAveraging::new(step, 0.8.try_into().unwrap());
                    covariance.reset();
                }
            }
            for (actual, expected) in [step.value(), matrix[0], matrix[1], matrix[2]]
                .into_iter()
                .zip(&value[3..])
            {
                assert!(
                    (actual - expected).abs() < 1e-9 * (1.0 + expected.abs()),
                    "case {case} row {i}: {actual} != {expected}"
                );
            }
        }
    }
}
