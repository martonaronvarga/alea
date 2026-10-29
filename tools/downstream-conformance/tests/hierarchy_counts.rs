use alea::target::LogDensityGradient;
use alea_downstream_conformance::modeling::{Observation, hierarchy, hierarchy_counts};

fn expanded(groups: &[(usize, usize); 2]) -> Vec<Observation> {
    groups
        .iter()
        .enumerate()
        .flat_map(|(group, &(events, trials))| {
            (0..trials).map(move |i| Observation::Logistic {
                group,
                event: i < events,
            })
        })
        .collect()
}
#[test]
fn counts_preserve_sequence_likelihood_and_pullbacks_including_empty_groups() {
    for groups in [
        [(3, 12), (10, 12)],
        [(0, 8), (8, 8)],
        [(0, 0), (1, 4)],
        [(0, 0), (0, 0)],
    ] {
        let scalar = hierarchy(&expanded(&groups)).into_target().unwrap();
        let counts = hierarchy_counts(&groups).into_target().unwrap();
        for q in [
            [0.1, -0.7, -0.2, 0.3],
            [-2., 0.2, 1.5, -1.],
            [-30., -1., 2., 3.],
            [30., -1., -2., -3.],
        ] {
            let mut a = [0.; 4];
            let mut b = [0.; 4];
            let x = scalar.logp_grad(&q, &mut a).unwrap();
            let y = counts.logp_grad(&q, &mut b).unwrap();
            assert!((x - y).abs() < 1e-10);
            for j in 0..4 {
                assert!((a[j] - b[j]).abs() < 1e-10);
                let mut plus = q;
                let mut minus = q;
                plus[j] += 1e-5;
                minus[j] -= 1e-5;
                let fd = (counts.logp_grad(&plus, &mut [0.; 4]).unwrap()
                    - counts.logp_grad(&minus, &mut [0.; 4]).unwrap())
                    / 2e-5;
                assert!((b[j] - fd).abs() < 1e-6 * (1. + fd.abs()));
            }
        }
    }
}
#[test]
fn graph_size_depends_on_groups_not_the_number_of_trials() {
    let a = hierarchy_counts(&[(25, 100), (75, 100)]).expression_count();
    let b = hierarchy_counts(&[(2500, 10000), (7500, 10000)]).expression_count();
    assert_eq!(a, b);
    assert!(a < 50);
    assert!(hierarchy(&expanded(&[(25, 100), (75, 100)])).expression_count() > a);
}
