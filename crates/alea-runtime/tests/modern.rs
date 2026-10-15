use alea_runtime::diagnostics::modern::{self, Error};
use std::collections::BTreeMap;

#[test]
fn executed_arviz_rank_folded_bulk_tail_mcse_and_energy_fixtures() {
    let mut cases: BTreeMap<&str, Vec<Vec<f64>>> = BTreeMap::new();
    for line in include_str!("fixtures/modern-draws.csv").lines().skip(1) {
        let cells: Vec<_> = line.split(',').collect();
        let chain: usize = cells[1].parse().unwrap();
        let values = cases.entry(cells[0]).or_default();
        values.resize_with(values.len().max(chain + 1), Vec::new);
        assert_eq!(values[chain].len(), cells[2].parse::<usize>().unwrap());
        values[chain].push(cells[3].parse().unwrap());
    }
    for line in include_str!("fixtures/modern-expected.csv").lines().skip(1) {
        let cells: Vec<_> = line.split(',').collect();
        let chains = &cases[cells[0]];
        let views: Vec<_> = chains.iter().map(Vec::as_slice).collect();
        let result = modern::summarize(&views);
        let error = match cells[1] {
            "shape" => Some(Error::Shape),
            "nonfinite" => Some(Error::Nonfinite),
            "undefined" => Some(Error::Undefined),
            "ok" => None,
            _ => panic!("bad fixture"),
        };
        if let Some(e) = error {
            assert_eq!(result, Err(e), "{}", cells[0]);
            continue;
        }
        let s = result.unwrap_or_else(|e| panic!("{}: {e}", cells[0]));
        let actual = [
            s.rhat,
            s.rank_rhat,
            s.folded_rhat,
            s.bulk_ess,
            s.tail_ess,
            s.mean_ess,
            s.mcse_mean,
            s.mean,
            modern::ebfmi(&chains[0]).unwrap(),
        ];
        for (i, (&actual, expected)) in actual.iter().zip(&cells[2..]).enumerate() {
            let expected: f64 = expected.parse().unwrap();
            assert!(
                (actual - expected).abs() <= 2e-9 * expected.abs().max(1.0),
                "{} field{i}: {actual} != {expected}",
                cells[0]
            );
        }
        if ["shifted", "scales", "drift"].contains(&cells[0]) {
            assert!(s.rhat > 1.01);
        }
    }
}

#[test]
fn diagnostics_are_scale_invariant_and_reject_missing_or_constant_energies() {
    let chains: Vec<Vec<f64>> = (0..4)
        .map(|c| {
            (0..64)
                .map(|i| ((i * 17 + c * 7) % 61) as f64 - 30.0)
                .collect()
        })
        .collect();
    let baseline =
        modern::summarize(&chains.iter().map(Vec::as_slice).collect::<Vec<_>>()).unwrap();
    for scale in [1e-200, 1e200] {
        let scaled: Vec<Vec<_>> = chains
            .iter()
            .map(|c| c.iter().map(|x| x * scale).collect())
            .collect();
        let result =
            modern::summarize(&scaled.iter().map(Vec::as_slice).collect::<Vec<_>>()).unwrap();
        assert!((result.bulk_ess - baseline.bulk_ess).abs() < 1e-9);
        assert!((result.mean_ess - baseline.mean_ess).abs() < 1e-9);
        assert!((result.mcse_mean / scale - baseline.mcse_mean).abs() < 1e-10);
    }
    assert_eq!(modern::ebfmi(&[1.0; 5]), Err(Error::Undefined));
    assert_eq!(modern::ebfmi(&[1.0, f64::NAN, 2.0]), Err(Error::Nonfinite));
}
