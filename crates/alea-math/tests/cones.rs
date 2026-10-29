use alea_math::exact::{ConeSearch, ConeSearchOptions, ExactDotError, strict_cone_direction};
use std::cmp::Ordering;

#[test]
fn matches_independent_fraction_vertex_enumeration() {
    let mut cases = 0;
    for line in include_str!("support/cones-fraction.txt")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let fields: Vec<_> = line.split('|').collect();
        let p = fields[0].parse().unwrap();
        let parse = |s: &str| {
            s.split(',')
                .filter(|s| !s.is_empty())
                .map(|s| s.parse().unwrap())
                .collect::<Vec<f64>>()
        };
        let a = parse(fields[1]);
        let e = parse(fields[2]);
        let result = strict_cone_direction(p, &a, &e, ConeSearchOptions::default()).unwrap();
        match result {
            ConeSearch::Direction(d) => {
                assert_eq!(fields[3], "1", "{line}");
                let signs: Vec<_> = a.chunks_exact(p).map(|r| d.dot_sign(r).unwrap()).collect();
                assert!(signs.iter().all(|s| *s != Ordering::Less));
                assert!(signs.contains(&Ordering::Greater));
                assert!(
                    e.chunks_exact(p)
                        .all(|r| d.dot_sign(r).unwrap() == Ordering::Equal)
                );
            }
            ConeSearch::NoImprovingDirection => assert_eq!(fields[3], "0", "{line}"),
            ConeSearch::BudgetExceeded => panic!("unexpected budget exhaustion: {line}"),
        }
        cases += 1;
    }
    assert_eq!(cases, 180);
}

#[test]
fn budgets_and_invalid_input_do_not_claim_infeasibility() {
    let default = ConeSearchOptions::default();
    for options in [
        ConeSearchOptions {
            max_rows: 0,
            ..default
        },
        ConeSearchOptions {
            max_columns: 0,
            ..default
        },
        ConeSearchOptions {
            max_coefficient_bits: 0,
            ..default
        },
    ] {
        assert!(matches!(
            strict_cone_direction(1, &[1.], &[], options).unwrap(),
            ConeSearch::BudgetExceeded
        ));
    }
    // Enough room for inputs, but not for a generated elimination row.
    assert!(matches!(
        strict_cone_direction(
            1,
            &[1., -1.],
            &[],
            ConeSearchOptions {
                max_rows: 3,
                ..default
            }
        )
        .unwrap(),
        ConeSearch::BudgetExceeded
    ));
    assert_eq!(
        strict_cone_direction(2, &[1.], &[], default).unwrap_err(),
        ExactDotError::Shape
    );
    assert_eq!(
        strict_cone_direction(1, &[f64::NAN], &[], default).unwrap_err(),
        ExactDotError::NonFinite
    );
    assert_eq!(
        strict_cone_direction(0, &[1.], &[], default).unwrap_err(),
        ExactDotError::Shape
    );
}

#[test]
fn integer_rays_preserve_extreme_equalities_without_binary64_rounding() {
    let tiny = f64::from_bits(1);
    let ConeSearch::Direction(d) = strict_cone_direction(
        2,
        &[tiny, 0.],
        &[f64::MAX, -tiny],
        ConeSearchOptions::default(),
    )
    .unwrap() else {
        panic!("expected ray")
    };
    assert_eq!(d.dot_sign(&[tiny, 0.]).unwrap(), Ordering::Greater);
    assert_eq!(d.dot_sign(&[f64::MAX, -tiny]).unwrap(), Ordering::Equal);
    assert!(d.coefficients()[1].bits() > 2000);
    assert_eq!(d.dot_sign(&[1.]).unwrap_err(), ExactDotError::Shape);
    assert_eq!(
        d.dot_sign(&[f64::INFINITY, 0.]).unwrap_err(),
        ExactDotError::NonFinite
    );
    assert!(matches!(
        strict_cone_direction(2, &[tiny, 0., -tiny, 0.], &[], ConeSearchOptions::default())
            .unwrap(),
        ConeSearch::NoImprovingDirection
    ));
}
