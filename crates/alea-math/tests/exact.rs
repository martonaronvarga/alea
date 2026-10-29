use alea_math::exact::{ExactDotError, dot_sign};
use std::cmp::Ordering;
#[test]
fn agrees_with_independent_python_fraction_oracle() {
    for (line_number, line) in include_str!("support/exact-dot-fraction.txt")
        .lines()
        .enumerate()
    {
        if line.starts_with('#') {
            continue;
        }
        let parts: Vec<_> = line.split('|').collect();
        let parse = |s: &str| {
            s.split(',')
                .map(|v| f64::from_bits(u64::from_str_radix(v, 16).unwrap()))
                .collect::<Vec<_>>()
        };
        let expected = match parts[2] {
            "-1" => Ordering::Less,
            "0" => Ordering::Equal,
            "1" => Ordering::Greater,
            _ => panic!("invalid oracle sign"),
        };
        assert_eq!(
            dot_sign(&parse(parts[0]), &parse(parts[1])).unwrap(),
            expected,
            "fixture line {line_number}"
        );
    }
}
#[test]
fn zero_cancellation_overflow_and_underflow_have_exact_signs() {
    let tiny = f64::from_bits(1);
    assert_eq!(
        dot_sign(&[f64::MAX, f64::MAX], &[f64::MAX, -f64::MAX]),
        Ok(Ordering::Equal)
    );
    assert_eq!(
        dot_sign(&[f64::MAX, f64::MAX, tiny], &[f64::MAX, -f64::MAX, tiny]),
        Ok(Ordering::Greater)
    );
    assert_eq!(dot_sign(&[-tiny], &[tiny]), Ok(Ordering::Less));
    assert_eq!(dot_sign(&[-0.], &[f64::MAX]), Ok(Ordering::Equal));
    assert_eq!(dot_sign(&[], &[]), Ok(Ordering::Equal));
    assert_eq!(dot_sign(&[1.], &[]), Err(ExactDotError::Shape));
    assert_eq!(dot_sign(&[f64::NAN], &[1.]), Err(ExactDotError::NonFinite));
}
