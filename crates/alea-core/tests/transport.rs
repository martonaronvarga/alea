use alea_core::{
    target::{LogDensityGradient, evaluate},
    transport::FrozenTransportTarget,
};
use alea_math::{affine::LowRankAffineMap, buffer::OwnedBuffer};

#[derive(Debug, thiserror::Error)]
#[error("infallible")]
struct Never;

struct Gaussian;
impl LogDensityGradient for Gaussian {
    type Error = Never;
    fn dimension(&self) -> usize {
        2
    }
    fn logp_grad(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, Never> {
        gradient[0] = -q[0] / 4.0;
        gradient[1] = -q[1];
        Ok(-0.5 * (q[0] * q[0] / 4.0 + q[1] * q[1]))
    }
}

fn map() -> LowRankAffineMap {
    LowRankAffineMap::new(
        OwnedBuffer::from_fn(2, |i| [1.0, -2.0][i]),
        OwnedBuffer::from_fn(2, |i| [2.0, 0.5][i]),
        OwnedBuffer::new(0),
        OwnedBuffer::new(0),
    )
    .unwrap()
}

#[test]
fn transported_value_gradient_and_measure_match_finite_difference() {
    let transformed = FrozenTransportTarget::new(&Gaussian, map()).unwrap();
    let point = [0.3, -0.7];
    let mut gradient = [0.0; 2];
    let value = evaluate(&transformed, &point, &mut gradient).unwrap();
    let expected_x = [1.6, -2.35];
    let expected_value =
        -0.5 * (expected_x[0] * expected_x[0] / 4.0 + expected_x[1] * expected_x[1]);
    assert!((value - expected_value).abs() < 1e-14); // determinant is one
    let h = 1e-6;
    for coordinate in 0..2 {
        let mut plus = point;
        let mut minus = point;
        plus[coordinate] += h;
        minus[coordinate] -= h;
        let numerical = (evaluate(&transformed, &plus, &mut [0.0; 2]).unwrap()
            - evaluate(&transformed, &minus, &mut [0.0; 2]).unwrap())
            / (2.0 * h);
        assert!((gradient[coordinate] - numerical).abs() < 1e-9);
    }
}

#[test]
fn frozen_evaluation_reuses_storage() {
    let transformed = FrozenTransportTarget::new(&Gaussian, map()).unwrap();
    let point = [0.3, -0.7];
    let mut gradient = [0.0; 2];
    let measured = allocation_counter::measure(|| {
        for _ in 0..32 {
            evaluate(&transformed, &point, &mut gradient).unwrap();
        }
    });
    assert_eq!(measured.count_total, 0);
}
