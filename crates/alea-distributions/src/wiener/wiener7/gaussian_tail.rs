//! Endpoint-scaled Gaussian moments with bounded Laplace-series remainders.
const COEFFICIENTS: [[f64; 4]; 4] = [
    [1., 0., 0., 0.],
    [0.5, -1., 0., 0.],
    [0.25, -1., 1., 0.],
    [0.125, -0.75, 1.5, -1.],
];
const TAIL_COEFFICIENTS: [[f64; 4]; 4] = [
    [1., 0., 0., 0.],
    [1.5, 1., 0., 0.],
    [2.25, 3., 1., 0.],
    [3.375, 6.75, 4.5, 1.],
];

pub(super) fn moments(a: f64, b: f64) -> Option<([f64; 4], [f64; 4], f64)> {
    let l = b.abs() - a;
    let delta = (a / l) / l;
    if !a.is_finite() || !b.is_finite() || a < 0. || l < 40. || !(0. ..=0.0025).contains(&delta) {
        return None;
    }
    let mut term = [l.recip(); 4];
    for m in 1..4 {
        term[m] = term[m - 1] * m as f64 / l;
    }
    let tail: [f64; 4] = std::array::from_fn(|r| {
        (-l).exp()
            * TAIL_COEFFICIENTS[r]
                .iter()
                .zip(term)
                .map(|(c, t)| c * t)
                .sum::<f64>()
    });
    let mut values = [0.; 4];
    let mut compensation = [0.; 4];
    let mut best = None;
    let mut best_error = f64::INFINITY;
    for n in 0..32 {
        for m in 0..4 {
            let addition = if n % 2 == 0 { term[m] } else { -term[m] };
            let y = addition - compensation[m];
            let next = values[m] + y;
            compensation[m] = (next - values[m]) - y;
            values[m] = next;
            term[m] *= delta * ((2 * n + m + 1) as f64) * ((2 * n + m + 2) as f64) / (n + 1) as f64;
        }
        let sums: [f64; 4] = std::array::from_fn(|r| {
            let sign = if r % 2 == 0 { 1. } else { b.signum() };
            sign * COEFFICIENTS[r]
                .iter()
                .zip(values)
                .map(|(c, t)| c * t)
                .sum::<f64>()
        });
        // Taylor's e^(-x) remainder is <=x^N/N! for x>=0. Integrating
        // y^(2N+m)e^(-l*y) over [0,infinity) gives the next factorial term.
        let errors: [f64; 4] = std::array::from_fn(|r| {
            tail[r]
                + COEFFICIENTS[r]
                    .iter()
                    .zip(term)
                    .map(|(c, t)| c.abs() * t)
                    .sum::<f64>()
        });
        let error = errors.iter().copied().fold(0., f64::max);
        if sums[0] > 0. && sums.iter().chain(&errors).all(|x| x.is_finite()) && error < best_error {
            best = Some((sums, errors, -a / 4. + b.abs() / 2.));
            best_error = error;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn endpoint_moments_match_independent_continuous_ball_integrals() {
        let mut count = 0;
        for line in include_str!("../../../tests/support/wiener_moment_balls.csv")
            .lines()
            .skip(1)
        {
            let x: Vec<f64> = line.split(',').map(|x| x.parse().unwrap()).collect();
            let (moments, errors, _) = moments(x[0], x[1]).unwrap();
            for r in 0..4 {
                assert!(
                    (moments[r] - x[2 + 2 * r]).abs()
                        <= errors[r] + x[3 + 2 * r] + 2e-14 * moments[0],
                    "row {count},moment {r}"
                );
            }
            count += 1;
        }
        assert_eq!(count, 12);
    }
}
