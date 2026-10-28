//! Gaussian power moments on [-1/2,1/2] with a Cauchy remainder.
pub(super) type C = (f64, f64);

pub(super) fn mul(a: C, b: C) -> C {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
pub(super) fn scale(a: C, x: f64) -> C {
    (a.0 * x, a.1 * x)
}
pub(super) fn add(a: C, b: C) -> C {
    (a.0 + b.0, a.1 + b.1)
}

const fn monomials() -> [f64; 260] {
    let mut values = [0.; 260];
    let mut power = 1.;
    let mut i = 0;
    while i < values.len() {
        if i % 2 == 0 {
            values[i] = power / (i + 1) as f64;
        }
        power *= 0.5;
        i += 1;
    }
    values
}
pub(super) const MONOMIALS: [f64; 260] = monomials();

pub(super) fn plan(a: f64, b: f64) -> Option<(usize, [f64; 4])> {
    if a == 0. && b == 0. {
        return Some((1, [0.; 4]));
    }
    let a = a.abs();
    let radius = a / 4. + b / 2.;
    if !radius.is_finite() || radius > 16. {
        return None;
    }
    let terms = if radius <= 0.01 {
        16
    } else if radius <= 1. {
        48
    } else if radius <= 4. {
        80
    } else if radius <= 8. {
        128
    } else {
        256
    };
    let contour = (2. * terms as f64 / (b + (b * b + 8. * a * terms as f64).sqrt())).clamp(1., 16.);
    let ratio = 0.5 / contour;
    let remainder =
        (a * contour * contour + b * contour + terms as f64 * ratio.ln() - (-ratio).ln_1p()).exp();
    Some((
        terms,
        std::array::from_fn(|r| remainder / (2_f64.powi(r as i32) * (r + 1) as f64)),
    ))
}

/// Four moments of exp(-a*z²+b*z); negative a and complex b are allowed.
pub(super) fn complex_moments<const MOMENTS: usize>(a: f64, b: C) -> Option<([C; 4], [f64; 4])> {
    let (terms, errors) = plan(a, b.0.hypot(b.1))?;
    let (mut previous, mut coefficient) = ((0., 0.), (1., 0.));
    let mut sums = [(0., 0.); 4];
    let mut compensation = [(0., 0.); 4];
    for n in 0..terms {
        for r in 0..MOMENTS {
            let value = scale(coefficient, MONOMIALS[n + r]);
            let y = add(value, scale(compensation[r], -1.));
            let next = add(sums[r], y);
            compensation[r] = add(add(next, scale(sums[r], -1.)), scale(y, -1.));
            sums[r] = next;
        }
        let next = scale(
            add(mul(b, coefficient), scale(previous, -2. * a)),
            1. / (n + 1) as f64,
        );
        previous = coefficient;
        coefficient = next;
    }
    Some((sums, errors))
}
