//! Relative-energy weights with a linear ordinary range and logarithmic tails.
//! The ordinary range avoids repeatedly exponentiating differences at merges.
//! Its margin covers every addressable NUTS tree without mass overflow.
use crate::trajectory_weight::Weight as LogWeight;

#[derive(Debug, Clone, Copy)]
pub(super) enum CachedWeight {
    Linear(f64),
    Logarithmic(LogWeight),
}
impl CachedWeight {
    pub(super) fn leaf(log: f64) -> Self {
        if (-500.0..=500.0).contains(&log) {
            Self::Linear(log.exp())
        } else {
            Self::Logarithmic(LogWeight::leaf(log))
        }
    }
    pub(super) fn acceptance(self) -> f64 {
        match self {
            Self::Linear(w) => w.min(1.0),
            Self::Logarithmic(w) => w.max.min(0.0).exp(),
        }
    }
    fn logarithmic(self) -> LogWeight {
        match self {
            Self::Linear(w) => LogWeight::leaf(w.ln()),
            Self::Logarithmic(w) => w,
        }
    }
    pub(super) fn merge_select(self, other: Self, biased: bool) -> (Self, f64) {
        if let (Self::Linear(a), Self::Linear(b)) = (self, other) {
            let sum = a + b;
            if sum.is_finite() {
                return (
                    Self::Linear(sum),
                    if biased { (b / a).min(1.0) } else { b / sum },
                );
            }
        }
        let (w, probability) = self.logarithmic().merge_select(other.logarithmic(), biased);
        (Self::Logarithmic(w), probability)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn linear_and_logarithmic_weights_preserve_extreme_multiplicities() {
        for log in [
            -1e300, -1000.0, -501.0, -500.0, 0.0, 500.0, 501.0, 1000.0, 1e300,
        ] {
            let mut w = CachedWeight::leaf(log);
            for _ in 0..63 {
                let (doubled, p) = w.merge_select(w, false);
                assert_eq!(p, 0.5);
                assert_eq!(w.merge_select(w, true).1, 1.0);
                assert!((w.merge_select(doubled, false).1 - 2.0 / 3.0).abs() < 2e-14);
                w = doubled;
            }
        }
        for a in [
            -1000.0, -501.0, -500.0, -1.0, 0.0, 1.0, 500.0, 501.0, 1000.0,
        ] {
            for b in [
                -1000.0, -501.0, -500.0, -1.0, 0.0, 1.0, 500.0, 501.0, 1000.0,
            ] {
                let mut wa = CachedWeight::leaf(a);
                let mut wb = CachedWeight::leaf(b);
                for _ in 0..32 {
                    wa = wa.merge_select(wa, false).0;
                }
                for _ in 0..16 {
                    wb = wb.merge_select(wb, false).0;
                }
                for biased in [false, true] {
                    let ratio = (b - a) + (16.0 - 32.0) * 2.0_f64.ln();
                    let expected = if biased {
                        ratio.min(0.0).exp()
                    } else if ratio >= 0.0 {
                        1.0 / (1.0 + (-ratio).exp())
                    } else {
                        let r = ratio.exp();
                        r / (1.0 + r)
                    };
                    let actual = wa.merge_select(wb, biased).1;
                    assert!(
                        (actual - expected).abs() <= 3e-13 * expected + f64::from_bits(4),
                        "{a} {b} {biased} {actual} {expected}"
                    );
                }
            }
        }
    }
}
