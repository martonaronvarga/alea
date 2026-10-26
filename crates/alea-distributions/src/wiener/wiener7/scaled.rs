//! Rare-path, locally scaled quadrature. No natural-scale tail density is formed.
use super::*;

#[derive(Clone, Copy)]
struct Sum {
    scale: f64,
    value: [f64; 8],
}

impl Sum {
    fn new() -> Self {
        Self {
            scale: f64::NEG_INFINITY,
            value: [0.0; 8],
        }
    }

    fn add(&mut self, log_weight: f64, scores: [f64; 7]) {
        if log_weight > self.scale {
            let ratio = (self.scale - log_weight).exp();
            for v in &mut self.value {
                *v *= ratio;
            }
            self.scale = log_weight;
        }
        let weight = (log_weight - self.scale).exp();
        self.value[0] += weight;
        for (dst, score) in self.value[1..].iter_mut().zip(scores) {
            *dst += weight * score;
        }
    }
}

struct Region {
    lo: [f64; 2],
    hi: [f64; 2],
    sum: Sum,
    error: [f64; 8],
    split: usize,
}

impl Wiener7 {
    fn scaled_rule(
        core: &Wiener7Core,
        obs: &WienerObservation,
        lo: [f64; 2],
        hi: [f64; 2],
        order: usize,
    ) -> Option<(Sum, usize)> {
        let (nodes, weights) = Self::fixed_rule(order);
        let mut sum = Sum::new();
        let mut peak = f64::NEG_INFINITY;
        let mut split = 0;
        for (&node0, &weight0) in nodes.iter().zip(weights) {
            let (u0, w0) = Self::fixed_interval(node0, weight0, lo[0], hi[0]);
            let second = if core.dim == 2 { nodes.len() } else { 1 };
            for j in 0..second {
                let (u1, w1) = if core.dim == 2 {
                    Self::fixed_interval(nodes[j], weights[j], lo[1], hi[1])
                } else {
                    (0.0, 1.0)
                };
                let x = [u0, u1];
                let (lp, scores) = Self::point_log_scores(core, obs, &x)?;
                if lp == f64::NEG_INFINITY {
                    continue;
                }
                // Keep the volume in log space too, including truncated support.
                let log_weight = lp + w0.ln() + w1.ln();
                sum.add(log_weight, scores);
                if lp > peak {
                    peak = lp;
                    if core.dim == 2 {
                        let dw = (hi[0] - lo[0]) * core.sw * scores[2];
                        let dt = (hi[1] - lo[1]) * core.st0 * scores[1];
                        split = usize::from(dt.abs() > dw.abs());
                    }
                }
            }
        }
        (sum.scale.is_finite() && sum.value[0] > 0.0).then_some((sum, split))
    }

    fn scaled_region(
        core: &Wiener7Core,
        obs: &WienerObservation,
        lo: [f64; 2],
        hi: [f64; 2],
    ) -> Option<Region> {
        let (sum, split) = Self::scaled_rule(core, obs, lo, hi, 15)?;
        let (low, _) = Self::scaled_rule(core, obs, lo, hi, 7)?;
        let factor = (low.scale - sum.scale).exp();
        let error = std::array::from_fn(|i| (sum.value[i] - factor * low.value[i]).abs());
        if error.iter().any(|e| !e.is_finite()) {
            return None;
        }
        Some(Region {
            lo,
            hi,
            sum,
            error,
            split,
        })
    }

    pub(super) fn scaled_fixed(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
        order: usize,
    ) -> Wiener7Eval {
        match Self::scaled_rule(core, obs, core.xmin, core.xmax, order) {
            Some((s, _)) => Wiener7Eval {
                log_prob: s.scale + s.value[0].ln(),
                grad: Wiener7Grad::from_array(std::array::from_fn(|i| s.value[i + 1] / s.value[0])),
            },
            None => fail_eval_7(),
        }
    }

    /// Refine independent GL7/GL15 product rules with a separate scale per box.
    /// Their discrepancy estimates error; it is not a mathematical certificate.
    #[cold]
    pub(super) fn scaled_adaptive(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
    ) -> Wiener7Eval {
        let Some(first) = Self::scaled_region(core, obs, core.xmin, core.xmax) else {
            return fail_eval_7();
        };
        let mut regions = vec![first];
        let per_region = if core.dim == 2 {
            15 * 15 + 7 * 7
        } else {
            15 + 7
        };
        let mut evaluations = per_region;
        let tolerance = core.opts.req_rel_error;
        if tolerance < 64.0 * f64::EPSILON {
            return fail_eval_7();
        }
        loop {
            let scale = regions
                .iter()
                .map(|r| r.sum.scale)
                .fold(f64::NEG_INFINITY, f64::max);
            let mut total = [0.0; 8];
            let mut error = [0.0; 8];
            for r in &regions {
                let factor = (r.sum.scale - scale).exp();
                for i in 0..8 {
                    total[i] += factor * r.sum.value[i];
                    error[i] += factor * r.error[i];
                }
            }
            let density = total[0];
            let margin = density - error[0];
            let scores: [f64; 7] = std::array::from_fn(|i| total[i + 1] / density);
            let converged = margin > 0.0
                && error[0] / margin <= tolerance
                && scores.iter().enumerate().all(|(i, g)| {
                    g.is_finite()
                        && (error[i + 1] + g.abs() * error[0]) / margin
                            <= tolerance * g.abs().max(1.0)
                });
            if converged {
                return Wiener7Eval {
                    log_prob: scale + density.ln(),
                    grad: Wiener7Grad::from_array(scores),
                };
            }
            // Bounded rare path; failure stays visible through the checked API.
            if evaluations + 2 * per_region > 100_000 {
                return fail_eval_7();
            }
            let (index, _) = regions
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    let weight = (r.sum.scale - scale).exp();
                    let priority = (0..8)
                        .map(|k| weight * r.error[k] / total[k].abs().max(density))
                        .fold(0.0, f64::max);
                    (i, priority)
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .expect("at least one integration region");
            let old = &regions[index];
            let d = old.split;
            let middle = old.lo[d] + 0.5 * (old.hi[d] - old.lo[d]);
            if middle == old.lo[d] || middle == old.hi[d] {
                return fail_eval_7();
            }
            let mut left_hi = old.hi;
            left_hi[d] = middle;
            let mut right_lo = old.lo;
            right_lo[d] = middle;
            let Some(left) = Self::scaled_region(core, obs, old.lo, left_hi) else {
                return fail_eval_7();
            };
            let Some(right) = Self::scaled_region(core, obs, right_lo, old.hi) else {
                return fail_eval_7();
            };
            regions[index] = left;
            regions.push(right);
            evaluations += 2 * per_region;
        }
    }
}
