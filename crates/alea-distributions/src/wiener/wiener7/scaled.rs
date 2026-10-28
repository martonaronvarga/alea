//! Rare-path, locally scaled quadrature. No natural-scale tail density is formed.
use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Reduction {
    None,
    Start,
    Time,
}

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
    pub(super) fn reduced_start_integral(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
    ) -> Option<Wiener7Eval> {
        self.reduced_start_components::<true>(core, obs)
    }

    pub(super) fn reduced_start_density(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
    ) -> Option<f64> {
        self.reduced_start_components::<false>(core, obs)
            .map(|e| e.log_prob)
    }

    fn reduced_start_components<const SCORES: bool>(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
    ) -> Option<Wiener7Eval> {
        if core.dim != 2 {
            return None;
        }
        for x in [0., 0.5, 1.] {
            Self::averaged_start_components::<SCORES>(core, obs, x)?;
        }
        let value = self.scaled_integral(core, obs, SCORES, Reduction::Time);
        (value.log_prob.is_finite() && value.grad_array().iter().all(|x| x.is_finite()))
            .then_some(value)
    }
    fn averaged_start_components<const SCORES: bool>(
        core: &Wiener7Core,
        obs: &WienerObservation,
        x: f64,
    ) -> Option<(f64, [f64; 7])> {
        let mut point = core.clone();
        point.beta0 = core.beta0 + core.sw * (x - 0.5);
        point.sw = 0.;
        point.dim = 1;
        point.xmin = [0.; 2];
        point.xmax = [core.xmax[1], 1.];
        point.eps_series = point.eps_series.min(0.01 * core.opts.req_rel_error);
        if !SCORES {
            return Some((Wiener7.analytic_time_density(&point, obs)?, [0.; 7]));
        }
        let value = Wiener7.analytic_time(&point, obs)?;
        let mut scores = value.grad_array();
        scores[5] = (x - 0.5) * scores[2];
        Some((value.log_prob, scores))
    }
    fn scaled_rule<const SCORES: bool>(
        core: &Wiener7Core,
        obs: &WienerObservation,
        lo: [f64; 2],
        hi: [f64; 2],
        order: usize,
        second_order: usize,
        reduction: Reduction,
    ) -> Option<(Sum, usize)> {
        let dimension = if reduction != Reduction::None {
            1
        } else {
            core.dim
        };
        let (nodes, weights) = Self::fixed_rule(order);
        let (second_nodes, second_weights) = Self::fixed_rule(second_order);
        let mut sum = Sum::new();
        let mut peak = f64::NEG_INFINITY;
        let mut split = 0;
        for (&node0, &weight0) in nodes.iter().zip(weights) {
            let (u0, w0) = Self::fixed_interval(node0, weight0, lo[0], hi[0]);
            let second = if dimension == 2 {
                second_nodes.len()
            } else {
                1
            };
            for j in 0..second {
                let (u1, w1) = if dimension == 2 {
                    Self::fixed_interval(second_nodes[j], second_weights[j], lo[1], hi[1])
                } else {
                    (0.0, 1.0)
                };
                let x = [u0, u1];
                let (lp, scores) = match reduction {
                    Reduction::Start => {
                        if SCORES {
                            Self::averaged_time_log_scores(core, obs, x[0])?
                        } else {
                            (Self::averaged_time_log_density(core, obs, x[0])?, [0.; 7])
                        }
                    }
                    Reduction::Time => Self::averaged_start_components::<SCORES>(core, obs, x[0])?,
                    Reduction::None => {
                        if SCORES {
                            Self::point_log_scores(core, obs, &x)?
                        } else {
                            (Self::point_log_density(core, obs, &x)?, [0.; 7])
                        }
                    }
                };
                if lp == f64::NEG_INFINITY {
                    continue;
                }
                // Keep the volume in log space too, including truncated support.
                let log_weight = lp + w0.ln() + w1.ln();
                sum.add(log_weight, scores);
                if lp > peak {
                    peak = lp;
                    if SCORES && dimension == 2 {
                        let dw = (hi[0] - lo[0]) * core.sw * scores[2];
                        let dt = (hi[1] - lo[1]) * core.st0 * scores[1];
                        split = usize::from(dt.abs() > dw.abs());
                    }
                }
            }
        }
        (sum.scale.is_finite() && sum.value[0] > 0.0).then_some((sum, split))
    }

    fn use_time_fluxes(core: &Wiener7Core, obs: &WienerObservation) -> bool {
        // Nearby faces lose digits in both a difference and a width division.
        // Direct unit-cube derivatives remain stable in that narrow limit.
        Self::time_face_budget(core) >= 64. * f64::EPSILON
            && core.st0 / (obs.rt - core.tau0)
                >= (32. * f64::EPSILON / core.opts.req_rel_error).sqrt()
    }

    fn time_face_budget(core: &Wiener7Core) -> f64 {
        0.01 * core.opts.req_rel_error * core.st0.min(1.)
    }

    /// Integrate the two start-averaged time faces once. All fluxes remain in
    /// log scale, including densities below the normal binary64 range.
    fn time_faces(&self, core: &Wiener7Core, obs: &WienerObservation) -> Option<[f64; 2]> {
        let mut point = core.clone();
        point.tau0 = 0.;
        point.st0 = 0.;
        point.dim = usize::from(core.sw > 0.);
        point.xmin = [0.; 2];
        point.xmax = [1.; 2];
        point.opts.req_rel_error = Self::time_face_budget(core);
        point.eps_series = point.eps_series.min(point.opts.req_rel_error);
        let max_time = obs.rt - core.tau0;
        let mut faces = [f64::NEG_INFINITY; 2];
        for (log, t) in faces.iter_mut().zip([max_time - core.st0, max_time]) {
            if t <= 0. {
                continue;
            }
            let event = WienerObservation {
                rt: t,
                boundary: obs.boundary,
            };
            let lp = if point.dim == 0 {
                Self::point_log_density(&point, &event, &[0.5; 2])?
            } else {
                self.analytic_start_density(&point, &event)
                    .unwrap_or_else(|| {
                        self.scaled_integral(&point, &event, false, Reduction::None)
                            .log_prob
                    })
            };
            if !lp.is_finite() {
                return None;
            }
            *log = lp - core.st0.ln();
        }
        Some(faces)
    }

    fn start_face_budget(core: &Wiener7Core) -> Option<f64> {
        // Endpoint subtraction is unsuitable for a vanishing start width.
        // Leave that limit to the direct unit-cube derivative or moments.
        let budget = 0.01 * core.opts.req_rel_error * core.sw.min(1.);
        (core.sw > 0. && core.st0 > 0. && budget >= 64. * f64::EPSILON).then_some(budget)
    }

    /// Time-averaged densities at the two start faces, divided by start width.
    fn start_faces(&self, core: &Wiener7Core, obs: &WienerObservation) -> Option<[f64; 2]> {
        let budget = Self::start_face_budget(core)?;
        let mut point = core.clone();
        point.sw = 0.;
        point.dim = 1;
        point.xmin = [0.; 2];
        point.xmax = [core.xmax[1], 1.];
        point.opts.req_rel_error = budget;
        point.eps_series = point.eps_series.min(budget);
        let mut faces = [0.; 2];
        for (log, beta) in faces
            .iter_mut()
            .zip([core.beta0 - core.sw / 2., core.beta0 + core.sw / 2.])
        {
            point.beta0 = beta;
            let lp = if point.sv == 0. {
                let base = Wiener4Params::from_array([point.alpha, point.tau0, beta, point.delta]);
                Wiener4
                    .try_log_interval_prob(
                        obs.rt - point.st0,
                        obs.rt,
                        obs.boundary,
                        &base,
                        WienerOptions::new(budget),
                    )
                    .ok()
                    .map(|value| value - point.st0.ln())
            } else {
                None
            }
            .unwrap_or_else(|| {
                self.scaled_integral(&point, obs, false, Reduction::None)
                    .log_prob
            });
            if !lp.is_finite() {
                return None;
            }
            *log = lp - core.sw.ln();
        }
        Some(faces)
    }

    fn scaled_region(
        core: &Wiener7Core,
        obs: &WienerObservation,
        lo: [f64; 2],
        hi: [f64; 2],
        gradients: bool,
        reduction: Reduction,
        start_fluxes: bool,
    ) -> Option<Region> {
        let rule = |first, second| {
            let (mut sum, split) = if gradients {
                Self::scaled_rule::<true>(core, obs, lo, hi, first, second, reduction)?
            } else {
                Self::scaled_rule::<false>(core, obs, lo, hi, first, second, reduction)?
            };
            if !gradients {
                sum.value[1..].fill(0.);
            }
            // Time derivatives are obtained from the full boundary fluxes.
            if gradients && reduction != Reduction::Time && Self::use_time_fluxes(core, obs) {
                sum.value[2] = 0.;
                sum.value[7] = 0.;
            }
            if start_fluxes {
                sum.value[3] = 0.;
                sum.value[6] = 0.;
            }
            Some((sum, split))
        };
        let (sum, mut split) = rule(15, 15)?;
        let (low, _) = rule(7, 7)?;
        let factor = (low.scale - sum.scale).exp();
        let mut error = std::array::from_fn(|i| (sum.value[i] - factor * low.value[i]).abs());
        if core.dim == 2 && reduction == Reduction::None {
            // Mixed rules isolate the error in each coordinate. The peak slope
            // alone need not detect curvature in a derivative numerator.
            let (low_start, _) = rule(7, 15)?;
            let (low_time, _) = rule(15, 7)?;
            let factors = [
                (low_start.scale - sum.scale).exp(),
                (low_time.scale - sum.scale).exp(),
            ];
            let mut priority = [0_f64; 2];
            for (i, component_error) in error.iter_mut().enumerate() {
                let differences = [
                    (sum.value[i] - factors[0] * low_start.value[i]).abs(),
                    (sum.value[i] - factors[1] * low_time.value[i]).abs(),
                ];
                *component_error = component_error.max(differences[0] + differences[1]);
                let norm = if i == 0 && start_fluxes {
                    sum.value[0] * core.sw.min(1.)
                } else {
                    sum.value[i].abs().max(sum.value[0])
                };
                for d in 0..2 {
                    priority[d] = priority[d].max(differences[d] / norm);
                }
            }
            split = usize::from(priority[1] > priority[0]);
        }
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
        match Self::scaled_rule::<true>(
            core,
            obs,
            core.xmin,
            core.xmax,
            order,
            order,
            Reduction::None,
        ) {
            Some((s, _)) => Wiener7Eval {
                log_prob: s.scale + s.value[0].ln(),
                grad: Wiener7Grad::from_array(std::array::from_fn(|i| s.value[i + 1] / s.value[0])),
            },
            None => fail_eval_7(),
        }
    }

    pub(super) fn scaled_fixed_density(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
        order: usize,
    ) -> Wiener7Eval {
        match Self::scaled_rule::<false>(
            core,
            obs,
            core.xmin,
            core.xmax,
            order,
            order,
            Reduction::None,
        ) {
            Some((sum, _)) => Wiener7Eval {
                log_prob: sum.scale + sum.value[0].ln(),
                grad: NAN_GRAD_7,
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
        self.scaled_adaptive_components(core, obs, true)
    }

    #[cold]
    pub(super) fn scaled_density(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
    ) -> Wiener7Eval {
        self.scaled_adaptive_components(core, obs, false)
    }

    fn scaled_adaptive_components(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
        gradients: bool,
    ) -> Wiener7Eval {
        if core.dim == 2
            && (if gradients {
                Self::averaged_time_log_scores(core, obs, 0.).is_some()
            } else {
                Self::averaged_time_log_density(core, obs, 0.).is_some()
            })
        {
            let reduced = self.scaled_integral(core, obs, gradients, Reduction::Start);
            if reduced.log_prob.is_finite() && reduced.grad_array().iter().all(|v| v.is_finite()) {
                return reduced;
            }
        }
        self.scaled_integral(core, obs, gradients, Reduction::None)
    }

    fn scaled_integral(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
        gradients: bool,
        reduction: Reduction,
    ) -> Wiener7Eval {
        let start_faces =
            if gradients && reduction != Reduction::Time && Self::start_face_budget(core).is_some()
            {
                self.start_faces(core, obs)
            } else {
                None
            };
        let faces = if gradients && reduction != Reduction::Time && Self::use_time_fluxes(core, obs)
        {
            let Some(faces) = self.time_faces(core, obs) else {
                return fail_eval_7();
            };
            Some(faces)
        } else {
            None
        };
        let (lo, hi) = match reduction {
            Reduction::Start => ([0.; 2], [core.xmax[1], 1.]),
            Reduction::Time => ([0.; 2], [1.; 2]),
            Reduction::None => (core.xmin, core.xmax),
        };
        let Some(first) = Self::scaled_region(
            core,
            obs,
            lo,
            hi,
            gradients,
            reduction,
            start_faces.is_some(),
        ) else {
            return fail_eval_7();
        };
        let mut regions = vec![first];
        let per_region = if core.dim == 2 && reduction == Reduction::None {
            15 * 15 + 7 * 7 + 2 * 7 * 15
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
            let mut alpha_fluxes = false;
            if let Some(faces) = faces {
                let face = faces.map(|lp| (lp - scale).exp());
                let time = usize::from(core.sw > 0.);
                let budget = Self::time_face_budget(core);
                total[2] = face[0] - face[1];
                error[2] = budget * (face[0] + face[1]);
                total[7] =
                    core.xmax[time] * face[0] - core.xmin[time] * face[1] - total[0] / core.st0;
                error[7] = budget * (core.xmax[time] * face[0] + core.xmin[time] * face[1])
                    + error[0] / core.st0;
                let upper = obs.rt - core.tau0;
                let lower = (upper - core.st0).max(0.);
                // F(T,a,v,s)=F(T/a²,1,av,as). Apply this scale identity
                // to the two time limits instead of cancelling alpha scores
                // inside a sharply peaked first-passage density.
                let alpha = (2. * (lower * face[0] - upper * face[1])
                    + core.delta * total[4]
                    + core.sv * total[5])
                    / core.alpha;
                let alpha_error = (2. * budget * (lower * face[0] + upper * face[1])
                    + core.delta.abs() * error[4]
                    + core.sv * error[5])
                    / core.alpha;
                // Scaling can itself cancel large drift and face terms. Keep
                // the directly integrated derivative when its estimate is
                // better, rather than imposing an unattainable face budget.
                if alpha_error < error[1] {
                    total[1] = alpha;
                    error[1] = alpha_error;
                    alpha_fluxes = true;
                }
            }
            if let Some(faces) = start_faces {
                let face = faces.map(|lp| (lp - scale).exp());
                let budget = Self::start_face_budget(core).expect("start faces have a budget");
                total[3] = face[1] - face[0];
                error[3] = budget * (face[0] + face[1]);
                total[6] = 0.5 * (face[0] + face[1]) - total[0] / core.sw;
                error[6] = 0.5 * budget * (face[0] + face[1]) + error[0] / core.sw;
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
                        .map(|k| {
                            if k == 1 && alpha_fluxes {
                                return 0.;
                            }
                            let norm = if k == 0 && start_faces.is_some() {
                                density * core.sw.min(1.)
                            } else {
                                total[k].abs().max(density)
                            };
                            weight * r.error[k] / norm
                        })
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
            let Some(left) = Self::scaled_region(
                core,
                obs,
                old.lo,
                left_hi,
                gradients,
                reduction,
                start_faces.is_some(),
            ) else {
                return fail_eval_7();
            };
            let Some(right) = Self::scaled_region(
                core,
                obs,
                right_lo,
                old.hi,
                gradients,
                reduction,
                start_faces.is_some(),
            ) else {
                return fail_eval_7();
            };
            regions[index] = left;
            regions.push(right);
            evaluations += 2 * per_region;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_start_and_clipped_time_proposals_are_evaluable() {
        for (rt, lower, p) in [
            (
                0.469,
                false,
                [
                    2.4749736829675593,
                    0.16668455332354118,
                    0.4689899247129888,
                    0.6389540258904112,
                    1.0066854590943801,
                    0.7799458999413749,
                    0.34291442859487303,
                ],
            ),
            (
                0.675,
                true,
                [
                    1.3379729514065335,
                    0.4604482709501249,
                    0.48721932677946567,
                    0.6890406129269979,
                    0.017193277147217887,
                    0.7909296417576759,
                    0.3746925547442049,
                ],
            ),
            (
                0.675,
                true,
                [
                    1.5779382816950356,
                    0.3892536621148969,
                    0.3415871894921585,
                    0.5005374649192191,
                    349.4261431791342,
                    0.6348044548085039,
                    0.3751468294604952,
                ],
            ),
            (
                0.675,
                true,
                [
                    1.184370251981865,
                    0.3238302317741533,
                    0.19842378532536026,
                    2.014406521010839,
                    0.9636510388824948,
                    0.2652004524018589,
                    0.3619964232523812,
                ],
            ),
            (
                0.4184013856731187,
                false,
                [
                    1.5178881986936503,
                    0.2485551790941962,
                    0.6164806090045881,
                    1.2061561742322793,
                    1.2615614023185378,
                    0.4513465885975576,
                    0.1704964522307809,
                ],
            ),
            (
                0.3807531142811618,
                false,
                [
                    1.4218349687873668,
                    0.2630065224865575,
                    0.5287900854310247,
                    0.6889164658022419,
                    0.13240361346377336,
                    0.6492779888742992,
                    0.11596088802520918,
                ],
            ),
            (
                0.36303379407076364,
                false,
                [
                    1.4166544003274433,
                    0.2877173417664885,
                    0.5443025249813934,
                    0.6323720847103405,
                    0.25203635546122377,
                    0.6581086385541228,
                    0.08915750638575942,
                ],
            ),
            (
                0.41124039843703153,
                false,
                [
                    1.346855690424418,
                    0.2411090185689389,
                    0.5340249572171718,
                    0.8143655551175284,
                    0.648245807686055,
                    0.6034460819045603,
                    0.2018628898840445,
                ],
            ),
        ] {
            let observation = WienerObservation {
                rt,
                boundary: if lower {
                    Boundary::Lower
                } else {
                    Boundary::Upper
                },
            };
            let parameters =
                Wiener7Params::with_params(p[0], p[1], p[2], p[3], p[5], p[6], p[4]).unwrap();
            let result = Wiener7.try_fused(&observation, &parameters, WienerOptions::new(1e-9));
            assert!(result.is_ok(), "{parameters:?}: {result:?}");
            let result = result.unwrap();
            let core = Wiener7.core(&observation, &parameters, 1e-13).unwrap();
            // Independently composite GL25: no adaptive splitting decisions.
            let mut reference = Sum::new();
            let panels = 8;
            let max_time = rt - p[1];
            let mut time = (max_time - p[6]).max(max_time * 1e-14);
            while time < max_time {
                let end = (2. * time).min(max_time);
                for i in 0..panels {
                    let lo = [i as f64 / panels as f64, (max_time - end) / p[6]];
                    let hi = [(i + 1) as f64 / panels as f64, (max_time - time) / p[6]];
                    let (part, _) = Wiener7::scaled_rule::<true>(
                        &core,
                        &observation,
                        lo,
                        hi,
                        25,
                        25,
                        Reduction::None,
                    )
                    .unwrap();
                    reference.add(
                        part.scale + part.value[0].ln(),
                        std::array::from_fn(|k| part.value[k + 1] / part.value[0]),
                    );
                }
                time = end;
            }
            let expected = std::iter::once(reference.scale + reference.value[0].ln())
                .chain((1..8).map(|i| reference.value[i] / reference.value[0]));
            for (a, b) in std::iter::once(result.log_prob)
                .chain(result.grad_array())
                .zip(expected)
            {
                assert!(
                    (a - b).abs() <= 1e-9 * b.abs().max(1.),
                    "{parameters:?}: {a} vs {b}"
                );
            }
            eprintln!(
                "proposal_reference rt={rt} lower={lower} p={p:?} lp={} scores={:?}",
                result.log_prob,
                result.grad_array()
            );
        }
    }
}

#[cfg(test)]
mod atlasz_replay {
    use super::*;
    #[test]
    fn time_faces_meet_width_scaled_score_budgets() {
        let mut count = 0;
        for line in include_str!("../../../tests/support/wiener7_face_balls.csv")
            .lines()
            .skip(1)
        {
            let x: Vec<f64> = line.split(',').map(|s| s.parse().unwrap()).collect();
            let obs = WienerObservation {
                rt: x[0],
                boundary: if x[8] == 1. {
                    Boundary::Lower
                } else {
                    Boundary::Upper
                },
            };
            let p = Wiener7Params::with_params(x[1], x[2], x[3], x[4], x[6], x[7], x[5]).unwrap();
            let core = Wiener7.core(&obs, &p, 1e-8).unwrap();
            let e = Wiener7.scaled_adaptive(&core, &obs);
            for (i, actual) in std::iter::once(e.log_prob)
                .chain(e.grad_array())
                .enumerate()
            {
                let reference = x[9 + 2 * i];
                assert!(
                    (actual - reference).abs() <= x[10 + 2 * i] + 1e-8 * reference.abs().max(1.),
                    "row {count}, component {i}: {actual} vs {reference}"
                );
            }
            count += 1;
        }
        assert_eq!(count, 13);
    }
    #[test]
    fn density_fallback_does_not_require_score_convergence() {
        for (tau, drift, sd) in [(0., 1., 1e-8), (0.05, -1., 1.)] {
            let event = WienerObservation {
                rt: tau + 1e-6,
                boundary: Boundary::Upper,
            };
            let p = Wiener7Params::with_params(8., tau, 0.25, drift, 1e-8, 0.001, sd).unwrap();
            let result = Wiener7
                .try_log_prob(&event, &p, WienerOptions::new(1e-8))
                .unwrap();
            assert!(result.log_prob < -1e7 && result.grad_array().iter().all(|x| x.is_nan()));
        }
    }
    #[test]
    fn selected_boundary_clipped_time() {
        let event = WienerObservation {
            rt: 0.01,
            boundary: Boundary::Upper,
        };
        let p = Wiener7Params::with_params(0.1, 0., 0.999, 30., 0.001, 0.2, 0.).unwrap();
        let result = Wiener7.try_fused(&event, &p, WienerOptions::new(1e-8));
        assert!(result.is_ok(), "{result:?}");
    }
    #[test]
    fn clipped_opposite_boundary_retains_finite_density_and_scores() {
        let event = WienerObservation {
            rt: 2e-6,
            boundary: Boundary::Upper,
        };
        let p = Wiener7Params::with_params(0.7, 1e-6, 0.25, 30., 1e-8, 0.2, 0.01).unwrap();
        let options = WienerOptions::new(1e-8);
        let e = Wiener7.try_fused(&event, &p, options);
        assert!(e.is_ok(), "{e:?}");
        assert!(Wiener7.try_log_prob(&event, &p, options).is_ok());
    }
}
