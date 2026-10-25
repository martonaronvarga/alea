use super::*;

impl Wiener7 {
    #[cfg(test)]
    const FIXED_QUADRATURE_ORDER: usize = 25;

    pub(in crate::wiener) fn fixed_rule(order: usize) -> (&'static [f64], &'static [f64]) {
        match order {
            1 => (&GL_1_NODES, &GL_1_WTS),
            5 => (&GL_5_NODES, &GL_5_WTS),
            7 => (&GL_7_NODES, &GL_7_WTS),
            15 => (&GL_15_NODES, &GL_15_WTS),
            25 => (&GL_25_NODES, &GL_25_WTS),
            _ => (&GL_25_NODES, &GL_25_WTS),
        }
    }

    #[inline]
    pub(in crate::wiener) fn fixed_interval(
        node: f64,
        weight: f64,
        lo: f64,
        hi: f64,
    ) -> (f64, f64) {
        let width = hi - lo;
        (lo + width * node, width * weight)
    }

    pub(super) fn fixed_integral_1d<F>(lo: f64, hi: f64, order: usize, mut f: F) -> f64
    where
        F: FnMut(f64) -> f64,
    {
        let (nodes, weights) = Self::fixed_rule(order);
        nodes
            .iter()
            .zip(weights.iter())
            .map(|(&node, &weight)| {
                let (x, scaled_weight) = Self::fixed_interval(node, weight, lo, hi);
                scaled_weight * f(x)
            })
            .sum()
    }

    pub(in crate::wiener) fn fixed_integral_6(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
        order: usize,
    ) -> [f64; 6] {
        let mut val = [0.0; 6];
        let mut x = [0.0; 2];

        if core.dim == 1 {
            let (nodes, weights) = Self::fixed_rule(order);
            for (&node, &weight) in nodes.iter().zip(weights.iter()) {
                let (x0, scaled_weight) =
                    Self::fixed_interval(node, weight, core.xmin[0], core.xmax[0]);
                x[0] = x0;
                let mut local = [0.0; 6];
                Self::wiener7_integrand(core, obs, &x[..1], &mut local);
                add_scaled_6(&mut val, &local, scaled_weight);
            }
        } else {
            let (nodes, weights) = Self::fixed_rule(order);
            for (&node0, &weight0) in nodes.iter().zip(weights.iter()) {
                let (x0, scaled_weight0) =
                    Self::fixed_interval(node0, weight0, core.xmin[0], core.xmax[0]);
                x[0] = x0;
                for (&node1, &weight1) in nodes.iter().zip(weights.iter()) {
                    let (x1, scaled_weight1) =
                        Self::fixed_interval(node1, weight1, core.xmin[1], core.xmax[1]);
                    x[1] = x1;
                    let mut local = [0.0; 6];
                    Self::wiener7_integrand(core, obs, &x, &mut local);
                    let scaled_weight = scaled_weight0 * scaled_weight1;
                    for i in 0..6 {
                        val[i] += scaled_weight * local[i];
                    }
                }
            }
        }

        val
    }

    pub(in crate::wiener) fn wiener7_integrand(
        core: &Wiener7Core,
        obs: &WienerObservation,
        x: &[f64],
        fv: &mut [f64; 6],
    ) {
        let (tau, beta) = Self::map_point(core, x);
        if beta <= 0.0 || beta >= 1.0 {
            return;
        }

        let p5 = Wiener5Params::with_params_unchecked(core.alpha, tau, beta, core.delta, core.sv);
        let fused = Wiener5.fused(obs, &p5, core.eps_series);
        if fused.log_prob.is_finite() {
            let dens = fused.log_prob.exp();
            fv[0] = dens;
            fv[1] = dens * fused.grad.alpha;
            fv[2] = dens * fused.grad.tau;
            fv[3] = dens * fused.grad.beta;
            fv[4] = dens * fused.grad.delta;
            fv[5] = dens * fused.grad.s_delta;
        }
    }

    /// Integrate only the density over the hypercube using fixed Gauss-Legendre nodes.
    #[cfg(test)]
    pub(in crate::wiener) fn eval_density_fixed(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
    ) -> f64 {
        self.fixed_integral_6(core, obs, Self::FIXED_QUADRATURE_ORDER)[0]
    }

    pub(in crate::wiener) fn eval_density_fixed_order(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
        order: usize,
    ) -> f64 {
        self.fixed_integral_6(core, obs, order)[0]
    }

    /// Integrate density + 5 inner-parameter gradient components with fixed Gauss-Legendre nodes.
    /// Then compute s_beta and s_tau derivatives separately.
    pub(in crate::wiener) fn eval_fused_fixed_order(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
        order: usize,
    ) -> Wiener7Eval {
        let val = self.fixed_integral_6(core, obs, order);
        let e = self.eval_fused_from_integrals(core, obs, val, order);
        if e.log_prob.is_finite()
            && e.log_prob >= f64::MIN_POSITIVE.ln()
            && e.grad_array().iter().all(|g| g.is_finite())
        {
            e
        } else {
            self.scaled_fixed(core, obs, order)
        }
    }

    /// Integrate density + 5 inner-parameter gradient components with fixed Gauss-Legendre nodes.
    /// Then compute s_beta and s_tau derivatives separately.
    #[cfg(test)]
    pub(in crate::wiener) fn eval_fused_fixed(
        &self,
        core: &Wiener7Core,
        obs: &WienerObservation,
    ) -> Wiener7Eval {
        self.eval_fused_fixed_order(core, obs, Self::FIXED_QUADRATURE_ORDER)
    }

    /// Gauss‑Legendre nodes and weights on `[0,1]` (scaled from `[-1,1]`).
    /// n: number of points.
    pub fn gauss_legendre_01(n: usize) -> (Vec<f64>, Vec<f64>) {
        // Legendre nodes/weights on [-1,1] using symmetric Newton method
        let mut nodes = vec![0.0; n];
        let mut weights = vec![0.0; n];
        let m = n.div_ceil(2);
        for i in 1..=m {
            // Initial guess
            let mut x = (PI * (i as f64 - 0.25) / (n as f64 + 0.5)).cos();
            let mut dp = 0.0;
            // Iterate to find root
            for _ in 0..20 {
                let (p1, p2) = Self::legendre_poly(n as u32, x);
                dp = (n as f64) * (x * p1 - p2) / (x * x - 1.0);
                x -= p1 / dp;
                if x.abs() < 1e-15 {
                    break;
                }
            }
            // Scale to [0,1]
            let t = 0.5 * (x + 1.0);
            let w = 1.0 / ((1.0 - x * x) * dp * dp);
            nodes[i - 1] = t;
            weights[i - 1] = w;
            nodes[n - i] = 1.0 - t;
            weights[n - i] = w;
        }
        (nodes, weights)
    }

    /// Evaluate Legendre polynomial of degree n at x.
    fn legendre_poly(n: u32, x: f64) -> (f64, f64) {
        if n == 0 {
            return (1.0, 0.0);
        }
        if n == 1 {
            return (x, 1.0);
        }
        let mut p0 = 1.0;
        let mut p1 = x;
        let mut p = 0.0;
        for k in 1..n {
            let p2 = ((2 * k + 1) as f64 * x * p1 - k as f64 * p0) / (k + 1) as f64;
            p0 = p1;
            p1 = p2;
            p = p2;
        }
        // Return P_n(x) and P_{n-1}(x)
        (p, p0)
    }
}
