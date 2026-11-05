//! Private four-observation tiles. Dispatch covers the whole likelihood, not
//! individual products. Public data remain row-major; tiles are prepared once.
use super::{CompensatedSum, E, LogitEval, NormalEval, PoissonEval, RegressionData};

pub(super) const TILE: usize = 4;

pub(super) struct Design {
    columns: usize,
    tiles: std::sync::Arc<[[f64; TILE]]>,
    zero_offsets: bool,
    pub(super) broadcast: bool,
    pub(super) plan: super::RegressionPlan,
}
impl Design {
    pub(super) fn new<Y>(data: &RegressionData<Y>) -> Result<Self, E> {
        let zero_offsets = data.offsets.iter().all(|v| *v == 0.);
        let broadcast = data.columns == 1 && data.design.iter().all(|v| *v == 1.) && zero_offsets;
        let len = if broadcast {
            0
        } else {
            data.rows()
                .div_ceil(TILE)
                .checked_mul(data.columns)
                .ok_or(E::Shape)?
        };
        if len
            .checked_mul(size_of::<[f64; TILE]>())
            .is_none_or(|n| n > isize::MAX as usize)
        {
            return Err(E::Shape);
        }
        let tiles = if broadcast {
            std::sync::Arc::from([])
        } else {
            std::sync::Arc::clone(data.packed.get_or_init(|| {
                let mut tiles = vec![[0.; TILE]; len];
                for (i, row) in data.design.chunks(data.columns.max(1)).enumerate() {
                    for (j, &x) in row.iter().enumerate() {
                        tiles[(i / TILE) * data.columns + j][i % TILE] = x;
                    }
                }
                tiles.into()
            }))
        };
        Ok(Self {
            columns: data.columns,
            tiles,
            zero_offsets,
            broadcast,
            plan: super::RegressionPlan::new(
                super::RegressionExecution::Auto,
                data.rows(),
                data.columns,
                len * size_of::<[f64; TILE]>(),
                broadcast,
            ),
        })
    }

    pub(super) fn set_execution(&mut self, execution: super::RegressionExecution) {
        let arithmetic = self.plan.arithmetic();
        self.plan = super::RegressionPlan::new(
            execution,
            self.plan.rows(),
            self.columns,
            self.plan.packed_design_bytes(),
            self.broadcast,
        );
        self.plan.arithmetic = arithmetic;
    }

    #[inline(always)]
    fn predictors_policy<S: pulp::Simd, const STANDARD: bool>(
        &self,
        simd: S,
        tile: usize,
        beta: &[f64],
        offsets: &[f64],
    ) -> [f64; TILE] {
        if !STANDARD {
            return self.predictors(simd, tile, beta, offsets);
        }
        if self.broadcast {
            return [beta[0] + 0.; TILE];
        }
        let mut result = [0.; TILE];
        let columns = &self.tiles[tile * self.columns..][..self.columns];
        let (out, tail) = S::as_mut_simd_f64s(&mut result);
        let (offsets, ot) = S::as_simd_f64s(offsets);
        for (packet, (out, offset)) in out.iter_mut().zip(offsets).enumerate() {
            let mut sum = *offset;
            for (column, &b) in columns.iter().zip(beta) {
                sum = simd.add_f64s(
                    sum,
                    simd.mul_f64s(S::as_simd_f64s(column).0[packet], simd.splat_f64s(b)),
                );
            }
            *out = sum;
        }
        for (i, (out, offset)) in tail.iter_mut().zip(ot).enumerate() {
            *out = columns
                .iter()
                .zip(beta)
                .fold(*offset, |sum, (column, &b)| sum + column[i] * b);
        }
        result
    }

    #[inline(always)]
    pub(super) fn predictors<S: pulp::Simd>(
        &self,
        simd: S,
        tile: usize,
        beta: &[f64],
        offsets: &[f64],
    ) -> [f64; TILE] {
        if self.broadcast {
            return [beta[0] + 0.; TILE];
        }
        let mut result = [0.; TILE];
        let original_offsets = offsets;
        let columns = &self.tiles[tile * self.columns..][..self.columns];
        let beta = &beta[..self.columns];
        let (out, tail) = S::as_mut_simd_f64s(&mut result);
        let (offsets, ot) = S::as_simd_f64s(offsets);
        for (packet, (out, offset)) in out.iter_mut().zip(offsets).enumerate() {
            let mut sum = LaneSum::new(simd);
            // Initializing from a finite offset needs no error-free addition.
            // Keep +0 addition so signed zero matches the ordered scalar path.
            sum.sum = simd.add_f64s(simd.splat_f64s(0.), *offset);
            let skip = usize::from(self.zero_offsets && !beta.is_empty());
            if skip != 0 {
                let x = S::as_simd_f64s(&columns[0]).0[packet];
                let b = simd.splat_f64s(beta[0]);
                let product = simd.mul_f64s(x, b);
                sum.sum = simd.add_f64s(simd.splat_f64s(0.), product);
                sum.correction = simd.add_f64s(
                    simd.splat_f64s(0.),
                    simd.mul_add_f64s(x, b, simd.neg_f64s(product)),
                );
            }
            for (column, b) in columns[skip..].iter().zip(&beta[skip..]) {
                let x = S::as_simd_f64s(column).0[packet];
                let b = simd.splat_f64s(*b);
                let product = simd.mul_f64s(x, b);
                sum.add_twosum(simd, product);
                let finite =
                    simd.less_than_f64s(simd.abs_f64s(product), simd.splat_f64s(f64::INFINITY));
                let residual = simd.mul_add_f64s(x, b, simd.neg_f64s(product));
                sum.correction = simd.add_f64s(
                    sum.correction,
                    simd.select_f64s(finite, residual, simd.splat_f64s(0.)),
                );
            }
            *out = sum.value(simd);
        }
        // Also supports a future SIMD width greater than TILE without indexing
        // an empty packet slice. Current scalar/SSE2/AVX2 widths divide TILE.
        for (i, (out, offset)) in tail.iter_mut().zip(ot).enumerate() {
            let mut sum = CompensatedSum::default();
            sum.add(*offset);
            for (column, b) in columns.iter().zip(beta) {
                sum.add_product(column[i], *b);
            }
            *out = sum.total();
        }
        // TwoSum can overflow a virtual addend at representability limits even
        // when the ordered Neumaier dot is finite. Retry only affected lanes
        // with the original algorithm, including its product residuals.
        for (i, out) in result.iter_mut().enumerate() {
            if !out.is_finite() {
                let mut sum = CompensatedSum::default();
                sum.add(original_offsets[i]);
                for (column, b) in columns.iter().zip(beta) {
                    sum.add_product(column[i], *b);
                }
                *out = sum.total();
            }
        }
        result
    }
}

// Neumaier independently in every lane. No discarded product residuals, plain
// horizontal sum, or reassociation of predictor coefficient order.
pub(super) struct LaneSum<S: pulp::Simd> {
    sum: S::f64s,
    correction: S::f64s,
}
impl<S: pulp::Simd> LaneSum<S> {
    #[inline(always)]
    fn add_policy<const STANDARD: bool>(&mut self, simd: S, value: S::f64s) {
        if STANDARD {
            self.sum = simd.add_f64s(self.sum, value);
        } else {
            self.add(simd, value);
        }
    }
    #[inline(always)]
    fn finish_policy<const STANDARD: bool>(&self, total: &mut Sum<STANDARD>) {
        for &x in pulp::bytemuck::cast_slice::<S::f64s, f64>(std::slice::from_ref(&self.sum)) {
            total.add(x);
        }
        if !STANDARD {
            for &x in
                pulp::bytemuck::cast_slice::<S::f64s, f64>(std::slice::from_ref(&self.correction))
            {
                total.add(x);
            }
        }
    }
    #[inline(always)]
    fn add_twosum(&mut self, simd: S, value: S::f64s) {
        let next = simd.add_f64s(self.sum, value);
        let virtual_value = simd.sub_f64s(next, self.sum);
        let error = simd.add_f64s(
            simd.sub_f64s(self.sum, simd.sub_f64s(next, virtual_value)),
            simd.sub_f64s(value, virtual_value),
        );
        self.correction = simd.add_f64s(self.correction, error);
        self.sum = next;
    }
    #[inline(always)]
    fn new(simd: S) -> Self {
        Self {
            sum: simd.splat_f64s(0.),
            correction: simd.splat_f64s(0.),
        }
    }
    #[inline(always)]
    fn add(&mut self, simd: S, value: S::f64s) {
        let next = simd.add_f64s(self.sum, value);
        let large = simd.greater_than_or_equal_f64s(simd.abs_f64s(self.sum), simd.abs_f64s(value));
        let error = simd.select_f64s(
            large,
            simd.add_f64s(simd.sub_f64s(self.sum, next), value),
            simd.add_f64s(simd.sub_f64s(value, next), self.sum),
        );
        self.correction = simd.add_f64s(self.correction, error);
        self.sum = next;
    }
    #[inline(always)]
    fn value(&self, simd: S) -> S::f64s {
        simd.add_f64s(self.sum, self.correction)
    }
    #[cfg(test)]
    #[inline(always)]
    fn finish(&self, total: &mut CompensatedSum) {
        // Add heads AND residuals separately, not rounded lane totals. This
        // retains small corrections when large lane heads subsequently cancel.
        for &x in pulp::bytemuck::cast_slice::<S::f64s, f64>(std::slice::from_ref(&self.sum)) {
            total.add(x);
        }
        for &x in pulp::bytemuck::cast_slice::<S::f64s, f64>(std::slice::from_ref(&self.correction))
        {
            total.add(x);
        }
    }
}

// Compile-time policy removes compensation work from Standard specializations;
// selecting a policy occurs once per evaluation, not per observation.
#[derive(Default)]
pub(super) struct Sum<const STANDARD: bool> {
    plain: f64,
    compensated: CompensatedSum,
}
impl<const STANDARD: bool> Sum<STANDARD> {
    #[inline(always)]
    pub(super) fn add(&mut self, x: f64) {
        if STANDARD {
            self.plain += x;
        } else {
            self.compensated.add(x);
        }
    }
    #[inline(always)]
    pub(super) fn add_product(&mut self, x: f64, y: f64) {
        if STANDARD {
            self.plain += x * y;
        } else {
            self.compensated.add_product(x, y);
        }
    }
    #[inline(always)]
    pub(super) fn total(&self) -> f64 {
        if STANDARD {
            self.plain
        } else {
            self.compensated.total()
        }
    }
    #[inline(always)]
    pub(super) fn into_compensated(self) -> CompensatedSum {
        if STANDARD {
            let mut sum = CompensatedSum::default();
            sum.add(self.plain);
            sum
        } else {
            self.compensated
        }
    }
}

// Bounded range-reduced exp for the ordinary negative arguments of logit.
// Sixteen exact power-of-two bins with a rounded 2^(j/16) table reduce the
// remainder to |r| <= ln(2)/32. Degree-7 Taylor truncation < 1.3e-18 before
// f64/table rounding. This shortens the dependency chain without a tail cutoff.
// Large/tiny tails and nonfinite arguments use the existing scalar libm, not a
// cutoff, clipping, or a fabricated zero. See the independent Decimal fixture.
#[inline(always)]
fn exp_negative<S: pulp::Simd>(simd: S, x: &[f64; TILE]) -> [f64; TILE] {
    exp_packet::<S, false>(simd, x)
}

#[inline(always)]
fn exp_packet<S: pulp::Simd, const POSITIVE: bool>(simd: S, x: &[f64; TILE]) -> [f64; TILE] {
    // The table/polynomial is a vector kernel, not a faster scalar exp.
    // Scalar FMA may call software support routines; use libm also when the
    // architecture's vectors cannot fit a complete observation tile.
    if S::IS_SCALAR || std::mem::size_of::<S::f64s>() > std::mem::size_of::<[f64; TILE]>() {
        return x.map(f64::exp);
    }
    const POWERS: [f64; 16] = [
        1.,
        1.0442737824274138,
        1.0905077326652577,
        1.1387886347566916,
        1.189207115002721,
        1.241857812073484,
        1.2968395546510096,
        1.3542555469368927,
        std::f64::consts::SQRT_2,
        1.4768261459394993,
        1.5422108254079407,
        1.6104903319492543,
        1.681792830507429,
        1.7562521603732995,
        1.8340080864093424,
        1.9152065613971474,
    ];
    let upper = if POSITIVE { 32. } else { 0. };
    let mut reduced = [0.; TILE];
    let mut scales = [1.; TILE];
    for i in 0..TILE {
        if (-32. ..=upper).contains(&x[i]) {
            const MAGIC: f64 = 6_755_399_441_055_744.;
            let n = (x[i] * (16. * std::f64::consts::LOG2_E) + MAGIC) - MAGIC;
            reduced[i] = n.mul_add(-0.043_321_698_773_070_24, x[i]);
            reduced[i] = n.mul_add(-1.192_634_330_794_117_3e-11, reduced[i]);
            // Guarded integer n is in [-739,739]. Arithmetic shift/remainder
            // split it into an exponent and a nonnegative table index.
            let n = n as i32;
            scales[i] =
                f64::from_bits(((1023 + (n >> 4)) as u64) << 52) * POWERS[(n & 15) as usize];
        }
    }
    let mut out = [0.; TILE];
    let (out_packets, out_tail) = S::as_mut_simd_f64s(&mut out);
    let (r, rt) = S::as_simd_f64s(&reduced);
    let (scale, st) = S::as_simd_f64s(&scales);
    for ((out, r), scale) in out_packets.iter_mut().zip(r).zip(scale) {
        let mut p = simd.splat_f64s(1. / 5_040.);
        // These seven operations are deliberately explicit. The pinned nightly
        // retained an IntoIter loop and spilled its coefficients in this hot
        // kernel. Preserve exactly the original Horner/FMA order.
        p = simd.mul_add_f64s(p, *r, simd.splat_f64s(1. / 720.));
        p = simd.mul_add_f64s(p, *r, simd.splat_f64s(1. / 120.));
        p = simd.mul_add_f64s(p, *r, simd.splat_f64s(1. / 24.));
        p = simd.mul_add_f64s(p, *r, simd.splat_f64s(1. / 6.));
        p = simd.mul_add_f64s(p, *r, simd.splat_f64s(0.5));
        p = simd.mul_add_f64s(p, *r, simd.splat_f64s(1.));
        p = simd.mul_add_f64s(p, *r, simd.splat_f64s(1.));
        *out = simd.mul_f64s(p, *scale);
    }
    for ((out, r), scale) in out_tail.iter_mut().zip(rt).zip(st) {
        *out = r.exp() * scale;
    }
    for i in 0..TILE {
        if !(-32. ..=upper).contains(&x[i]) {
            out[i] = x[i].exp();
        }
    }
    out
}

// log1p on the bounded exp(-abs(eta)) domain. With t=x/(2+x),
// log1p(x)=2*(t+t^3/3+...+t^37/37)+R, |t|<=1/3 and
// |R|<=2*(1/3)^39/(39*(1-1/9))<1.5e-20, before rounding.
// Four independent Horner groups shorten the polynomial dependency chain.
// Scalar targets and subnormal/out-of-domain lanes retain libm, including tails.
#[inline(always)]
fn log1p_unit<S: pulp::Simd>(simd: S, x: &[f64; TILE]) -> [f64; TILE] {
    if S::IS_SCALAR {
        return x.map(f64::ln_1p);
    }
    let mut result = [0.; TILE];
    let (out, tail) = S::as_mut_simd_f64s(&mut result);
    let (input, input_tail) = S::as_simd_f64s(x);
    for (out, x) in out.iter_mut().zip(input) {
        let two = simd.splat_f64s(2.);
        let t = simd.div_f64s(*x, simd.add_f64s(two, *x));
        let z = simd.mul_f64s(t, t);
        let z2 = simd.mul_f64s(z, z);
        let z4 = simd.mul_f64s(z2, z2);
        let group = |a, b, c, d| {
            let p = simd.mul_add_f64s(z, simd.splat_f64s(d), simd.splat_f64s(c));
            let p = simd.mul_add_f64s(z, p, simd.splat_f64s(b));
            simd.mul_add_f64s(z, p, simd.splat_f64s(a))
        };
        let a = group(1. / 3., 1. / 5., 1. / 7., 1. / 9.);
        let b = group(1. / 11., 1. / 13., 1. / 15., 1. / 17.);
        let c = group(1. / 19., 1. / 21., 1. / 23., 1. / 25.);
        let d = group(1. / 27., 1. / 29., 1. / 31., 1. / 33.);
        let p = simd.mul_add_f64s(z, simd.splat_f64s(1. / 37.), simd.splat_f64s(1. / 35.));
        let p = simd.mul_add_f64s(z4, p, d);
        let p = simd.mul_add_f64s(z4, p, c);
        let p = simd.mul_add_f64s(z4, p, b);
        let p = simd.mul_add_f64s(z4, p, a);
        let twice_t = simd.mul_f64s(two, t);
        *out = simd.mul_add_f64s(simd.mul_f64s(twice_t, z), p, twice_t);
    }
    for (out, x) in tail.iter_mut().zip(input_tail) {
        *out = x.ln_1p();
    }
    for (out, x) in result.iter_mut().zip(x) {
        if !(2. * f64::MIN_POSITIVE..=1.).contains(x) {
            *out = x.ln_1p();
        }
    }
    result
}

impl NormalEval<'_> {
    #[inline(always)]
    pub(super) fn batch<S: pulp::Simd, const BROADCAST: bool, const STANDARD: bool>(
        self,
        simd: S,
        inverse: f64,
    ) -> Result<f64, E> {
        let Self {
            data,
            design,
            x,
            g,
            scores,
            broadcast,
        } = self;
        let p = data.columns;
        let normalizer = -x[p].ln() - 0.5 * std::f64::consts::TAU.ln();
        let mut quadratic = LaneSum::new(simd);
        let mut sigma = LaneSum::new(simd);
        let mut mean = LaneSum::new(simd);
        let mut total = Sum::<STANDARD>::default();
        let mut sigma_score = Sum::<STANDARD>::default();
        let mut mean_score = Sum::<STANDARD>::default();
        scores.reset();
        let (observations, tail) = data.observations.as_chunks::<TILE>();
        let (offsets, _) = data.offsets.as_chunks::<TILE>();
        for (tile, (y, offset)) in observations.iter().zip(offsets).enumerate() {
            let eta = if BROADCAST {
                [x[0] + 0.; TILE]
            } else {
                design.predictors_policy::<S, STANDARD>(simd, tile, &x[..p], offset)
            };
            let mut residual = [0.; TILE];
            let (residuals, _) = S::as_mut_simd_f64s(&mut residual);
            let (y, _) = S::as_simd_f64s(y);
            let (eta, _) = S::as_simd_f64s(&eta);
            let inv = simd.splat_f64s(inverse);
            for ((out, y), eta) in residuals.iter_mut().zip(y).zip(eta) {
                let z = simd.mul_f64s(simd.sub_f64s(*y, *eta), inv);
                quadratic.add_policy::<STANDARD>(
                    simd,
                    simd.mul_f64s(simd.mul_f64s(simd.splat_f64s(-0.5), z), z),
                );
                sigma.add_policy::<STANDARD>(
                    simd,
                    simd.mul_f64s(simd.sub_f64s(simd.mul_f64s(z, z), simd.splat_f64s(1.)), inv),
                );
                *out = simd.mul_f64s(z, inv);
                if BROADCAST {
                    mean.add_policy::<STANDARD>(simd, *out);
                }
            }
            if !BROADCAST && p != 0 {
                let start = tile * TILE * p;
                scores.add_tile_policy::<S, STANDARD>(
                    simd,
                    std::array::from_fn(|r| &data.design[start + r * p..start + (r + 1) * p]),
                    &residual,
                );
            }
        }
        quadratic.finish_policy(&mut total);
        sigma.finish_policy(&mut sigma_score);
        if BROADCAST {
            mean.finish_policy(&mut mean_score);
        }
        for (r, &y) in tail.iter().enumerate() {
            let i = observations.len() * TILE + r;
            let eta = if BROADCAST {
                x[0] + 0.
            } else {
                data.predictor_with_arithmetic::<STANDARD>(i, &x[..p])
            };
            let z = (y - eta) * inverse;
            total.add(-0.5 * z * z);
            sigma_score.add((z * z - 1.) * inverse);
            let r = z * inverse;
            if BROADCAST {
                mean_score.add(r);
            } else {
                scores.add_scaled_policy::<S, STANDARD>(simd, &data.design[i * p..(i + 1) * p], r);
            }
        }
        total.add_product(normalizer, data.rows() as f64);
        if BROADCAST {
            scores.sums[0] = mean_score.total();
            scores.corrections[0] = 0.;
        }
        // Lane-wise partials can overflow before cancellation between lanes.
        // Retry the original ordered/divided calculation, not a clipped answer.
        if sigma_score.total().is_finite()
            && let Ok(value) = scores.finish(total.into_compensated(), &mut g[..p])
        {
            g[p] = sigma_score.total();
            return Ok(value);
        }
        NormalEval {
            data,
            design,
            broadcast,
            x,
            g,
            scores,
        }
        .evaluate::<S, false, BROADCAST, STANDARD>(simd, inverse)
    }
}

impl LogitEval<'_> {
    #[inline(always)]
    pub(super) fn batch<S: pulp::Simd, const STANDARD: bool>(self, simd: S) -> Result<f64, E> {
        let Self {
            data,
            design,
            x,
            g,
            scores,
        } = self;
        let mut total = Sum::<STANDARD>::default();
        let mut likelihood = LaneSum::new(simd);
        scores.reset();
        let (ys, tail) = data.observations.as_chunks::<TILE>();
        let (offsets, _) = data.offsets.as_chunks::<TILE>();
        for (tile, (ys, offset)) in ys.iter().zip(offsets).enumerate() {
            let eta = design.predictors_policy::<S, STANDARD>(simd, tile, x, offset);
            if eta.iter().any(|e| !e.is_finite()) {
                return Err(E::Numerical);
            }
            let negative_abs = [-eta[0].abs(), -eta[1].abs(), -eta[2].abs(), -eta[3].abs()];
            let e = exp_negative(simd, &negative_abs);
            let logs = log1p_unit(simd, &e);
            let mut residuals = [0.; TILE];
            let signs = [
                if ys[0] { -1. } else { 1. },
                if ys[1] { -1. } else { 1. },
                if ys[2] { -1. } else { 1. },
                if ys[3] { -1. } else { 1. },
            ];
            let (es, _) = S::as_simd_f64s(&e);
            let (etas, _) = S::as_simd_f64s(&eta);
            let (logs, _) = S::as_simd_f64s(&logs);
            let (signs, _) = S::as_simd_f64s(&signs);
            let (rs, rt) = S::as_mut_simd_f64s(&mut residuals);
            for ((((r, e), eta), log), sign) in rs.iter_mut().zip(es).zip(etas).zip(logs).zip(signs)
            {
                let z = simd.mul_f64s(*eta, *sign);
                likelihood.add_policy::<STANDARD>(
                    simd,
                    simd.neg_f64s(simd.add_f64s(simd.max_f64s(z, simd.splat_f64s(0.)), *log)),
                );
                let numerator = simd.select_f64s(
                    simd.greater_than_or_equal_f64s(z, simd.splat_f64s(0.)),
                    simd.splat_f64s(1.),
                    *e,
                );
                let slope = simd.div_f64s(numerator, simd.add_f64s(simd.splat_f64s(1.), *e));
                *r = simd.mul_f64s(simd.neg_f64s(*sign), slope);
            }
            // Future SIMD widths larger than TILE retain the scalar calculation.
            for (r, out) in rt.iter_mut().enumerate() {
                let z = if ys[r] { -eta[r] } else { eta[r] };
                total.add(-(z.max(0.) + e[r].ln_1p()));
                let slope = if z >= 0. {
                    1. / (1. + e[r])
                } else {
                    e[r] / (1. + e[r])
                };
                *out = if ys[r] { slope } else { -slope };
            }
            let p = data.columns;
            let start = tile * TILE * p;
            scores.add_tile_policy::<S, STANDARD>(
                simd,
                std::array::from_fn(|r| &data.design[start + r * p..start + (r + 1) * p]),
                &residuals,
            );
        }
        likelihood.finish_policy(&mut total);
        for (r, &y) in tail.iter().enumerate() {
            let i = ys.len() * TILE + r;
            let eta = data.predictor_with_arithmetic::<STANDARD>(i, x);
            if !eta.is_finite() {
                return Err(E::Numerical);
            }
            let (value, residual) =
                super::Bernoulli::logit_log_prob_score(eta, y).map_err(super::error)?;
            total.add(value);
            scores.add_scaled_policy::<S, STANDARD>(
                simd,
                &data.design[i * data.columns..(i + 1) * data.columns],
                residual,
            );
        }
        scores.finish(total.into_compensated(), g)
    }
}

impl PoissonEval<'_> {
    #[inline(always)]
    pub(super) fn batch<S: pulp::Simd, const STANDARD: bool>(self, simd: S) -> Result<f64, E> {
        let Self {
            data,
            design,
            normalizers,
            x,
            g,
            scores,
        } = self;
        scores.reset();
        let mut total = Sum::<STANDARD>::default();
        let mut likelihood = LaneSum::new(simd);
        let (ys, tail) = data.observations.as_chunks::<TILE>();
        let (offsets, _) = data.offsets.as_chunks::<TILE>();
        for (tile, (ys, offset)) in ys.iter().zip(offsets).enumerate() {
            let eta = design.predictors_policy::<S, STANDARD>(simd, tile, x, offset);
            if eta.iter().any(|v| !v.is_finite()) {
                return Err(E::Numerical);
            }
            let rates = exp_packet::<S, true>(simd, &eta);
            if rates.iter().any(|v| !v.is_finite()) {
                return Err(E::Numerical);
            }
            let mut residuals = [0.; TILE];
            let counts = ys.map(|y| y as f64);
            let (es, et) = S::as_simd_f64s(&eta);
            let (rates, rate_tail) = S::as_simd_f64s(&rates);
            let (counts, count_tail) = S::as_simd_f64s(&counts);
            let (constants, constant_tail) = S::as_simd_f64s(&normalizers[tile * TILE..][..TILE]);
            let (rs, rt) = S::as_mut_simd_f64s(&mut residuals);
            for ((((r, eta), rate), count), constant) in
                rs.iter_mut().zip(es).zip(rates).zip(counts).zip(constants)
            {
                likelihood.add_policy::<STANDARD>(
                    simd,
                    simd.sub_f64s(simd.sub_f64s(simd.mul_f64s(*count, *eta), *rate), *constant),
                );
                *r = simd.sub_f64s(*count, *rate);
            }
            for ((((out, eta), rate), y), constant) in rt
                .iter_mut()
                .zip(et)
                .zip(rate_tail)
                .zip(count_tail)
                .zip(constant_tail)
            {
                total.add(y * eta - rate - constant);
                *out = y - rate;
            }
            let p = data.columns;
            let start = tile * TILE * p;
            scores.add_tile_policy::<S, STANDARD>(
                simd,
                std::array::from_fn(|r| &data.design[start + r * p..start + (r + 1) * p]),
                &residuals,
            );
        }
        likelihood.finish_policy(&mut total);
        for (r, &y) in tail.iter().enumerate() {
            let i = ys.len() * TILE + r;
            let eta = data.predictor_with_arithmetic::<STANDARD>(i, x);
            let rate = eta.exp();
            if !eta.is_finite() || !rate.is_finite() {
                return Err(E::Numerical);
            }
            total.add(y as f64 * eta - rate - normalizers[i]);
            scores.add_scaled_policy::<S, STANDARD>(
                simd,
                &data.design[i * data.columns..(i + 1) * data.columns],
                y as f64 - rate,
            );
        }
        scores.finish(total.into_compensated(), g)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn zero_offset_initialization_preserves_product_residuals_and_failure_admission() {
        use super::*;
        for p in 0..18 {
            for first in [-0., 1e16, 1e-300, f64::from_bits(1), f64::MAX] {
                let matrix = (0..TILE * p)
                    .map(|i| {
                        if i % p.max(1) == 0 {
                            first
                        } else {
                            [1., -1e16, 0.125][i % 3]
                        }
                    })
                    .collect();
                let data = RegressionData::new(p, matrix, vec![0.; TILE]).unwrap();
                let design = Design::new(&data).unwrap();
                let beta: Vec<_> = (0..p).map(|j| if j % 3 == 0 { 2. } else { 1. }).collect();
                struct Run<'a>(&'a Design, &'a [f64]);
                impl pulp::WithSimd for Run<'_> {
                    type Output = [f64; TILE];
                    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                        self.0.predictors(simd, 0, self.1, &[0.; TILE])
                    }
                }
                let actual = pulp::Arch::new().dispatch(Run(&design, &beta));
                for (i, a) in actual.into_iter().enumerate() {
                    let expected = data.linear_predictor(i, &beta);
                    if expected.is_finite() {
                        assert_eq!(a.to_bits(), expected.to_bits(), "p={p},first={first}");
                    } else {
                        assert!(!a.is_finite());
                    }
                }
            }
        }
    }

    #[test]
    fn bounded_log1p_matches_decimal_dense_grid_and_exact_scalar_tails() {
        use super::*;
        struct Log([f64; TILE]);
        impl pulp::WithSimd for Log {
            type Output = [f64; TILE];
            fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                log1p_unit(simd, &self.0)
            }
        }
        for line in include_str!("../../../tests/support/regression-log1p.csv")
            .lines()
            .filter(|line| !line.starts_with('#'))
        {
            let (x, expected) = line.split_once(',').unwrap();
            let x = f64::from_bits(u64::from_str_radix(x, 16).unwrap());
            let expected = u64::from_str_radix(expected, 16).unwrap();
            for actual in [
                pulp::Arch::new().dispatch(Log([x; TILE])),
                log1p_unit(pulp::Scalar, &[x; TILE]),
            ] {
                assert!(
                    actual.iter().all(|a| a.to_bits().abs_diff(expected) <= 4),
                    "x={x},actual={actual:?}"
                );
            }
        }
        for i in 0..50_000 {
            let x = std::array::from_fn(|r| (i * TILE + r) as f64 / 200_000.);
            for (x, actual) in x.into_iter().zip(pulp::Arch::new().dispatch(Log(x))) {
                assert!(actual.to_bits().abs_diff(x.ln_1p().to_bits()) <= 4, "x={x}");
            }
        }
        for x in [-1., -0.5, -0., 0., f64::from_bits(1), 1.1, f64::INFINITY] {
            for actual in pulp::Arch::new().dispatch(Log([x; TILE])) {
                assert_eq!(actual.to_bits(), x.ln_1p().to_bits());
            }
        }
        let mixed = [f64::NAN, f64::from_bits(3), 0.5, f64::INFINITY];
        let actual = pulp::Arch::new().dispatch(Log(mixed));
        assert!(actual[0].is_nan());
        assert_eq!(actual[1].to_bits(), mixed[1].ln_1p().to_bits());
        assert!(actual[2].to_bits().abs_diff(0.5_f64.ln_1p().to_bits()) <= 4);
        assert_eq!(actual[3], f64::INFINITY);
    }

    #[test]
    fn prepared_tiles_share_only_immutable_design_across_responses_offsets_and_families() {
        use super::*;
        let data = RegressionData::new(2, vec![1., -1., 1., 2., 1., 3.], vec![0.; 3]).unwrap();
        let original = Design::new(&data).unwrap();
        let changed = data
            .with_observations(vec![true, false, true])
            .unwrap()
            .with_offsets(vec![0.5; 3])
            .unwrap();
        let next = Design::new(&changed).unwrap();
        assert!(std::sync::Arc::ptr_eq(&original.tiles, &next.tiles));
        assert_eq!(
            next.predictors(pulp::Scalar, 0, &[1., 2.], &[0.5; TILE]),
            [-0.5, 5.5, 7.5, 0.5]
        );
        assert!(data.with_observations(vec![1_u64]).is_err());
        assert_eq!(data.observations(), &[0.; 3]);
        let separate = RegressionData::new(2, data.design().to_vec(), vec![0.; 3]).unwrap();
        assert!(!std::sync::Arc::ptr_eq(
            &original.tiles,
            &Design::new(&separate).unwrap().tiles
        ));
        let unit = RegressionData::new(1, vec![1.; 3], vec![0.; 3]).unwrap();
        assert!(Design::new(&unit).unwrap().tiles.is_empty());
        // A broadcast binding must not cache an empty pack needed by a later
        // non-broadcast binding of the same unit-column design.
        let offset = unit
            .with_observations(vec![2_u64; 3])
            .unwrap()
            .with_offsets(vec![1.; 3])
            .unwrap();
        assert_eq!(Design::new(&offset).unwrap().tiles.len(), 1);
    }
    #[test]
    fn native_batch_reporting_does_not_allocate_unused_coefficient_scores() {
        use super::super::*;
        let mut b = ModelBuilder::new();
        let beta: Vec<_> = (0..64)
            .map(|j| b.real(&format!("b{j}")).unwrap().expr())
            .collect();
        let sigma = b.positive("sigma").unwrap();
        let data = RegressionData::new(64, vec![1.; 128], vec![0., 1.]).unwrap();
        let normal = NormalRegression::new(&beta, &sigma, data.clone()).unwrap();
        let logit = BernoulliLogitRegression::new(
            &beta,
            data.with_observations(vec![false, true]).unwrap(),
        )
        .unwrap();
        let poisson =
            PoissonLogRegression::new(&beta, data.with_observations(vec![0_u64, 2]).unwrap())
                .unwrap();
        assert_eq!(
            allocation_counter::measure(|| {
                std::hint::black_box(normal.batch_workspace());
                std::hint::black_box(logit.batch_workspace());
                std::hint::black_box(poisson.batch_workspace());
            })
            .count_total,
            0
        );
    }
    use super::*;

    #[test]
    fn scalar_exponential_keeps_libm_in_range_and_at_limits() {
        for x in [
            [-32., -1., -0., 0.],
            [0.01, 1., 31.9, 32.],
            [-745., 710., f64::NEG_INFINITY, f64::INFINITY],
        ] {
            let expected = x.map(|v: f64| v.exp().to_bits());
            assert_eq!(
                exp_packet::<_, true>(pulp::Scalar, &x).map(f64::to_bits),
                expected
            );
            assert_eq!(exp_negative(pulp::Scalar, &x).map(f64::to_bits), expected);
        }
        assert!(
            exp_packet::<_, true>(pulp::Scalar, &[f64::NAN; TILE])
                .iter()
                .all(|v| v.is_nan())
        );
    }

    #[test]
    fn signed_exponential_matches_decimal_and_mixed_scalar_tails() {
        struct Signed([f64; TILE]);
        impl pulp::WithSimd for Signed {
            type Output = [f64; TILE];
            fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
                exp_packet::<S, true>(simd, &self.0)
            }
        }
        for fixture in [
            include_str!("../../../tests/support/regression-exp.csv"),
            include_str!("../../../tests/support/regression-exp-positive.csv"),
        ] {
            for line in fixture.lines().filter(|line| !line.starts_with('#')) {
                let (x, expected) = line.split_once(',').unwrap();
                let x = f64::from_bits(u64::from_str_radix(x, 16).unwrap());
                let expected = u64::from_str_radix(expected, 16).unwrap();
                for actual in [
                    pulp::Arch::new().dispatch(Signed([x; TILE])),
                    exp_packet::<_, true>(pulp::Scalar, &[x; TILE]),
                ] {
                    assert!(
                        actual.iter().all(|a| a.to_bits().abs_diff(expected) <= 4),
                        "x={x}"
                    );
                }
            }
        }
        for i in 0..10_000 {
            let x = std::array::from_fn(|r| -32. + 64. * (i * TILE + r) as f64 / 40_000.);
            let actual = pulp::Arch::new().dispatch(Signed(x));
            for (x, a) in x.into_iter().zip(actual) {
                assert!(a.to_bits().abs_diff(x.exp().to_bits()) <= 4, "x={x}");
            }
        }
        let x = [-745., 710., 0., f64::NAN];
        let a = pulp::Arch::new().dispatch(Signed(x));
        assert_eq!(a[0].to_bits(), x[0].exp().to_bits());
        assert_eq!(a[1], f64::INFINITY);
        assert_eq!(a[2], 1.);
        assert!(a[3].is_nan());
    }

    #[test]
    fn normal_reciprocal_fast_path_preserves_extreme_scale_admission() {
        for scale in [
            f64::from_bits(1),
            f64::MIN_POSITIVE,
            1e-200,
            1e-154,
            1.,
            1e150,
            1e154,
            1e200,
            f64::MAX,
        ] {
            let data = RegressionData::new(0, vec![], vec![scale; 8]).unwrap();
            let design = Design::new(&data).unwrap();
            let mut scores = super::super::Scores::new(0);
            let mut g = [17.];
            let actual = pulp::Arch::new()
                .dispatch(NormalEval {
                    data: &data,
                    design: &design,
                    broadcast: false,
                    x: &[scale],
                    g: &mut g,
                    scores: &mut scores,
                })
                .unwrap();
            let expected = 8. * (-0.5 - scale.ln() - 0.5 * std::f64::consts::TAU.ln());
            assert!(
                (actual - expected).abs() < 2e-12 * (1. + expected.abs()),
                "scale={scale}"
            );
            assert_eq!(g, [0.]);
        }
        // Summing z^2 would overflow although sum(-z^2/2) is representable.
        let data = RegressionData::new(0, vec![], vec![6e303; 4]).unwrap();
        let design = Design::new(&data).unwrap();
        let mut scores = super::super::Scores::new(0);
        let mut g = [0.];
        let value = pulp::Arch::new()
            .dispatch(NormalEval {
                data: &data,
                design: &design,
                broadcast: false,
                x: &[1e150],
                g: &mut g,
                scores: &mut scores,
            })
            .unwrap();
        assert!(value.is_finite() && g[0].is_finite());
        assert!((value / (-7.2e307) - 1.).abs() < 1e-14);
    }

    struct Exp([f64; TILE]);
    impl pulp::WithSimd for Exp {
        type Output = [f64; TILE];
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            exp_negative(simd, &self.0)
        }
    }
    struct Predict<'a>(&'a Design, usize, &'a [f64], &'a [f64]);
    impl pulp::WithSimd for Predict<'_> {
        type Output = [f64; TILE];
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            self.0.predictors(simd, self.1, self.2, self.3)
        }
    }

    /// Isolated work controls, not additive percentages of a fused evaluation.
    /// Run only on an allocated benchmark host; never part of correctness CI.
    #[test]
    #[ignore = "release profiling on an allocated remote compute node only"]
    fn regression_stage_profile() {
        use std::{hint::black_box, time::Instant};
        fn setting(name: &str, fallback: usize) -> usize {
            std::env::var(name).map_or(fallback, |v| v.parse().expect("positive integer"))
        }
        struct Work<'a> {
            stage: &'a str,
            data: &'a RegressionData<bool>,
            design: &'a Design,
            beta: &'a [f64],
            scores: &'a mut super::super::Scores,
            calls: usize,
        }
        impl pulp::WithSimd for Work<'_> {
            type Output = f64;
            fn with_simd<S: pulp::Simd>(self, simd: S) -> f64 {
                match self.stage {
                    "predictors" => self.run::<S, 0>(simd),
                    "exp" => self.run::<S, 1>(simd),
                    "log1p" => self.run::<S, 2>(simd),
                    "scores" => self.run::<S, 3>(simd),
                    "exp_signed" => self.run::<S, 4>(simd),
                    _ => panic!("unknown regression profile stage"),
                }
            }
        }
        impl Work<'_> {
            fn run<S: pulp::Simd, const STAGE: usize>(self, simd: S) -> f64 {
                let mut checksum = 0.;
                let p = self.data.columns();
                for _ in 0..self.calls {
                    if STAGE == 3 {
                        self.scores.reset();
                    }
                    for tile in 0..self.data.rows() / TILE {
                        let input = black_box([-0.25, -1., -4., -16.]);
                        match STAGE {
                            0 => {
                                checksum += black_box(self.design.predictors(
                                    simd,
                                    tile,
                                    black_box(self.beta),
                                    black_box(&[0.; TILE]),
                                ))[0]
                            }
                            1 => checksum += black_box(exp_negative(simd, &input))[0],
                            2 => {
                                checksum +=
                                    black_box(log1p_unit(simd, &black_box([0.25, 0.5, 0.75, 1.])))
                                        [0]
                            }
                            3 => {
                                let first = tile * TILE * p;
                                self.scores.add_tile(
                                    simd,
                                    std::array::from_fn(|r| {
                                        black_box(
                                            &self.data.design()[first + r * p..first + (r + 1) * p],
                                        )
                                    }),
                                    &input,
                                );
                                black_box(&self.scores);
                            }
                            4 => {
                                checksum += black_box(exp_packet::<S, true>(
                                    simd,
                                    &black_box([-4., -0.25, 1., 16.]),
                                ))[0]
                            }
                            _ => unreachable!(),
                        }
                    }
                    if STAGE == 3 {
                        checksum += black_box(self.scores.sums[0]);
                    }
                }
                black_box(checksum)
            }
        }
        let rows = setting("ALEA_PROFILE_ROWS", 10_000);
        let p = setting("ALEA_PROFILE_COLUMNS", 8);
        let calls = setting("ALEA_PROFILE_CALLS", 1_000);
        assert!(rows > 0 && rows.is_multiple_of(TILE) && p > 0 && calls > 0);
        let stage = std::env::var("ALEA_PROFILE_STAGE").unwrap_or_else(|_| "predictors".into());
        let scalar = std::env::var("ALEA_PROFILE_SCALAR").is_ok_and(|v| v == "1");
        let data = RegressionData::new(
            p,
            (0..rows.checked_mul(p).unwrap())
                .map(|i| ((i % 31) as f64 - 15.) / 32.)
                .collect(),
            vec![false; rows],
        )
        .unwrap();
        let design = Design::new(&data).unwrap();
        let beta = vec![0.25; p];
        let mut scores = super::super::Scores::new(p);
        let start = Instant::now();
        let work = Work {
            stage: &stage,
            data: &data,
            design: &design,
            beta: &beta,
            scores: &mut scores,
            calls,
        };
        let checksum = if scalar {
            pulp::WithSimd::with_simd(work, pulp::Scalar)
        } else {
            pulp::Arch::new().dispatch(work)
        };
        let seconds = start.elapsed().as_secs_f64();
        assert!(checksum.is_finite());
        println!(
            "{{\"stage\":\"{stage}\",\"scalar\":{scalar},\"rows\":{rows},\"columns\":{p},\"calls\":{calls},\"seconds\":{seconds},\"checksum\":{checksum}}}"
        );
    }

    #[test]
    fn packed_predictors_match_scalar_product_residuals_and_all_shapes() {
        for p in 0..=17 {
            for n in 1..=17 {
                let design = (0..n * p)
                    .map(|i| match i % 5 {
                        0 => 1e16,
                        1 => 1.,
                        2 => -1e16,
                        3 => 1. + 2_f64.powi(-27),
                        _ => f64::MIN_POSITIVE,
                    })
                    .collect();
                let data = RegressionData::new(p, design, vec![0.; n])
                    .unwrap()
                    .with_offsets((0..n).map(|i| i as f64 / 13.).collect())
                    .unwrap();
                let prepared = Design::new(&data).unwrap();
                let beta: Vec<_> = (0..p)
                    .map(|i| if i % 5 == 3 { 1. - 2_f64.powi(-27) } else { 1. })
                    .collect();
                for block in 0..n.div_ceil(TILE) {
                    let offset = std::array::from_fn::<_, TILE, _>(|r| {
                        data.offsets.get(block * TILE + r).copied().unwrap_or(0.)
                    });
                    let vector =
                        pulp::Arch::new().dispatch(Predict(&prepared, block, &beta, &offset));
                    let scalar = prepared.predictors(pulp::Scalar, block, &beta, &offset);
                    for r in 0..TILE.min(n - block * TILE) {
                        let expected = data.linear_predictor(block * TILE + r, &beta);
                        assert_eq!(vector[r].to_bits(), expected.to_bits(), "p={p},n={n},r={r}");
                        assert_eq!(scalar[r].to_bits(), expected.to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn exponential_matches_independent_decimal_and_dense_scalar_oracles() {
        let fixture = include_str!("../../../tests/support/regression-exp.csv");
        for line in fixture.lines().filter(|line| !line.starts_with('#')) {
            let (x, expected) = line.split_once(',').unwrap();
            let x = f64::from_bits(u64::from_str_radix(x, 16).unwrap());
            let expected = u64::from_str_radix(expected, 16).unwrap();
            for actual in [
                pulp::Arch::new().dispatch(Exp([x; TILE])),
                exp_negative(pulp::Scalar, &[x; TILE]),
            ] {
                for a in actual {
                    assert!(
                        a.to_bits().abs_diff(expected) <= 4,
                        "x={x},actual={a},expected={}",
                        f64::from_bits(expected)
                    );
                }
            }
        }
        for i in 0..10_000 {
            let x = std::array::from_fn(|r| -32. * (i * TILE + r) as f64 / 40_000.);
            let actual = pulp::Arch::new().dispatch(Exp(x));
            for (x, a) in x.into_iter().zip(actual) {
                assert!(a.to_bits().abs_diff(x.exp().to_bits()) <= 4, "{x}");
            }
        }
        for x in [
            f64::NEG_INFINITY,
            -1000.,
            -746.,
            -745.,
            -40.,
            0.,
            0.1,
            1000.,
            f64::INFINITY,
        ] {
            for a in pulp::Arch::new().dispatch(Exp([x; TILE])) {
                assert_eq!(a.to_bits(), x.exp().to_bits(), "scalar fallback x={x}");
            }
        }
        assert!(
            pulp::Arch::new()
                .dispatch(Exp([f64::NAN; TILE]))
                .iter()
                .all(|x| x.is_nan())
        );
    }

    #[test]
    fn lane_merge_keeps_small_corrections_when_large_heads_cancel() {
        struct Sum;
        impl pulp::WithSimd for Sum {
            type Output = f64;
            fn with_simd<S: pulp::Simd>(self, simd: S) -> f64 {
                let mut sum = LaneSum::new(simd);
                for row in [[1e16, 1., -1e16, 0.], [1., 0., 0., 0.]] {
                    for v in S::as_simd_f64s(&row).0 {
                        sum.add(simd, *v);
                    }
                }
                let mut total = CompensatedSum::default();
                sum.finish(&mut total);
                total.total()
            }
        }
        assert_eq!(pulp::Arch::new().dispatch(Sum), 2.);
    }
}
