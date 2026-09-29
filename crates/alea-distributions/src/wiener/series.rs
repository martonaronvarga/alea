use core::f64::consts::FRAC_1_PI;
use std::f64::consts::{PI, TAU};

use super::LN_PI;

#[inline]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn ceil_to_usize(value: f64, fallback: usize) -> usize {
    if value.is_finite() && value <= usize::MAX as f64 {
        value as usize
    } else {
        fallback
    }
}

#[inline]
#[allow(clippy::cast_precision_loss)]
fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

/// Large-time truncation count from the Navarro/Gondan style bound
/// This is the count for the π-series
///
/// t_prime = (y - tau) / alpha^2
#[inline]
pub(super) fn k_l(t_prime: f64, log_eps: f64) -> usize {
    if super::unlikely(!t_prime.is_finite() || !log_eps.is_finite() || t_prime <= 0.0) {
        return 1;
    }

    let sqrt_t = t_prime.sqrt();
    let inv_sqrt_t = sqrt_t.recip();

    let k1 = {
        let log_x = LN_PI + t_prime.ln() + log_eps; // ln(pi * t' * eps)
        if log_x < 0.0 {
            (-2.0 * log_x / (PI * PI * t_prime)).sqrt()
        } else {
            0.0
        }
    };

    let k2 = FRAC_1_PI * inv_sqrt_t;

    let k = k1.max(k2).ceil().max(1.0);

    ceil_to_usize(k, usize::MAX)
}

/// Small-time truncation count.
#[inline]
pub(super) fn k_s(t_prime: f64, w: f64, log_eps: f64) -> usize {
    const LN_TAU: f64 = 1.8378770664093453_f64; // ln(2π)

    if super::unlikely(
        !t_prime.is_finite() || !w.is_finite() || !log_eps.is_finite() || t_prime <= 0.0,
    ) {
        return 0;
    }

    let sqrt_2t = (2.0 * t_prime).sqrt();
    let u_eps = (LN_TAU + 2.0 * (t_prime.ln() + log_eps)).min(-1.0);
    let term1 = 0.5 * (sqrt_2t + (1.0 - w));

    let s = (-2.0 * u_eps - 2.0).sqrt();
    let arg = t_prime * (s - u_eps);
    let term2 = 0.5 * (arg.sqrt() + (1.0 - w));

    let k = term1.max(term2).ceil().max(0.0);

    ceil_to_usize(k, usize::MAX)
}

#[inline]
#[allow(dead_code)]
pub(super) fn k_s_grad_w(t_prime: f64, w: f64, log_eps: f64) -> usize {
    const LN_TAU: f64 = 1.8378770664093453;

    if super::unlikely(
        !t_prime.is_finite() || !w.is_finite() || !log_eps.is_finite() || t_prime <= 0.0,
    ) {
        return 0;
    }
    let sqrt_2t = (2.0 * t_prime).sqrt();
    let one_m_w = 1.0 - w;
    let n1 = 0.5 * (sqrt_2t + one_m_w);

    let u_eps = (LN_TAU + 2.0 * (t_prime.ln() + log_eps)).min(-1.0);
    let s = (-2.0 * u_eps - 2.0).sqrt();
    let arg = t_prime * (s - u_eps);
    let n2 = 0.5 * (arg.sqrt() + one_m_w);

    let k = n1.max(n2).ceil().max(0.0);
    ceil_to_usize(k, usize::MAX)
}

#[inline]
pub(super) fn k_l_grad_w(t_prime: f64, log_eps: f64) -> usize {
    const LN_4_OVER_9: f64 = 0.8109302162163288;
    const TWO_LN_PI: f64 = 2.0 * LN_PI; // 2*ln(π)

    if super::unlikely(!t_prime.is_finite() || !log_eps.is_finite() || t_prime <= 0.0) {
        return 1;
    }
    let inv_t = t_prime.recip();
    let n1 = (2.0 * inv_t).sqrt() * FRAC_1_PI;

    let u_eps_arg = LN_4_OVER_9 + TWO_LN_PI + 3.0 * t_prime.ln() + 2.0 * log_eps;
    let u_eps = u_eps_arg.min(-1.0);
    let arg = -(u_eps - (-2.0 * u_eps - 2.0).sqrt());
    let n2 = if arg > 0.0 {
        (arg * inv_t).sqrt() * FRAC_1_PI
    } else {
        0.0
    };
    let k = n1.max(n2).ceil().max(1.0);
    ceil_to_usize(k, usize::MAX)
}

#[inline]
#[allow(dead_code)]
pub(super) fn small_time_log_series(t_prime: f64, w: f64, k: usize) -> Option<f64> {
    const LN_TAU: f64 = 1.8378770664093453_f64; // ln(2π)

    if super::unlikely(
        !t_prime.is_finite() || !w.is_finite() || t_prime <= 0.0 || !(0.0..1.0).contains(&w),
    ) {
        return None;
    }

    let one_m_w = 1.0 - w;
    let inv_two_t = 0.5 / t_prime;
    let scale = one_m_w * one_m_w * inv_two_t;
    let log_pref = -0.5 * LN_TAU - 1.5 * t_prime.ln();

    let mut pos = one_m_w;
    let mut neg = 0.0;
    let mut c_pos = 0.0;
    let mut c_neg = 0.0;

    for j in 1..=k {
        let two_j = 2.0 * (usize_to_f64(j));
        let xp = two_j + one_m_w;
        let xm = two_j - one_m_w;

        let ep = (scale - xp * xp * inv_two_t).exp();
        let en = (scale - xm * xm * inv_two_t).exp();

        // Kahan-compensated accumulation for the positive half
        let y = xp * ep - c_pos;
        let t = pos + y;
        c_pos = (t - pos) - y;
        pos = t;

        // Kahan-compensated accumulation for the negative half
        let y = xm * en - c_neg;
        let t = neg + y;
        c_neg = (t - neg) - y;
        neg = t;
    }

    if super::unlikely(!pos.is_finite() || !neg.is_finite()) {
        return None;
    }

    if pos <= neg {
        return Some(f64::NEG_INFINITY);
    }

    let ratio = neg / pos;
    Some(log_pref - scale + pos.ln() + (-ratio).ln_1p())
}

#[inline]
#[allow(dead_code)]
pub(super) fn small_time_series_raw(t_prime: f64, w: f64, k: usize) -> Option<f64> {
    if super::unlikely(
        !t_prime.is_finite() || !w.is_finite() || t_prime <= 0.0 || !(0.0..1.0).contains(&w),
    ) {
        return None;
    }

    let a = 1.0 - w; // a in (0, 1)
    if k == 0 {
        return Some(a);
    }

    let inv_t = t_prime.recip();
    // Kahan compensation for the alternating accumulation
    let mut sum = a;
    let mut c = 0.0;

    for j in 1..=k {
        let jf = usize_to_f64(j);
        let two_j = 2.0 * jf;
        let xp = two_j + a;
        let xm = two_j - a;

        // Exact algebra:
        // xp^2 - a^2 = 4j(j + a)
        // xm^2 - a^2 = 4j(j - a)
        let arg_p = 2.0 * jf * (jf + a) * inv_t;
        let arg_m = 2.0 * jf * (jf - a) * inv_t;

        let term = xp * (-arg_p).exp() - xm * (-arg_m).exp();

        let y = term - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    if sum.is_finite() { Some(sum) } else { None }
}

#[inline]
#[allow(dead_code)]
pub(super) fn small_time_dr_dt(t_prime: f64, w: f64, k: usize) -> Option<f64> {
    if super::unlikely(
        !t_prime.is_finite() || !w.is_finite() || t_prime <= 0.0 || !(0.0..1.0).contains(&w),
    ) {
        return None;
    }

    if k == 0 {
        return Some(0.0);
    }

    let a = 1.0 - w;
    let inv_t = t_prime.recip();
    // Kahan compensation helps when the alternating contributions nearly cancel.
    let mut sum = 0.0;
    let mut c = 0.0;

    for j in 1..=k {
        let jf = usize_to_f64(j);
        let two_j = 2.0 * jf;
        let xp = two_j + a;
        let xm = two_j - a;

        // Exact algebra:
        // xp^2 - a^2 = 4j(j + a)
        // xm^2 - a^2 = 4j(j - a)
        let arg_p = 2.0 * jf * (jf + a) * inv_t;
        let arg_m = 2.0 * jf * (jf - a) * inv_t;

        // d/dt' of exp(-arg) = exp(-arg) * (arg / t')
        let term = xp * (-arg_p).exp() * (arg_p * inv_t) - xm * (-arg_m).exp() * (arg_m * inv_t);

        let y = term - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    if sum.is_finite() { Some(sum) } else { None }
}

/// d/dw of the scaled raw small-time sum R_s(t', w).
#[inline]
#[allow(dead_code)]
pub(super) fn small_time_dr_dw(t_prime: f64, w: f64, k: usize) -> Option<f64> {
    if super::unlikely(
        !t_prime.is_finite() || !w.is_finite() || t_prime <= 0.0 || !(0.0..1.0).contains(&w),
    ) {
        return None;
    }

    let a = 1.0 - w;
    let inv_t = 1.0 / t_prime;
    // j = 0 contribution: d/dw of (1 - w) = -1.
    let mut sum = -1.0;
    let mut c = 0.0; // Kahan

    for j in 1..=k {
        let jf = usize_to_f64(j);
        let two_j = 2.0 * jf;
        let xp = two_j + a;
        let xm = two_j - a;

        // Exact simplifications:
        // xp^2 - a^2 = 4j(j + a)
        // xm^2 - a^2 = 4j(j - a)
        let arg_p = 2.0 * jf * (jf + a) * inv_t;
        let arg_m = 2.0 * jf * (jf - a) * inv_t;

        let ep = (-arg_p).exp();
        let em = (-arg_m).exp();

        let dep_da = ep * (1.0 - xp * 2.0 * jf * inv_t);
        let dem_da = em * (1.0 - xm * 2.0 * jf * inv_t);

        let dsum_da = dep_da + dem_da;
        let dsum_dw = -dsum_da; // da/dw = -1

        let y = dsum_dw - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    if sum.is_finite() { Some(sum) } else { None }
}

#[inline]
pub(super) fn small_time_scaled_accum(t_prime: f64, w: f64, k: usize) -> Option<(f64, f64, f64)> {
    if super::unlikely(
        !t_prime.is_finite() || !w.is_finite() || t_prime <= 0.0 || !(0.0..1.0).contains(&w),
    ) {
        return None;
    }

    match k {
        2 => small_time_scaled_accum_fixed::<2>(t_prime, w),
        3 => small_time_scaled_accum_fixed::<3>(t_prime, w),
        _ => small_time_scaled_accum_loop(t_prime, w, k),
    }
}

#[inline]
fn small_time_scaled_accum_loop(t_prime: f64, w: f64, k: usize) -> Option<(f64, f64, f64)> {
    let a = 1.0 - w;
    let inv_t = t_prime.recip();

    let mut raw = a;
    let mut d_t = 0.0;
    let mut d_w = -1.0;
    let mut c_raw = 0.0;
    let mut c_dt = 0.0;
    let mut c_dw = 0.0;

    for j in 1..=k {
        add_small_scaled_term(
            j, a, inv_t, &mut raw, &mut d_t, &mut d_w, &mut c_raw, &mut c_dt, &mut c_dw,
        );
    }

    if raw.is_finite() && d_t.is_finite() && d_w.is_finite() {
        Some((raw, d_t, d_w))
    } else {
        None
    }
}

#[inline]
fn small_time_scaled_accum_fixed<const K: usize>(t_prime: f64, w: f64) -> Option<(f64, f64, f64)> {
    let a = 1.0 - w;
    let inv_t = t_prime.recip();

    let mut raw = a;
    let mut d_t = 0.0;
    let mut d_w = -1.0;
    let mut c_raw = 0.0;
    let mut c_dt = 0.0;
    let mut c_dw = 0.0;

    let mut j = 1;
    while j <= K {
        add_small_scaled_term(
            j, a, inv_t, &mut raw, &mut d_t, &mut d_w, &mut c_raw, &mut c_dt, &mut c_dw,
        );
        j += 1;
    }

    if raw.is_finite() && d_t.is_finite() && d_w.is_finite() {
        Some((raw, d_t, d_w))
    } else {
        None
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn add_small_scaled_term(
    j: usize,
    a: f64,
    inv_t: f64,
    raw: &mut f64,
    d_t: &mut f64,
    d_w: &mut f64,
    c_raw: &mut f64,
    c_dt: &mut f64,
    c_dw: &mut f64,
) {
    let jf = usize_to_f64(j);
    let two_j = 2.0 * jf;
    let xp = two_j + a;
    let xm = two_j - a;
    let arg_p = 2.0 * jf * (jf + a) * inv_t;
    let arg_m = 2.0 * jf * (jf - a) * inv_t;
    let ep = (-arg_p).exp();
    let em = (-arg_m).exp();

    let term_raw = xp * ep - xm * em;
    let y = term_raw - *c_raw;
    let t = *raw + y;
    *c_raw = (t - *raw) - y;
    *raw = t;

    let term_dt = xp * ep * (arg_p * inv_t) - xm * em * (arg_m * inv_t);
    let y = term_dt - *c_dt;
    let t = *d_t + y;
    *c_dt = (t - *d_t) - y;
    *d_t = t;

    let dep_da = ep * (1.0 - xp * two_j * inv_t);
    let dem_da = em * (1.0 - xm * two_j * inv_t);
    let term_dw = -(dep_da + dem_da);
    let y = term_dw - *c_dw;
    let t = *d_w + y;
    *c_dw = (t - *d_w) - y;
    *d_w = t;
}

#[inline]
#[allow(dead_code)]
pub(super) fn large_time_log_series(t_prime: f64, w: f64, k: usize) -> Option<f64> {
    const LN_PI: f64 = 1.144729885849400174143427351353058711_f64;
    if super::unlikely(
        !t_prime.is_finite() || !w.is_finite() || t_prime <= 0.0 || !(0.0..1.0).contains(&w),
    ) {
        return None;
    }
    if k == 0 {
        return None; // log(0) is not finite
    }

    let base_exp = 0.5 * PI * PI * t_prime;
    let log_pref = LN_PI - base_exp;

    // Recurrence for sin(j*pi*q), cos(j*pi*q), q = 1 - w.
    let theta = PI * (1.0 - w);
    let (mut sin_j, mut cos_j) = theta.sin_cos();
    let (sin_theta, cos_theta) = (sin_j, cos_j);

    // Compensated split accumulation: terms with positive and negative sign separately
    let mut pos = 0.0;
    let mut neg = 0.0;
    let mut c_pos = 0.0;
    let mut c_neg = 0.0;

    for j in 1..=k {
        let jf = usize_to_f64(j);

        // Exact exponent factor relative to j = 1:
        // exp(-(j^2 - 1) * π^2 t' / 2)
        let delta = (jf * jf - 1.0) * base_exp;
        let weight = jf * (-delta).exp();
        let signed = weight * sin_j;

        if signed >= 0.0 {
            let y = signed - c_pos;
            let t = pos + y;
            c_pos = (t - pos) - y;
            pos = t;
        } else {
            let y = -signed - c_neg;
            let t = neg + y;
            c_neg = (t - neg) - y;
            neg = t;
        }

        if j != k {
            let next_sin = sin_j.mul_add(cos_theta, cos_j * sin_theta);
            let next_cos = cos_j.mul_add(cos_theta, -sin_j * sin_theta);
            sin_j = next_sin;
            cos_j = next_cos;
        }
    }

    if super::unlikely(!pos.is_finite() || !neg.is_finite() || pos <= neg || pos <= 0.0) {
        return None;
    }

    let ratio = neg / pos;
    if !(0.0..1.0).contains(&ratio) {
        return None;
    }

    Some(log_pref + pos.ln() + (-ratio).ln_1p())
}

#[inline]
pub(super) fn large_time_scaled_accum(t_prime: f64, w: f64, k: usize) -> Option<(f64, f64, f64)> {
    if super::unlikely(
        !t_prime.is_finite() || !w.is_finite() || t_prime <= 0.0 || !(0.0..1.0).contains(&w),
    ) {
        return None;
    }

    if k == 0 {
        return Some((0.0, 0.0, 0.0));
    }

    match k {
        4 => large_time_scaled_accum_fixed::<4>(t_prime, w),
        5 => large_time_scaled_accum_fixed::<5>(t_prime, w),
        6 => large_time_scaled_accum_fixed::<6>(t_prime, w),
        _ => large_time_scaled_accum_loop(t_prime, w, k),
    }
}

#[inline]
fn large_time_scaled_accum_loop(t_prime: f64, w: f64, k: usize) -> Option<(f64, f64, f64)> {
    let theta = PI * (1.0 - w);
    let (mut s_j, mut c_j) = theta.sin_cos();
    let (sin_theta, cos_theta) = (s_j, c_j);

    let half_pi2 = 0.5 * PI * PI;
    let half_pi2_t = half_pi2 * t_prime;

    let mut sum = 0.0;
    let mut d_t = 0.0;
    let mut d_w = 0.0;
    let mut c_sum = 0.0;
    let mut c_dt = 0.0;
    let mut c_dw = 0.0;

    for j in 1..=k {
        let jf = usize_to_f64(j);
        let jj = jf * jf;

        let delta = (jj - 1.0) * half_pi2_t;
        let e = (-delta).exp();

        let term = jf * s_j * e;
        let term_dt = -half_pi2 * jf * (jj - 1.0) * s_j * e;
        let term_dw = -jj * PI * c_j * e;

        let y = term - c_sum;
        let t = sum + y;
        c_sum = (t - sum) - y;
        sum = t;

        let y = term_dt - c_dt;
        let t = d_t + y;
        c_dt = (t - d_t) - y;
        d_t = t;

        let y = term_dw - c_dw;
        let t = d_w + y;
        c_dw = (t - d_w) - y;
        d_w = t;

        if j != k {
            let next_s = s_j.mul_add(cos_theta, c_j * sin_theta);
            let next_c = c_j.mul_add(cos_theta, -s_j * sin_theta);
            s_j = next_s;
            c_j = next_c;
        }
    }

    if sum.is_finite() && d_t.is_finite() && d_w.is_finite() {
        Some((sum, d_t, d_w))
    } else {
        None
    }
}

#[inline]
fn large_time_scaled_accum_fixed<const K: usize>(t_prime: f64, w: f64) -> Option<(f64, f64, f64)> {
    let theta = PI * (1.0 - w);
    let (mut s_j, mut c_j) = theta.sin_cos();
    let (sin_theta, cos_theta) = (s_j, c_j);

    let half_pi2 = 0.5 * PI * PI;
    let half_pi2_t = half_pi2 * t_prime;

    let mut sum = 0.0;
    let mut d_t = 0.0;
    let mut d_w = 0.0;
    let mut c_sum = 0.0;
    let mut c_dt = 0.0;
    let mut c_dw = 0.0;

    let mut j = 1;
    while j <= K {
        let jf = usize_to_f64(j);
        let jj = jf * jf;
        let e = (-((jj - 1.0) * half_pi2_t)).exp();

        let term = jf * s_j * e;
        let term_dt = -half_pi2 * jf * (jj - 1.0) * s_j * e;
        let term_dw = -jj * PI * c_j * e;

        let y = term - c_sum;
        let t = sum + y;
        c_sum = (t - sum) - y;
        sum = t;

        let y = term_dt - c_dt;
        let t = d_t + y;
        c_dt = (t - d_t) - y;
        d_t = t;

        let y = term_dw - c_dw;
        let t = d_w + y;
        c_dw = (t - d_w) - y;
        d_w = t;

        let next_s = s_j.mul_add(cos_theta, c_j * sin_theta);
        let next_c = c_j.mul_add(cos_theta, -s_j * sin_theta);
        s_j = next_s;
        c_j = next_c;
        j += 1;
    }

    if sum.is_finite() && d_t.is_finite() && d_w.is_finite() {
        Some((sum, d_t, d_w))
    } else {
        None
    }
}

#[inline]
pub(super) fn small_branch_fused(t_prime: f64, w: f64, k: usize) -> Option<(f64, f64, f64)> {
    let (raw, d_r_dt, d_r_dw) = small_time_scaled_accum(t_prime, w, k)?;
    if raw <= 0.0 {
        return None;
    }

    let a = 1.0 - w;
    let inv_t = t_prime.recip();
    let scale = 0.5 * a * a * inv_t;

    let log_series = -0.5 * TAU.ln() - 1.5 * t_prime.ln() - scale + raw.ln();
    let dlog_dtprime = -1.5 * inv_t + scale * inv_t + d_r_dt / raw;
    let dlog_dw = a * inv_t + d_r_dw / raw;

    Some((log_series, dlog_dtprime, dlog_dw))
}

#[inline]
pub(super) fn large_branch_fused(t_prime: f64, w: f64, k: usize) -> Option<(f64, f64, f64)> {
    let (raw, d_r_dt, d_r_dw) = large_time_scaled_accum(t_prime, w, k)?;
    if raw <= 0.0 {
        return None;
    }

    let half_pi2 = 0.5 * PI * PI;
    let log_series = PI.ln() - half_pi2 * t_prime + raw.ln();
    let dlog_dtprime = -half_pi2 + d_r_dt / raw;
    let dlog_dw = d_r_dw / raw;

    Some((log_series, dlog_dtprime, dlog_dw))
}
