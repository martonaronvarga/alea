#[inline(always)]
pub fn positive_finite(x: f64) -> f64 {
    x.max(f64::MIN_POSITIVE)
}

#[inline(always)]
pub fn finite_or_neg_inf(x: f64) -> f64 {
    if x.is_finite() { x } else { f64::NEG_INFINITY }
}

#[inline(always)]
pub fn log_accept_ratio(current: f64, proposal: f64) -> f64 {
    if proposal.is_finite() {
        proposal - current
    } else {
        f64::NEG_INFINITY
    }
}

#[inline(always)]
pub fn sigmoid(theta: f64) -> f64 {
    if theta >= 0.0 {
        let e = (-theta).exp();
        1.0 / (1.0 + e)
    } else {
        let e = theta.exp();
        e / (1.0 + e)
    }
}

#[inline(always)]
pub fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

#[inline(always)]
pub fn softplus(x: f64) -> f64 {
    if x > 30.0 {
        x
    } else if x < -30.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[inline(always)]
pub fn log_sigmoid(x: f64) -> f64 {
    -softplus(-x)
}

#[inline(always)]
pub fn log1m_sigmoid(x: f64) -> f64 {
    -softplus(x)
}
