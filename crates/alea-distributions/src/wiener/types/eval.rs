use super::*;

#[derive(Copy, Clone, Debug)]
pub struct Wiener4Grad {
    pub alpha: f64,
    pub tau: f64,
    pub beta: f64,
    pub delta: f64,
}

impl Wiener4Grad {
    pub const DIM: usize = 4;
    #[must_use]
    pub fn to_array(&self) -> [f64; 4] {
        [self.alpha, self.tau, self.beta, self.delta]
    }
    #[must_use]
    pub fn from_array(arr: [f64; 4]) -> Self {
        Self {
            alpha: arr[0],
            tau: arr[1],
            beta: arr[2],
            delta: arr[3],
        }
    }
}

impl Default for Wiener4Grad {
    fn default() -> Self {
        Self {
            alpha: f64::NAN,
            tau: f64::NAN,
            beta: f64::NAN,
            delta: f64::NAN,
        }
    }
}

pub(in crate::wiener) const NAN_GRAD_4: Wiener4Grad = Wiener4Grad {
    alpha: f64::NAN,
    tau: f64::NAN,
    beta: f64::NAN,
    delta: f64::NAN,
};

impl std::ops::Index<usize> for Wiener4Grad {
    type Output = f64;
    fn index(&self, idx: usize) -> &Self::Output {
        match idx {
            0 => &self.alpha,
            1 => &self.tau,
            2 => &self.beta,
            3 => &self.delta,
            _ => panic!("Wiener4Grad index out of range: {idx}"),
        }
    }
}

impl std::ops::IndexMut<usize> for Wiener4Grad {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        match idx {
            0 => &mut self.alpha,
            1 => &mut self.tau,
            2 => &mut self.beta,
            3 => &mut self.delta,
            _ => panic!("Wiener4Grad index out of range: {idx}"),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Wiener4Eval {
    pub log_prob: f64,
    pub grad: Wiener4Grad,
}

impl Wiener4Eval {
    #[must_use]
    pub fn grad_array(&self) -> [f64; 4] {
        self.grad.to_array()
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub(in crate::wiener) struct Wiener4Core {
    pub(in crate::wiener) t: f64,
    pub(in crate::wiener) a: f64,
    pub(in crate::wiener) t_prime: f64,
    pub(in crate::wiener) w_eff: f64,
    pub(in crate::wiener) q_eff: f64,
    pub(in crate::wiener) v_eff: f64,
    pub(in crate::wiener) pref: f64,
    pub(in crate::wiener) log_eps_eff: f64,
    pub(in crate::wiener) beta_sign: f64,
    pub(in crate::wiener) delta_sign: f64,
}

#[derive(Copy, Clone, Debug)]
pub struct Wiener5Grad {
    pub alpha: f64,
    pub tau: f64,
    pub beta: f64,
    pub delta: f64,
    pub s_delta: f64,
}

impl Wiener5Grad {
    pub const DIM: usize = 5;
    #[must_use]
    pub fn to_array(&self) -> [f64; 5] {
        [self.alpha, self.tau, self.beta, self.delta, self.s_delta]
    }
    #[must_use]
    pub fn from_array(arr: [f64; 5]) -> Self {
        Self {
            alpha: arr[0],
            tau: arr[1],
            beta: arr[2],
            delta: arr[3],
            s_delta: arr[4],
        }
    }
}

impl Default for Wiener5Grad {
    fn default() -> Self {
        Self {
            alpha: f64::NAN,
            tau: f64::NAN,
            beta: f64::NAN,
            delta: f64::NAN,
            s_delta: f64::NAN,
        }
    }
}

impl std::ops::Index<usize> for Wiener5Grad {
    type Output = f64;
    fn index(&self, idx: usize) -> &Self::Output {
        match idx {
            0 => &self.alpha,
            1 => &self.tau,
            2 => &self.beta,
            3 => &self.delta,
            4 => &self.s_delta,
            _ => panic!("Wiener5Grad index out of range: {idx}"),
        }
    }
}

impl std::ops::IndexMut<usize> for Wiener5Grad {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        match idx {
            0 => &mut self.alpha,
            1 => &mut self.tau,
            2 => &mut self.beta,
            3 => &mut self.delta,
            4 => &mut self.s_delta,
            _ => panic!("Wiener5Grad index out of range: {idx}"),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Wiener5Eval {
    pub log_prob: f64,
    pub grad: Wiener5Grad,
}

impl Wiener5Eval {
    #[must_use]
    pub fn grad_array(&self) -> [f64; 5] {
        self.grad.to_array()
    }
}

pub(in crate::wiener) const NAN_GRAD_5: Wiener5Grad = Wiener5Grad {
    alpha: f64::NAN,
    tau: f64::NAN,
    beta: f64::NAN,
    delta: f64::NAN,
    s_delta: f64::NAN,
};

pub(in crate::wiener) const FAIL_EVAL_5: Wiener5Eval = Wiener5Eval {
    log_prob: f64::NEG_INFINITY,
    grad: NAN_GRAD_5,
};

#[derive(Copy, Clone, Debug, Default)]
pub(in crate::wiener) struct Wiener5Core {
    pub(in crate::wiener) base: Wiener4Core,
    pub(in crate::wiener) sv: f64,
    pub(in crate::wiener) sv2: f64,
    pub(in crate::wiener) lam: f64,
}

#[derive(Copy, Clone, Debug)]
pub struct Wiener7Grad {
    pub alpha: f64,
    pub tau: f64,
    pub beta: f64,
    pub delta: f64,
    pub s_delta: f64,
    pub s_beta: f64,
    pub s_tau: f64,
}

impl Wiener7Grad {
    pub const DIM: usize = 7;
    #[must_use]
    pub fn to_array(&self) -> [f64; 7] {
        [
            self.alpha,
            self.tau,
            self.beta,
            self.delta,
            self.s_delta,
            self.s_beta,
            self.s_tau,
        ]
    }
    #[must_use]
    pub fn from_array(arr: [f64; 7]) -> Self {
        Self {
            alpha: arr[0],
            tau: arr[1],
            beta: arr[2],
            delta: arr[3],
            s_delta: arr[4],
            s_beta: arr[5],
            s_tau: arr[6],
        }
    }
}

impl Default for Wiener7Grad {
    fn default() -> Self {
        Self {
            alpha: f64::NAN,
            tau: f64::NAN,
            beta: f64::NAN,
            delta: f64::NAN,
            s_delta: f64::NAN,
            s_beta: f64::NAN,
            s_tau: f64::NAN,
        }
    }
}

pub(in crate::wiener) const NAN_GRAD_7: Wiener7Grad = Wiener7Grad {
    alpha: f64::NAN,
    tau: f64::NAN,
    beta: f64::NAN,
    delta: f64::NAN,
    s_delta: f64::NAN,
    s_beta: f64::NAN,
    s_tau: f64::NAN,
};

impl std::ops::Index<usize> for Wiener7Grad {
    type Output = f64;
    fn index(&self, idx: usize) -> &Self::Output {
        match idx {
            0 => &self.alpha,
            1 => &self.tau,
            2 => &self.beta,
            3 => &self.delta,
            4 => &self.s_delta,
            5 => &self.s_beta,
            6 => &self.s_tau,
            _ => panic!("Wiener7Grad index out of range: {idx}"),
        }
    }
}

impl std::ops::IndexMut<usize> for Wiener7Grad {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        match idx {
            0 => &mut self.alpha,
            1 => &mut self.tau,
            2 => &mut self.beta,
            3 => &mut self.delta,
            4 => &mut self.s_delta,
            5 => &mut self.s_beta,
            6 => &mut self.s_tau,
            _ => panic!("Wiener7Grad index out of range: {idx}"),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Wiener7Eval {
    pub log_prob: f64,
    pub grad: Wiener7Grad,
}

impl Wiener7Eval {
    #[must_use]
    pub fn grad_array(&self) -> [f64; 7] {
        self.grad.to_array()
    }
}

#[derive(Clone, Debug)]
pub(in crate::wiener) struct Wiener7Core {
    pub(in crate::wiener) alpha: f64,
    pub(in crate::wiener) tau0: f64,
    pub(in crate::wiener) beta0: f64,
    pub(in crate::wiener) delta: f64,
    pub(in crate::wiener) sv: f64,
    pub(in crate::wiener) sw: f64,
    pub(in crate::wiener) st0: f64,
    pub(in crate::wiener) dim: usize,
    pub(in crate::wiener) xmin: [f64; 2],
    pub(in crate::wiener) xmax: [f64; 2],
    pub(in crate::wiener) eps_series: f64,
    pub(in crate::wiener) opts: Options,
}
