use kernels::{
    buffer::OwnedBuffer,
    density::LogDensity,
    kernel::Kernel,
    metric::{CholeskyFactor, DenseMetric, IdentityMetric, Metric},
    numeric::{finite_or_neg_inf, log_accept_ratio, positive_finite},
    proposal::{LogProposalRatio, Proposal},
    state::LogProbState,
};
use rand::{Rng, RngExt};
use tracing::{debug, span, Level};

const DELTA_TARGET: f64 = 0.234;
const GAMMA: f64 = 0.05;
const T0: f64 = 10.0;
const KAPPA: f64 = 0.75;

#[derive(Debug, Clone)]
pub struct RwmhConfig {
    pub step_size: f64,
    pub n_warmup: usize,
    pub n_draws: usize,
    pub adapt_step_size: bool,
    pub target_accept_rate: f64,
}

impl Default for RwmhConfig {
    fn default() -> Self {
        Self {
            step_size: 1.0,
            n_warmup: 1_000,
            n_draws: 1_000,
            adapt_step_size: true,
            target_accept_rate: DELTA_TARGET,
        }
    }
}

impl RwmhConfig {
    pub fn with_step_size(mut self, s: f64) -> Self {
        self.step_size = s;
        self
    }
    pub fn with_warmup(mut self, n: usize) -> Self {
        self.n_warmup = n;
        self
    }

    pub fn with_draws(mut self, n: usize) -> Self {
        self.n_draws = n;
        self
    }

    pub fn with_adapt_step_size(mut self, adapt: bool) -> Self {
        self.adapt_step_size = adapt;
        self
    }

    pub fn with_target_accept_rate(mut self, delta: f64) -> Self {
        self.target_accept_rate = delta;
        self
    }
}

#[derive(Debug)]
pub struct Draws {
    data: OwnedBuffer,
    n_draws: usize,
    dim: usize,
}

impl Draws {
    fn new(n_draws: usize, dim: usize) -> Self {
        let len = n_draws.checked_mul(dim).expect("draw buffer overflow");
        Self {
            data: OwnedBuffer::new(len),
            n_draws,
            dim,
        }
    }

    pub fn n_draws(&self) -> usize {
        self.n_draws
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn row(&self, i: usize) -> &[f64] {
        let start = i * self.dim;
        &self.data[start..start + self.dim]
    }

    fn row_mut(&mut self, i: usize) -> &mut [f64] {
        let start = i * self.dim;
        &mut self.data[start..start + self.dim]
    }
}

#[derive(Debug)]
struct DualAvg {
    mu: f64,
    h_bar: f64,
    log_eps_bar: f64,
    m: usize,
    delta: f64,
}

impl DualAvg {
    fn new(initial_step_size: f64, target_accept_rate: f64) -> Self {
        assert!(initial_step_size.is_finite() && initial_step_size > 0.0);
        assert!(
            target_accept_rate.is_finite() && target_accept_rate > 0.0 && target_accept_rate < 1.0
        );

        Self {
            mu: (10.0 * initial_step_size).ln(),
            h_bar: 0.0,
            log_eps_bar: initial_step_size.ln(),
            m: 0,
            delta: target_accept_rate,
        }
    }

    fn update(&mut self, accept_prob: f64) -> f64 {
        self.m += 1;
        let m = self.m as f64;

        let w = 1.0 / (m + T0);
        self.h_bar = (1.0 - w) * self.h_bar + w * (self.delta - accept_prob);

        let log_eps = self.mu - (m.sqrt() / GAMMA) * self.h_bar;
        let decay = m.powf(-KAPPA);
        self.log_eps_bar = decay * log_eps + (1.0 - decay) * self.log_eps_bar;
        positive_finite(log_eps.exp())
    }

    fn final_step_size(&self) -> f64 {
        positive_finite(self.log_eps_bar.exp())
    }
}

pub struct Rwmh<M, S> {
    pub config: RwmhConfig,
    metric: M,
    spare_normal: Option<f64>,
    dim: usize,
    noise: OwnedBuffer,
    proposal: OwnedBuffer,
    accepted_warmup: usize,
    total_warmup: usize,
    accepted_main: usize,
    total_main: usize,
    dual_avg: Option<DualAvg>,
    _marker: std::marker::PhantomData<S>,
}

impl<M, S> Rwmh<M, S>
where
    M: Metric,
    S: LogProbState,
{
    #[inline]
    pub fn new(config: RwmhConfig, metric: M) -> Self {
        assert!(config.step_size.is_finite() && config.step_size > 0.0);

        let dim = metric.dim();
        let dual_avg = config
            .adapt_step_size
            .then(|| DualAvg::new(config.step_size, config.target_accept_rate));

        Self {
            config,
            metric,
            spare_normal: None,
            dim,
            noise: OwnedBuffer::new(dim),
            proposal: OwnedBuffer::new(dim),
            accepted_warmup: 0,
            total_warmup: 0,
            accepted_main: 0,
            total_main: 0,
            dual_avg,
            _marker: std::marker::PhantomData,
        }
    }
    fn draw_standard_normal<R: Rng + ?Sized>(&mut self, rng: &mut R) -> f64 {
        if let Some(z) = self.spare_normal.take() {
            return z;
        }

        loop {
            let u = 2.0 * rng.random::<f64>() - 1.0;
            let v = 2.0 * rng.random::<f64>() - 1.0;
            let s = u * u + v * v;
            if (0.0..1.0).contains(&s) {
                let scale = (-2.0 * s.ln() / s).sqrt();
                self.spare_normal = Some(v * scale);
                return u * scale;
            }
        }
    }

    pub fn warmup_acceptance_rate(&self) -> f64 {
        if self.total_warmup == 0 {
            0.0
        } else {
            self.accepted_warmup as f64 / self.total_warmup as f64
        }
    }
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_main == 0 {
            0.0
        } else {
            self.accepted_main as f64 / self.total_main as f64
        }
    }

    pub fn current_step_size(&self) -> f64 {
        self.config.step_size
    }

    fn finish_warmup(&mut self) {
        if let Some(step_size) = self.dual_avg.as_ref().map(|da| da.final_step_size()) {
            self.config.step_size = step_size;
        }
    }

    fn propose<R: Rng + ?Sized>(&mut self, state: &S, rng: &mut R) {
        for i in 0..self.noise.len() {
            let z = self.draw_standard_normal(rng);
            self.noise.as_mut_slice()[i] = z;
        }
        self.metric
            .apply_sqrt(self.noise.as_slice(), self.proposal.as_mut_slice());

        let x = state.position();
        let y = self.proposal.as_mut_slice();
        for i in 0..self.dim {
            y[i] = x[i] + self.config.step_size * y[i];
        }
    }

    fn step_impl<D, R>(&mut self, state: &mut S, target: &D, rng: &mut R) -> (bool, f64)
    where
        D: LogDensity,
        R: Rng + ?Sized,
    {
        debug_assert_eq!(state.dim(), self.dim);

        if !state.log_prob().is_finite() {
            state.initialize_log_prob(target);
        }

        let current_lp = finite_or_neg_inf(state.log_prob());
        self.propose(state, rng);

        let proposal_lp = finite_or_neg_inf(target.log_prob(self.proposal.as_slice()));
        let log_alpha = log_accept_ratio(current_lp, proposal_lp).min(0.0);
        let accept_prob = log_alpha.exp();

        let accepted = rng.random::<f64>() < accept_prob;

        if accepted {
            state
                .position_mut()
                .copy_from_slice(self.proposal.as_slice());
            state.set_log_prob(proposal_lp);
        }

        (accepted, accept_prob)
    }

    pub fn sample<D, R>(&mut self, target: &D, state: &mut S, rng: &mut R) -> Draws
    where
        D: LogDensity,
        R: Rng + ?Sized,
    {
        debug_assert_eq!(state.dim(), self.dim);

        if !state.log_prob().is_finite() {
            state.initialize_log_prob(target);
        }

        self.total_warmup = 0;
        self.total_main = 0;
        self.accepted_warmup = 0;
        self.accepted_main = 0;
        self.spare_normal = None;

        if self.config.adapt_step_size {
            self.dual_avg = Some(DualAvg::new(
                self.config.step_size,
                self.config.target_accept_rate,
            ));
        } else {
            self.dual_avg = None;
        }

        let warmup_span = span!(Level::DEBUG, "rwmh_warmup", n_warmup = self.config.n_warmup);
        let _enter = warmup_span.enter();

        for i in 0..self.config.n_warmup {
            let (accepted, accept_prob) = self.step_impl(state, target, rng);

            self.total_warmup += 1;
            self.accepted_warmup += accepted as usize;

            if let Some(step_size) = self.dual_avg.as_mut().map(|da| da.update(accept_prob)) {
                self.config.step_size = step_size;
            }

            if i > 0 && i % 100 == 0 {
                debug!(
                    step = i,
                    accept_rate = self.warmup_acceptance_rate(),
                    step_size = self.config.step_size,
                    "Warmup progress"
                );
            }
        }

        if self.config.n_warmup > 0 {
            self.finish_warmup();
        }

        let mut draws = Draws::new(self.config.n_draws, self.dim);

        for i in 0..self.config.n_draws {
            let (accepted, _) = self.step_impl(state, target, rng);

            self.total_main += 1;
            self.accepted_main += accepted as usize;

            draws.row_mut(i).copy_from_slice(state.position());
        }

        draws
    }
}

impl<S> Rwmh<IdentityMetric, S>
where
    S: LogProbState,
{
    pub fn isotropic(config: RwmhConfig, dim: usize) -> Self {
        Self::new(config, IdentityMetric::new(dim))
    }
}

impl<S> Rwmh<DenseMetric, S>
where
    S: LogProbState,
{
    pub fn dense_cholesky(config: RwmhConfig, factor: CholeskyFactor) -> Self {
        Self::new(config, DenseMetric::new(factor))
    }
}

impl<M, S, D> Kernel<D> for Rwmh<M, S>
where
    M: Metric,
    S: LogProbState,
    D: LogDensity,
{
    type State = S;

    fn initialize(&mut self, state: &mut Self::State, target: &D) {
        if !state.log_prob().is_finite() {
            state.initialize_log_prob(target);
        }
    }

    fn step<R: Rng + ?Sized>(&mut self, state: &mut Self::State, target: &D, rng: &mut R) -> bool {
        let (accepted, _) = self.step_impl(state, target, rng);
        accepted
    }
}
