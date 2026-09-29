use rand::Rng;

pub trait Proposal {
    fn dim(&self) -> usize;

    fn propose_into<R: Rng + ?Sized>(
        &mut self,
        current: &[f64],
        proposal: &mut [f64],
        step_size: f64,
        rng: &mut R,
    );
}

pub trait LogProposalRatio {
    fn log_proposal_ratio(&self, _from: &[f64], _to: &[f64]) -> f64 {
        0.0
    }
}

impl<T> LogProposalRatio for T where T: Proposal {}
