pub trait StepSizeAdapter {
    fn reset(&mut self, initial_step_size: f64);
    fn update(&mut self, accept_prob: f64) -> f64;
    fn finalize(&self) -> f64;
}

pub trait AdaptationSchedule {
    fn in_warmup(&self, iter: usize) -> bool;
}
