use crate::{
    buffer::OwnedBuffer,
    density::{GradLogDensity, LogDensity},
};

#[derive(Debug)]
pub struct ChainState<P = OwnedBuffer, A = ()> {
    pub position: P,
    pub log_prob: f64,
    pub aux: A,
}

#[derive(Debug)]
pub struct GradientBuffers<B = OwnedBuffer> {
    pub gradient: B,
    pub momentum: B,
}

pub trait GradientStorage {
    fn gradient(&self) -> &[f64];
    fn gradient_mut(&mut self) -> &mut [f64];
    fn momentum(&self) -> &[f64];
    fn momentum_mut(&mut self) -> &mut [f64];
}

pub trait LogProbState {
    fn dim(&self) -> usize;
    fn position(&self) -> &[f64];
    fn position_mut(&mut self) -> &mut [f64];
    fn log_prob(&self) -> f64;
    fn set_log_prob(&mut self, value: f64);

    #[inline]
    fn initialize_log_prob<D: LogDensity<Point = [f64]>>(&mut self, density: &D) {
        self.set_log_prob(density.log_prob(self.position()));
    }
}

pub trait GradientState: LogProbState {
    fn gradient(&self) -> &[f64];
    fn gradient_mut(&mut self) -> &mut [f64];
    fn momentum(&self) -> &[f64];
    fn momentum_mut(&mut self) -> &mut [f64];
    fn with_position_and_gradient_mut<T, F>(&mut self, f: F) -> T
    where
        F: FnOnce(&[f64], &mut [f64]) -> T;

    #[inline]
    fn initialize_gradient<D: GradLogDensity<Point = [f64], Gradient = [f64]>>(
        &mut self,
        density: &D,
    ) {
        self.with_position_and_gradient_mut(|position, gradient| {
            debug_assert_eq!(position.len(), gradient.len());
            density.grad_log_prob(position, gradient);
        });
    }
}

impl ChainState<OwnedBuffer, ()> {
    pub fn new(dim: usize) -> Self {
        let mut position = OwnedBuffer::new(dim);
        position.fill(0.0);

        Self {
            position,
            log_prob: f64::NAN,
            aux: (),
        }
    }
}

impl<P, A> ChainState<P, A> {
    pub fn with_aux(position: P, aux: A) -> Self {
        Self {
            position,
            log_prob: f64::NAN,
            aux,
        }
    }
}

impl<P, A> LogProbState for ChainState<P, A>
where
    P: AsRef<[f64]> + AsMut<[f64]>,
{
    #[inline]
    fn dim(&self) -> usize {
        self.position.as_ref().len()
    }

    #[inline]
    fn position(&self) -> &[f64] {
        self.position.as_ref()
    }

    #[inline]
    fn position_mut(&mut self) -> &mut [f64] {
        self.position.as_mut()
    }

    #[inline]
    fn log_prob(&self) -> f64 {
        self.log_prob
    }

    #[inline]
    fn set_log_prob(&mut self, value: f64) {
        self.log_prob = value;
    }
}

impl<P, A> GradientState for ChainState<P, A>
where
    P: AsRef<[f64]> + AsMut<[f64]>,
    A: GradientStorage,
{
    #[inline]
    fn gradient(&self) -> &[f64] {
        self.aux.gradient()
    }

    #[inline]
    fn gradient_mut(&mut self) -> &mut [f64] {
        self.aux.gradient_mut()
    }

    #[inline]
    fn momentum(&self) -> &[f64] {
        self.aux.momentum()
    }

    #[inline]
    fn momentum_mut(&mut self) -> &mut [f64] {
        self.aux.momentum_mut()
    }

    #[inline]
    fn with_position_and_gradient_mut<T, F>(&mut self, f: F) -> T
    where
        F: FnOnce(&[f64], &mut [f64]) -> T,
    {
        let position = self.position.as_ref();
        let gradient = self.aux.gradient_mut();
        f(position, gradient)
    }
}

impl<B> GradientStorage for GradientBuffers<B>
where
    B: AsRef<[f64]> + AsMut<[f64]>,
{
    #[inline]
    fn gradient(&self) -> &[f64] {
        self.gradient.as_ref()
    }

    #[inline]
    fn gradient_mut(&mut self) -> &mut [f64] {
        self.gradient.as_mut()
    }

    #[inline]
    fn momentum(&self) -> &[f64] {
        self.momentum.as_ref()
    }

    #[inline]
    fn momentum_mut(&mut self) -> &mut [f64] {
        self.momentum.as_mut()
    }
}
