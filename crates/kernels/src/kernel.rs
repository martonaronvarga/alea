use rand::Rng;

pub trait Kernel<D: ?Sized> {
    type State;

    fn initialize(&mut self, _state: &mut Self::State, _target: &D) {}
    fn step<R: Rng + ?Sized>(&mut self, state: &mut Self::State, target: &D, rng: &mut R) -> bool;
}
