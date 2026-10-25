//! Private reusable immutable phase slots. Trees own counted index handles.
//!
//! No handle escapes NUTS. A slot is writable only after removal from the free
//! list; it returns there after its last tree/edge handle is released. At the
//! next trajectory all scratch handles are detached and counts are rebuilt,
//! including after an unwinding model or metric call.

use super::*;

#[derive(Debug)]
pub(super) struct Slot<'a, T: LogDensityGradient + ?Sized> {
    pub(super) phase: PhaseState<'a, T>,
    pub(super) velocity: OwnedBuffer,
    pub(super) turning_scale: Option<f64>,
    pub(super) position_scale: f64,
    pub(super) momentum_scale: f64,
    references: usize,
}

#[derive(Debug)]
pub(super) struct StatePool<'a, T: LogDensityGradient + ?Sized> {
    slots: Vec<Slot<'a, T>>,
    free: Vec<usize>,
}

impl<'a, T: LogDensityGradient + ?Sized> StatePool<'a, T> {
    pub(super) fn new(phase: &PhaseState<'a, T>, capacity: usize) -> Self {
        Self {
            slots: (0..capacity)
                .map(|_| Slot {
                    phase: phase.clone(),
                    velocity: OwnedBuffer::new(phase.dimension()),
                    turning_scale: None,
                    position_scale: 0.0,
                    momentum_scale: 0.0,
                    references: 0,
                })
                .collect(),
            free: (0..capacity).collect(),
        }
    }

    pub(super) fn admit_turning<const POSITION: bool>(
        &mut self,
        index: usize,
        scales: Option<[f64; 2]>,
    ) {
        let slot = &mut self.slots[index];
        slot.turning_scale = scales.map(|s| s[0]);
        if let Some([a, b]) = scales {
            slot.momentum_scale = if POSITION { a } else { b };
            if POSITION {
                slot.position_scale = b;
            }
        }
    }

    pub(super) fn reset(&mut self) {
        self.free.clear();
        for (index, slot) in self.slots.iter_mut().enumerate() {
            slot.references = 0;
            self.free.push(index);
        }
    }

    pub(super) fn allocate(&mut self) -> usize {
        let index = self.free.pop().expect("bounded live NUTS phase handles");
        debug_assert_eq!(self.slots[index].references, 0);
        self.slots[index].references = 1;
        index
    }

    pub(super) fn get(&self, index: usize) -> &Slot<'a, T> {
        debug_assert!(self.slots[index].references > 0);
        &self.slots[index]
    }

    pub(super) fn vacant(&mut self, index: usize) -> &mut Slot<'a, T> {
        debug_assert_eq!(self.slots[index].references, 1);
        &mut self.slots[index]
    }

    pub(super) fn retain(&mut self, index: usize) {
        debug_assert!(self.slots[index].references > 0);
        self.slots[index].references += 1;
    }

    pub(super) fn release(&mut self, index: usize) {
        let count = &mut self.slots[index].references;
        debug_assert!(*count > 0);
        *count -= 1;
        if *count == 0 {
            self.free.push(index);
        }
    }

    pub(super) fn replace(&mut self, old: &mut usize, new: usize) {
        if *old != new {
            self.retain(new);
            self.release(*old);
            *old = new;
        }
    }
}
