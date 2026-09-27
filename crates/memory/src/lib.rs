//! Initialized, SIMD-aligned storage, isolated from FFI for Miri validation.
#![forbid(unsafe_code)]

use aligned_vec::{AVec, ConstAlign};
use std::ops::{Deref, DerefMut};

/// Contiguous initialized storage with at least 64-byte base alignment.
///
/// The allocation uses `max(64, align_of::<T>())`; elements remain tightly packed,
/// with no per-element padding. Subslice offsets do not inherit base alignment.
/// Allocations and growth preserve alignment. Reuse capacity in numerical loops.
///
/// This safe wrapper exposes only initialized storage from `aligned-vec`.
/// It deliberately does not expose raw-parts or unchecked length APIs.
#[derive(Debug, Clone)]
pub struct OwnedBuffer<T = f64> {
    data: AVec<T, ConstAlign<64>>,
}

impl<T: Default> OwnedBuffer<T> {
    /// Creates `len` initialized elements.
    pub fn new(len: usize) -> Self {
        Self::from_fn(len, |_| T::default())
    }

    /// Resizes with default values, dropping removed elements and retaining capacity.
    pub fn resize(&mut self, new_len: usize) {
        if new_len <= self.len() {
            self.truncate(new_len);
        } else {
            self.data.reserve(new_len - self.len());
            while self.len() < new_len {
                self.data.push(T::default());
            }
        }
    }
}

impl<T> OwnedBuffer<T> {
    /// Builds initialized elements directly, avoiding zero-fill followed by overwrite.
    /// If `init` panics, already constructed elements are dropped.
    pub fn from_fn(len: usize, init: impl FnMut(usize) -> T) -> Self {
        Self {
            data: AVec::from_iter(64, (0..len).map(init)),
        }
    }

    /// Guaranteed base alignment; subslices may have a weaker alignment.
    #[inline]
    pub fn alignment(&self) -> usize {
        self.data.alignment()
    }

    /// Drops elements beyond `len`, retaining capacity.
    pub fn truncate(&mut self, len: usize) {
        self.data.truncate(len);
    }

    /// Drops all elements, retaining capacity.
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Removes and returns a range, clamping endpoints to the current length.
    ///
    /// # Panics
    /// Panics if the clamped start exceeds the clamped end.
    pub fn drain(&mut self, range: std::ops::Range<usize>) -> Vec<T> {
        let start = range.start.min(self.len());
        let end = range.end.min(self.len());
        assert!(start <= end, "drain start exceeds end");
        let count = end - start;
        // Allocate before mutation so allocation failure cannot partially drain.
        let mut removed = Vec::with_capacity(count);
        self.data.as_mut_slice()[start..].rotate_left(count);
        for _ in 0..count {
            removed.push(self.data.pop().expect("drain count is within length"));
        }
        removed.reverse();
        removed
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T> Deref for OwnedBuffer<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> DerefMut for OwnedBuffer<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> AsRef<[T]> for OwnedBuffer<T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AsMut<[T]> for OwnedBuffer<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

#[derive(Clone, Copy)]
pub struct VecView<'a, T> {
    data: &'a [T],
}

pub struct VecViewMut<'a, T> {
    data: &'a mut [T],
}

impl<'a, T> VecView<'a, T> {
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a, T> VecViewMut<'a, T> {
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, rc::Rc};

    #[test]
    fn growth_initializes_and_shrink_preserves_capacity() {
        let mut buffer = OwnedBuffer::<String>::new(2);
        buffer[0].push_str("kept");
        buffer.resize(128);
        assert_eq!(buffer[0], "kept");
        assert!(buffer[1..].iter().all(String::is_empty));
        let capacity = buffer.capacity();
        buffer.truncate(1);
        buffer.resize(3);
        assert_eq!(buffer.capacity(), capacity);
        assert!(buffer[1..].iter().all(String::is_empty));
        assert_eq!(buffer.drain(0..1), ["kept"]);
        buffer.clear();
        assert!(buffer.is_empty());
        assert_eq!(buffer.capacity(), capacity);
    }

    #[test]
    fn removed_and_remaining_elements_are_dropped_once() {
        #[derive(Default)]
        struct Tracked(Option<Rc<Cell<usize>>>);
        impl Drop for Tracked {
            fn drop(&mut self) {
                if let Some(count) = &self.0 {
                    count.set(count.get() + 1);
                }
            }
        }
        let count = Rc::new(Cell::new(0));
        let mut buffer = OwnedBuffer::<Tracked>::new(4);
        for item in buffer.iter_mut() {
            item.0 = Some(Rc::clone(&count));
        }
        buffer.truncate(3);
        assert_eq!(count.get(), 1);
        let drained = buffer.drain(0..1);
        assert_eq!(count.get(), 1);
        drop(drained);
        assert_eq!(count.get(), 2);
        buffer.resize(1);
        assert_eq!(count.get(), 3);
        drop(buffer);
        assert_eq!(count.get(), 4);
    }

    #[test]
    fn supports_zero_sized_and_overaligned_elements() {
        #[repr(align(128))]
        #[derive(Default)]
        struct Aligned(u8);
        let mut buffer = OwnedBuffer::<Aligned>::new(0);
        buffer.resize(5);
        assert_eq!(buffer.as_ptr().addr() % 128, 0);
        assert!(buffer.iter().all(|x| x.0 == 0));
        let mut zst = OwnedBuffer::<()>::new(4);
        zst.resize(100);
        assert_eq!(zst.drain(10..90).len(), 80);
        assert_eq!(zst.len(), 20);
    }

    #[test]
    fn base_alignment_survives_growth_clone_and_reuse() {
        assert_eq!(size_of::<OwnedBuffer>(), size_of::<Vec<f64>>());
        for len in [0, 1, 7, 8, 9, 31, 64, 257] {
            let mut buffer = OwnedBuffer::from_fn(len, |i| i as f64);
            assert_eq!(buffer.alignment(), 64);
            assert_eq!(buffer.as_ptr().addr() % 64, 0);
            let cloned = buffer.clone();
            assert_eq!(cloned.as_ptr().addr() % 64, 0);
            assert_eq!(cloned.as_slice(), buffer.as_slice());
            buffer.resize(len + 33);
            assert_eq!(buffer.as_ptr().addr() % 64, 0);
            assert!(buffer[len..].iter().all(|&x| x == 0.0));
            let pointer = buffer.as_ptr();
            buffer.clear();
            buffer.resize(len);
            assert_eq!(buffer.as_ptr(), pointer);
            assert_eq!(buffer.as_ptr().addr() % 64, 0);
        }
    }

    #[test]
    fn initialization_panic_drops_constructed_values() {
        struct Tracked(Rc<Cell<usize>>);
        impl Drop for Tracked {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        let count = Rc::new(Cell::new(0));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            OwnedBuffer::from_fn(5, |i| {
                assert_ne!(i, 3, "intentional initializer panic");
                Tracked(Rc::clone(&count))
            })
        }));
        assert!(result.is_err());
        assert_eq!(count.get(), 3);
    }

    #[test]
    fn drain_preserves_order_and_reuses_allocation() {
        let mut buffer = OwnedBuffer::from_fn(6, |i| i.to_string());
        let pointer = buffer.as_ptr();
        assert_eq!(buffer.drain(2..4), ["2", "3"]);
        assert_eq!(buffer.as_slice(), ["0", "1", "4", "5"]);
        assert_eq!(buffer.as_ptr(), pointer);
        assert!(buffer.drain(10..20).is_empty());
        assert_eq!(buffer.drain(2..20), ["4", "5"]);
        assert_eq!(buffer.as_slice(), ["0", "1"]);
    }
}
