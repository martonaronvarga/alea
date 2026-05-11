use std::{
    alloc::{alloc, alloc_zeroed, dealloc, Layout},
    mem,
    ops::{Deref, DerefMut},
    ptr::{self, NonNull},
};

#[derive(Debug)]
pub struct OwnedBuffer<T = f64> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
    layout: Option<Layout>,
}

impl<T: Clone> OwnedBuffer<T> {
    #[inline]
    pub fn fill(&mut self, value: T) {
        self.as_mut_slice().fill(value);
    }
}

impl<T> OwnedBuffer<T> {
    const ALIGN: usize = 32;

    pub fn new(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                capacity: 0,
                layout: None,
            };
        }

        let size = len.checked_mul(size_of::<T>()).expect("size overflow");
        let layout = Layout::from_size_align(size, Self::ALIGN).expect("invalid layout");

        let ptr = unsafe { alloc(layout) as *mut T };
        let ptr = NonNull::new(ptr).expect("allocation failed");

        Self {
            ptr,
            len,
            capacity: len,
            layout: Some(layout),
        }
    }

    fn layout_for(capacity: usize) -> Option<Layout> {
        let size = capacity
            .checked_mul(mem::size_of::<T>())
            .expect("size overflow");
        // Ensure alignment is at least max(ALIGN, align_of::<T>())
        let align = Self::ALIGN.max(mem::align_of::<T>());
        Layout::from_size_align(size, align).ok()
    }

    /// Resizes the buffer. If new_len > capacity, reallocates memory.
    pub fn resize(&mut self, new_len: usize) {
        if new_len <= self.capacity {
            self.len = new_len;
            return;
        }

        // 1. Calculate and validate new layout
        let new_layout = Self::layout_for(new_len).expect("invalid layout");

        // 2. Allocate new block
        let new_ptr_raw = unsafe { alloc(new_layout) };
        let new_ptr = NonNull::new(new_ptr_raw as *mut T).expect("allocation failed");

        // 3. Copy existing data
        if self.len > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr.as_ptr(), self.len);
            }
        }

        // 4. Deallocate old memory if it existed
        if let Some(old_layout) = self.layout {
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, old_layout);
            }
        }

        // 5. Update state
        self.ptr = new_ptr;
        self.capacity = new_len;
        self.layout = Some(new_layout);
        self.len = new_len;
    }

    /// Reduces length, keeping existing capacity.
    pub fn truncate(&mut self, len: usize) {
        if len < self.len {
            self.len = len;
        }
    }

    /// Sets length to zero, keeping existing capacity.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Removes a range and returns it as a standard Vec.
    pub fn drain(&mut self, range: std::ops::Range<usize>) -> Vec<T>
    where
        T: Copy,
    {
        let start = range.start.min(self.len);
        let end = range.end.min(self.len);

        let drained = self.as_slice()[start..end].to_vec();

        // Shift remaining elements to close the gap
        let remaining_start = end;
        let count = self.len - end;
        unsafe {
            ptr::copy(
                self.ptr.as_ptr().add(remaining_start),
                self.ptr.as_ptr().add(start),
                count,
            );
        }
        self.len -= end - start;
        drained
    }
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }
}

impl<T> Drop for OwnedBuffer<T> {
    fn drop(&mut self) {
        if let Some(layout) = self.layout {
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

impl<T> Deref for OwnedBuffer<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for OwnedBuffer<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
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
