//! Owned benchmark-only binding to the sealed per-model third-AD ABI.
//! Production crates remain unsafe-free and do not depend on this helper.
use crate::target::Fault;
use libloading::Library;
use std::{
    ffi::{CStr, CString, c_char, c_void},
    path::Path,
    ptr::NonNull,
    sync::Mutex,
};
type New = unsafe extern "C" fn(*const c_char, *mut c_char, usize) -> *mut c_void;
type DropModel = unsafe extern "C" fn(*mut c_void);
type Dimension = unsafe extern "C" fn(*const c_void) -> usize;
type Evaluate = unsafe extern "C" fn(
    *const c_void,
    *const f64,
    usize,
    *mut f64,
    usize,
    *mut f64,
    usize,
    *mut c_char,
    usize,
) -> i32;
struct Handle(NonNull<c_void>);
// SAFETY: the helper owns immutable Stan model data. AD tapes are created and
// consumed inside each call on that thread. The enclosing mutex serializes all
// access, and no pointer or tape escapes a call. Destruction occurs after calls join.
unsafe impl Send for Handle {}
pub struct ThirdAd {
    handle: Mutex<Handle>,
    dimension: usize,
    evaluate: Evaluate,
    destruct: DropModel,
    _library: Library,
}
impl ThirdAd {
    pub fn open(library: &Path, data: &Path) -> Result<Self, Fault> {
        // SAFETY: benchmark admission accepts only a source-sealed, locally built
        // helper. ABI/version and every required symbol are checked before creating
        // a model. Library lifetime extends through handle destruction.
        let lib = unsafe { Library::new(library) }.map_err(|e| Fault(e.to_string()))?;
        let (construct, destruct, dimension, evaluate) = unsafe {
            let version = *lib
                .get::<unsafe extern "C" fn() -> u32>(b"alea_third_abi\0")
                .map_err(|e| Fault(e.to_string()))?;
            if version() != 1 {
                return Err(Fault("third-AD ABI mismatch".into()));
            }
            (
                *lib.get::<New>(b"alea_third_new\0")
                    .map_err(|e| Fault(e.to_string()))?,
                *lib.get::<DropModel>(b"alea_third_drop\0")
                    .map_err(|e| Fault(e.to_string()))?,
                *lib.get::<Dimension>(b"alea_third_dimension\0")
                    .map_err(|e| Fault(e.to_string()))?,
                *lib.get::<Evaluate>(b"alea_third_eval\0")
                    .map_err(|e| Fault(e.to_string()))?,
            )
        };
        let path =
            CString::new(data.to_string_lossy().as_bytes()).map_err(|e| Fault(e.to_string()))?;
        let mut error = [0 as c_char; 4096];
        // SAFETY: live NUL-terminated path and writable fixed error buffer, matching
        // the checked ABI. A successful result is uniquely owned until Drop.
        let handle =
            NonNull::new(unsafe { construct(path.as_ptr(), error.as_mut_ptr(), error.len()) })
                .ok_or_else(|| message(&error))?;
        // SAFETY: the live handle was produced by this exact helper's constructor.
        let d = unsafe { dimension(handle.as_ptr()) };
        if d == 0 || d > 128 {
            // SAFETY: unique live handle has not been published or destroyed.
            unsafe { destruct(handle.as_ptr()) };
            return Err(Fault("invalid third-AD model dimension".into()));
        }
        Ok(Self {
            handle: Mutex::new(Handle(handle)),
            dimension: d,
            evaluate,
            destruct,
            _library: lib,
        })
    }
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    pub fn evaluate(&self, q: &[f64], hessian: &mut [f64], third: &mut [f64]) -> Result<(), Fault> {
        let d = self.dimension;
        if q.len() != d
            || hessian.len() != d * d
            || third.len() != d * d * d
            || q.iter().any(|x| !x.is_finite())
        {
            return Err(Fault("third-AD shape/input mismatch".into()));
        }
        let handle = self.handle.lock().unwrap_or_else(|p| p.into_inner());
        let mut error = [0 as c_char; 4096];
        // SAFETY: exact lengths checked above; disjoint mutable slices are valid for
        // the duration of the serialized call. The live model and code outlive it.
        let status = unsafe {
            (self.evaluate)(
                handle.0.as_ptr(),
                q.as_ptr(),
                q.len(),
                hessian.as_mut_ptr(),
                hessian.len(),
                third.as_mut_ptr(),
                third.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(message(&error));
        }
        if hessian.iter().chain(third.iter()).any(|x| !x.is_finite()) {
            return Err(Fault("nonfinite third-AD output".into()));
        }
        Ok(())
    }
}
impl Drop for ThirdAd {
    fn drop(&mut self) {
        let handle = self.handle.get_mut().unwrap_or_else(|p| p.into_inner());
        // SAFETY: &mut self excludes live calls; this is the sole destructor call.
        // The library field is still alive until after this Drop implementation.
        unsafe { (self.destruct)(handle.0.as_ptr()) };
    }
}
fn message(buffer: &[c_char; 4096]) -> Fault {
    // SAFETY: the zero-initialized caller buffer stays terminated even if the
    // helper supplies no message; the checked ABI writes at most length-1 bytes.
    Fault(
        unsafe { CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned(),
    )
}
