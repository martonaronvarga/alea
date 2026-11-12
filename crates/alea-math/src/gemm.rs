//! Checked row-major GEMM adapters. Accelerated reductions are not compensated
//! and need not reproduce scalar reduction bits. No global thread policy changes.

/// Numerical requirements for dense matrix products.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum MatrixArithmetic {
    /// Ordered compensated reference arithmetic.
    #[default]
    Compensated,
    /// Permits provider reassociation; no compensated accuracy equivalence.
    Standard,
}
/// Provider preference, independent of numerical requirements. Unavailable
/// explicit providers remain expressible and fail during resolution.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum MatrixProvider {
    /// Ordered reference implementation.
    Reference,
    /// Choose only among implementations compatible with the arithmetic request.
    #[default]
    Auto,
    /// Explicit Faer request, even when the feature is absent.
    Faer,
    /// Explicit LP64 BLAS request, even when the feature is absent.
    Blas,
}
/// CPU implementation preference. Provider libraries cannot promise scalar ISA.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum MatrixCpu {
    /// Permit the selected library to dispatch its ISA.
    #[default]
    Auto,
    /// Reference scalar-loop implementation; not a compiler vectorization flag.
    Scalar,
}
/// Thread ownership. HostConfigured permits external BLAS configuration; it is
/// not a measured or enforced thread-count bound. No global setting is changed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum MatrixThreads {
    /// Require the reference or sequential Faer implementation.
    #[default]
    Sequential,
    /// Permit BLAS threading configured and owned by the caller.
    HostConfigured,
}
/// Cold per-operation matrix request. Cargo features never relax arithmetic.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MatrixRequest {
    /// Required accumulation contract.
    pub arithmetic: MatrixArithmetic,
    /// Preferred compatible provider.
    pub provider: MatrixProvider,
    /// CPU implementation preference, independent of arithmetic.
    pub cpu: MatrixCpu,
    /// Thread ownership requirement; no process-global setting is mutated.
    pub threads: MatrixThreads,
}
/// Rejection before executing or generating a matrix operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum MatrixPolicyError {
    #[error("matrix provider is unavailable: {0:?}")]
    Unavailable(MatrixProvider),
    #[error("matrix request is incompatible: {0:?}")]
    Incompatible(MatrixRequest),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Implementation {
    Reference,
    #[cfg(feature = "faer")]
    Faer,
    #[cfg(feature = "openblas")]
    Blas,
}
/// Resolved physical call and its original request. Constructed by cold
/// resolution, cached by the builder, and copied into checked GEMM calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedMatrix {
    request: MatrixRequest,
    implementation: Implementation,
}
impl MatrixRequest {
    /// Explicitly permits provider arithmetic, leaving BLAS threads host-owned.
    pub const fn standard(provider: MatrixProvider) -> Self {
        Self {
            arithmetic: MatrixArithmetic::Standard,
            provider,
            cpu: MatrixCpu::Auto,
            threads: MatrixThreads::HostConfigured,
        }
    }
    /// Resolves availability and compatibility once without executing arithmetic,
    /// probing process-global BLAS state, or creating worker pools.
    /// # Errors
    /// Rejects unavailable providers or unsupported arithmetic/ISA/thread requests.
    /// Reference Standard has no existing conventional GEMM and is rejected.
    pub fn resolve(self) -> Result<ResolvedMatrix, MatrixPolicyError> {
        use MatrixProvider as P;
        let incompatible = || MatrixPolicyError::Incompatible(self);
        let implementation = match self.provider {
            P::Reference | P::Auto if self.arithmetic == MatrixArithmetic::Compensated => {
                Implementation::Reference
            }
            P::Reference => return Err(incompatible()),
            P::Auto => {
                if self.cpu == MatrixCpu::Scalar {
                    return Err(incompatible());
                }
                #[cfg(feature = "faer")]
                {
                    Implementation::Faer
                }
                #[cfg(all(not(feature = "faer"), feature = "openblas"))]
                {
                    if self.threads == MatrixThreads::Sequential {
                        return Err(incompatible());
                    }
                    Implementation::Blas
                }
                #[cfg(not(any(feature = "faer", feature = "openblas")))]
                {
                    return Err(MatrixPolicyError::Unavailable(P::Auto));
                }
            }
            P::Faer => {
                if self.arithmetic != MatrixArithmetic::Standard || self.cpu == MatrixCpu::Scalar {
                    return Err(incompatible());
                }
                #[cfg(feature = "faer")]
                {
                    Implementation::Faer
                }
                #[cfg(not(feature = "faer"))]
                {
                    return Err(MatrixPolicyError::Unavailable(P::Faer));
                }
            }
            P::Blas => {
                if self.arithmetic != MatrixArithmetic::Standard
                    || self.cpu == MatrixCpu::Scalar
                    || self.threads != MatrixThreads::HostConfigured
                {
                    return Err(incompatible());
                }
                #[cfg(feature = "openblas")]
                {
                    Implementation::Blas
                }
                #[cfg(not(feature = "openblas"))]
                {
                    return Err(MatrixPolicyError::Unavailable(P::Blas));
                }
            }
        };
        Ok(ResolvedMatrix {
            request: self,
            implementation,
        })
    }
}
impl ResolvedMatrix {
    /// Canonical physical reference call used by generated source.
    pub const COMPENSATED: Self = Self {
        request: MatrixRequest {
            arithmetic: MatrixArithmetic::Compensated,
            provider: MatrixProvider::Auto,
            cpu: MatrixCpu::Auto,
            threads: MatrixThreads::Sequential,
        },
        implementation: Implementation::Reference,
    };
    #[cfg(feature = "faer")]
    /// Canonical physical sequential Faer call used by generated source.
    pub const FAER: Self = Self {
        request: MatrixRequest::standard(MatrixProvider::Faer),
        implementation: Implementation::Faer,
    };
    #[cfg(feature = "openblas")]
    /// Canonical physical host-configured BLAS call used by generated source.
    pub const BLAS: Self = Self {
        request: MatrixRequest::standard(MatrixProvider::Blas),
        implementation: Implementation::Blas,
    };
    /// Original caller request; resolution does not erase Auto preferences.
    pub fn requested(self) -> MatrixRequest {
        self.request
    }
    /// Actual provider, with no unresolved Auto variant.
    pub fn provider(self) -> MatrixProvider {
        match self.implementation {
            Implementation::Reference => MatrixProvider::Reference,
            #[cfg(feature = "faer")]
            Implementation::Faer => MatrixProvider::Faer,
            #[cfg(feature = "openblas")]
            Implementation::Blas => MatrixProvider::Blas,
        }
    }
    /// Actual CPU contract. Accelerated libraries retain their own ISA dispatch.
    pub fn cpu(self) -> MatrixCpu {
        if self.is_accelerated() {
            MatrixCpu::Auto
        } else {
            MatrixCpu::Scalar
        }
    }
    /// Actual thread ownership. Faer and reference execution are sequential.
    pub fn threads(self) -> MatrixThreads {
        if self.provider() == MatrixProvider::Blas {
            MatrixThreads::HostConfigured
        } else {
            MatrixThreads::Sequential
        }
    }
    /// Whether this call uses reassociated provider arithmetic.
    pub fn is_accelerated(self) -> bool {
        self.implementation != Implementation::Reference
    }
    /// Static physical call spelling for source rendering, not route selection.
    pub fn source_name(self) -> &'static str {
        match self.implementation {
            Implementation::Reference => "COMPENSATED",
            #[cfg(feature = "faer")]
            Implementation::Faer => "FAER",
            #[cfg(feature = "openblas")]
            Implementation::Blas => "BLAS",
        }
    }
}

/// Attempts `out = op(a) * op(b)` (or adds the product when `add` is true).
/// Dimensions describe the logical product `(m, k, n)`; transpose flags describe
/// the physical row-major operands. Returns false without writes for the
/// compensated policy, invalid lengths/overflow, or unsupported BLAS dimensions.
/// Empty products are handled without calling foreign code. Numerical admission
/// of inputs/outputs belongs to the caller; no clipping or finite repair occurs.
pub fn try_gemm(
    kernel: ResolvedMatrix,
    a: &[f64],
    b: &[f64],
    out: &mut [f64],
    dimensions: (usize, usize, usize),
    transpose: (bool, bool),
    add: bool,
) -> bool {
    let (m, k, n) = dimensions;
    if !kernel.is_accelerated()
        || m.checked_mul(k) != Some(a.len())
        || k.checked_mul(n) != Some(b.len())
        || m.checked_mul(n) != Some(out.len())
    {
        return false;
    }
    if m == 0 || n == 0 || k == 0 {
        if !add {
            out.fill(0.);
        }
        return true;
    }
    let (ta, tb) = transpose;
    let _ = (ta, tb); // Both optional backends can be disabled.
    match kernel.implementation {
        Implementation::Reference => false,
        #[cfg(feature = "faer")]
        Implementation::Faer => {
            use faer::{Accum, MatMut, MatRef, Par};
            let a = if ta {
                MatRef::from_row_major_slice(a, k, m).transpose()
            } else {
                MatRef::from_row_major_slice(a, m, k)
            };
            let b = if tb {
                MatRef::from_row_major_slice(b, n, k).transpose()
            } else {
                MatRef::from_row_major_slice(b, k, n)
            };
            faer::linalg::matmul::matmul(
                MatMut::from_row_major_slice_mut(out, m, n),
                if add { Accum::Add } else { Accum::Replace },
                a,
                b,
                1.,
                Par::Seq,
            );
            true
        }
        #[cfg(feature = "openblas")]
        Implementation::Blas => {
            use cblas::{Layout, Transpose, dgemm};
            let (Ok(m), Ok(k), Ok(n)) = (i32::try_from(m), i32::try_from(k), i32::try_from(n))
            else {
                return false;
            };
            // SAFETY: checked exact slice lengths and positive LP64 dimensions.
            // Physical row widths are lda=(ta?m:k), ldb=(tb?k:n), ldc=n.
            // Rust borrows guarantee output cannot overlap either input. All
            // initialized buffers stay borrowed until this synchronous call ends.
            unsafe {
                dgemm(
                    Layout::RowMajor,
                    if ta {
                        Transpose::Ordinary
                    } else {
                        Transpose::None
                    },
                    if tb {
                        Transpose::Ordinary
                    } else {
                        Transpose::None
                    },
                    m,
                    n,
                    k,
                    1.,
                    a,
                    if ta { m } else { k },
                    b,
                    if tb { k } else { n },
                    if add { 1. } else { 0. },
                    out,
                    n,
                );
            }
            true
        }
    }
}
