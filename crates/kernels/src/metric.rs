use crate::{buffer::OwnedBuffer, numeric::positive_finite};

#[cfg(feature = "openblas")]
#[inline]
fn dense_apply_inverse_openblas(dim: usize, lower_col_major: &[f64], src: &[f64], dst: &mut [f64]) {
    debug_assert_eq!(src.len(), dim);
    debug_assert_eq!(dst.len(), dim);

    if !core::ptr::eq(src.as_ptr(), dst.as_ptr()) {
        dst.copy_from_slice(src);
    }

    unsafe {
        cblas::dtrsv(
            cblas::Layout::ColumnMajor,
            cblas::Part::Lower,
            cblas::Transpose::None,
            cblas::Diagonal::Generic,
            dim as i32,
            lower_col_major,
            dim as i32,
            dst,
            1,
        );
        cblas::dtrsv(
            cblas::Layout::ColumnMajor,
            cblas::Part::Lower,
            cblas::Transpose::Ordinary,
            cblas::Diagonal::Generic,
            dim as i32,
            lower_col_major,
            dim as i32,
            dst,
            1,
        );
    }
}

#[cfg(feature = "openblas")]
#[inline]
fn dense_apply_sqrt_openblas(dim: usize, lower_col_major: &[f64], src: &[f64], dst: &mut [f64]) {
    debug_assert_eq!(src.len(), dim);
    debug_assert_eq!(dst.len(), dim);

    if !core::ptr::eq(src.as_ptr(), dst.as_ptr()) {
        dst.copy_from_slice(src);
    }

    unsafe {
        cblas::dtrmv(
            cblas::Layout::ColumnMajor,
            cblas::Part::Lower,
            cblas::Transpose::None,
            cblas::Diagonal::Generic,
            dim as i32,
            lower_col_major,
            dim as i32,
            dst,
            1,
        );
    }
}

#[cfg(feature = "simd")]
#[inline]
fn dense_apply_inverse_simd(dim: usize, lower_col_major: &[f64], src: &[f64], dst: &mut [f64]) {
    use std::simd::num::SimdFloat;
    use std::simd::{Simd, StdFloat};

    // f64x8 = 512-bit: one AVX-512 instruction or two AVX2.
    const LANES: usize = 8;
    type Vf = Simd<f64, LANES>;

    debug_assert_eq!(lower_col_major.len(), dim * dim);
    debug_assert_eq!(src.len(), dim);
    debug_assert_eq!(dst.len(), dim);

    if !core::ptr::eq(src.as_ptr(), dst.as_ptr()) {
        dst.copy_from_slice(src);
    }

    // Forward solve: L y = b
    //
    // Column-oriented algorithm. For column k of L (contiguous in col-major):
    //   dst[k]      /= L[k, k]
    //   dst[k+1..]  -= col_k[k+1..] * dst[k]        ← contiguous SIMD update
    //
    for k in 0..dim {
        // col_k: full column k, length dim, contiguous in memory
        let col_k = &lower_col_major[k * dim..][..dim];

        let diag = col_k[k];
        dst[k] /= diag;
        let xk = dst[k];

        let start = k + 1;
        if start >= dim {
            continue;
        }

        let col_rest = &col_k[start..]; // L[start..dim, k] — contiguous
        let dst_rest = &mut dst[start..]; // same range in dst
        let n = dst_rest.len();
        let vxk = Vf::splat(xk);

        let mut i = 0;

        while i + LANES <= n {
            let lv = Vf::from_slice(&col_rest[i..]);
            let dv = Vf::from_slice(&dst_rest[i..]);
            // FMA: dst -= col * xk  →  -col * xk + dst
            lv.mul_add(-vxk, dv).copy_to_slice(&mut dst_rest[i..]);
            i += LANES;
        }
        // Scalar tail
        while i < n {
            dst_rest[i] -= col_rest[i] * xk;
            i += 1;
        }
    }

    // Backward solve: Lᵀ x = y
    //
    // Lᵀ[i, j] = L[j, i] = lower_col_major[j + i*dim] = col_i[j]
    // Column i of L is contiguous and covers rows 0..dim.
    //
    // Back-substitution:
    //   for i = dim-1 downto 0:
    //     x[i] = (y[i] - Σ_{j>i} col_i[j] · x[j]) / L[i,i]
    //
    // split_at_mut(i+1) separates the write target (dst[i]) from the
    // read range (dst[i+1..dim]) without unsafe, satisfying the borrow checker.
    for i in (0..dim).rev() {
        let col_i = &lower_col_major[i * dim..][..dim]; // col i of L — contiguous

        // dst[..i+1]: write target includes dst[i]
        // dst[i+1..]: read-only source for the dot product
        let (dst_lo, dst_hi) = dst.split_at_mut(i + 1);

        let acc = if !dst_hi.is_empty() {
            let col_rest = &col_i[i + 1..]; // L[i+1..dim, i] = Lᵀ[i, i+1..dim]
            let n = dst_hi.len();
            let mut j = 0;

            let mut vs0 = Vf::splat(0.0);
            let mut vs1 = Vf::splat(0.0);
            let mut vs2 = Vf::splat(0.0);
            let mut vs3 = Vf::splat(0.0);
            while j + 4 * LANES <= n {
                // FMA: vsum += col * dst
                vs0 = Vf::from_slice(&col_rest[j..]).mul_add(Vf::from_slice(&dst_hi[j..]), vs0);
                vs1 = Vf::from_slice(&col_rest[j + LANES..])
                    .mul_add(Vf::from_slice(&dst_hi[j + LANES..]), vs1);
                vs2 = Vf::from_slice(&col_rest[j + 2 * LANES..])
                    .mul_add(Vf::from_slice(&dst_hi[j + 2 * LANES..]), vs2);
                vs3 = Vf::from_slice(&col_rest[j + 3 * LANES..])
                    .mul_add(Vf::from_slice(&dst_hi[j + 3 * LANES..]), vs3);
                j += 4 * LANES;
            }
            let vsum = (vs0 + vs1) + (vs2 + vs3);
            let mut s = vsum.reduce_sum();
            while j < n {
                s += col_rest[j] * dst_hi[j];
                j += 1;
            }
            s
        } else {
            0.0
        };

        dst_lo[i] = (dst_lo[i] - acc) / col_i[i];
    }
}

#[cfg(feature = "simd")]
#[inline]
fn dense_apply_sqrt_simd(dim: usize, lower_col_major: &[f64], src: &[f64], dst: &mut [f64]) {
    use std::simd::Simd;
    use std::simd::StdFloat;

    const LANES: usize = 8;
    const BLOCK: usize = 16;
    type Vf = Simd<f64, LANES>;

    debug_assert_eq!(src.len(), dim);
    debug_assert_eq!(dst.len(), dim);

    dst.fill(0.0);

    let mut jb = 0;
    while jb < dim {
        let jend = (jb + BLOCK).min(dim);

        for j in jb..jend {
            let zj = Vf::splat(src[j]);
            let col = &lower_col_major[j * dim..(j + 1) * dim];

            let mut i = j;
            while i + LANES <= dim {
                use std::simd::StdFloat;

                let c = Vf::from_slice(&col[i..i + LANES]);
                let d = Vf::from_slice(&dst[i..i + LANES]);

                let updated = zj.mul_add(c, d);
                updated.copy_to_slice(&mut dst[i..i + LANES]);

                i += LANES;
            }

            while i < dim {
                dst[i] += col[i] * src[j];
                i += 1;
            }
        }

        jb = jend;
    }
}

#[cfg(feature = "faer")]
#[inline]
fn dense_apply_inverse_faer(dim: usize, lower_col_major: &[f64], src: &[f64], dst: &mut [f64]) {
    use faer::linalg::triangular_solve::{
        solve_lower_triangular_in_place, solve_upper_triangular_in_place,
    };
    use faer::{MatMut, MatRef, Par};

    debug_assert_eq!(lower_col_major.len(), dim * dim);
    debug_assert_eq!(src.len(), dim);
    debug_assert_eq!(dst.len(), dim);

    if !core::ptr::eq(src.as_ptr(), dst.as_ptr()) {
        dst.copy_from_slice(src);
    }

    let l = MatRef::from_column_major_slice(lower_col_major, dim, dim);
    solve_lower_triangular_in_place(
        l,
        MatMut::from_column_major_slice_mut(dst, dim, 1),
        Par::Seq,
    );

    let lt = l.transpose();
    solve_upper_triangular_in_place(
        lt,
        MatMut::from_column_major_slice_mut(dst, dim, 1),
        Par::Seq,
    );
}

#[cfg(feature = "faer")]
#[inline]
fn dense_apply_sqrt_faer(dim: usize, lower_col_major: &[f64], src: &[f64], dst: &mut [f64]) {
    use faer::linalg::matmul::triangular::{matmul_with_conj, BlockStructure};
    use faer::{Accum, Conj, MatMut, MatRef, Par};

    debug_assert_eq!(lower_col_major.len(), dim * dim);
    debug_assert_eq!(src.len(), dim);
    debug_assert_eq!(dst.len(), dim);

    let lhs = MatRef::from_column_major_slice(lower_col_major, dim, dim);
    let rhs = MatRef::from_column_major_slice(src, dim, 1);
    let dst = MatMut::from_column_major_slice_mut(dst, dim, 1);

    matmul_with_conj(
        dst,
        BlockStructure::Rectangular,
        Accum::Replace,
        lhs,
        BlockStructure::TriangularLower,
        Conj::No,
        rhs,
        BlockStructure::Rectangular,
        Conj::No,
        1.0,
        Par::Seq,
    );
}

#[allow(dead_code)]
#[inline]
fn dense_apply_inverse_default(dim: usize, lower_col_major: &[f64], src: &[f64], dst: &mut [f64]) {
    debug_assert_eq!(src.len(), dim);
    debug_assert_eq!(dst.len(), dim);

    if !core::ptr::eq(src.as_ptr(), dst.as_ptr()) {
        dst.copy_from_slice(src);
    }

    const BLOCK: usize = 32;

    let mut jb = 0;
    while jb < dim {
        let jend = (jb + BLOCK).min(dim);

        for i in jb..jend {
            let mut acc = dst[i];

            let mut kb = 0;
            while kb < jb {
                let kend = (kb + BLOCK).min(jb);
                let mut k = kb;
                while k < kend {
                    acc -= lower_col_major[i + k * dim] * dst[k];
                    k += 1;
                }
                kb = kend;
            }

            let mut k = jb;
            while k < i {
                acc -= lower_col_major[i + k * dim] * dst[k];
                k += 1;
            }

            dst[i] = acc / lower_col_major[i + i * dim];
        }

        jb = jend;
    }

    let mut jb = dim;
    while jb > 0 {
        let j0 = jb.saturating_sub(BLOCK);

        for i in (j0..jb).rev() {
            let mut acc = dst[i];

            let mut kb = jb;
            while kb < dim {
                let kend = (kb + BLOCK).min(dim);
                let mut k = kb;
                while k < kend {
                    acc -= lower_col_major[k + i * dim] * dst[k];
                    k += 1;
                }
                kb = kend;
            }

            let mut k = i + 1;
            while k < jb {
                acc -= lower_col_major[k + i * dim] * dst[k];
                k += 1;
            }

            dst[i] = acc / lower_col_major[i + i * dim];
        }

        jb = j0;
    }
}

#[allow(dead_code)]
#[inline]
fn dense_apply_sqrt_default(dim: usize, lower_col_major: &[f64], src: &[f64], dst: &mut [f64]) {
    debug_assert_eq!(src.len(), dim);
    debug_assert_eq!(dst.len(), dim);

    dst.fill(0.0);

    const BLOCK: usize = 32;

    let mut jb = 0;
    while jb < dim {
        let jend = (jb + BLOCK).min(dim);

        for j in jb..jend {
            let zj = src[j];
            let col = &lower_col_major[j * dim..(j + 1) * dim];

            let mut i = j;
            while i + 4 <= dim {
                dst[i] += col[i] * zj;
                dst[i + 1] += col[i + 1] * zj;
                dst[i + 2] += col[i + 2] * zj;
                dst[i + 3] += col[i + 3] * zj;
                i += 4;
            }

            while i < dim {
                dst[i] += col[i] * zj;
                i += 1;
            }
        }

        jb = jend;
    }
}

pub trait Metric {
    fn dim(&self) -> usize;
    fn apply_inverse(&self, src: &[f64], dst: &mut [f64]);
    fn apply_sqrt(&self, src: &[f64], dst: &mut [f64]);
    fn log_det(&self) -> f64;
}

#[derive(Debug, Clone, Copy)]
pub struct IdentityMetric {
    dim: usize,
}

impl IdentityMetric {
    #[inline]
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Metric for IdentityMetric {
    #[inline]
    fn dim(&self) -> usize {
        self.dim
    }
    #[inline]
    fn apply_sqrt(&self, src: &[f64], dst: &mut [f64]) {
        dst.copy_from_slice(src);
    }

    #[inline]
    fn apply_inverse(&self, src: &[f64], dst: &mut [f64]) {
        dst.copy_from_slice(src);
    }

    #[inline]
    fn log_det(&self) -> f64 {
        0.0
    }
}

#[derive(Debug)]
pub struct DiagonalMetric {
    diag: OwnedBuffer,
}

impl DiagonalMetric {
    pub fn new(mut diag: OwnedBuffer) -> Self {
        for mut d in diag.iter_mut() {
            *d = positive_finite(*d);
        }

        Self { diag }
    }
}

impl Metric for DiagonalMetric {
    #[inline]
    fn dim(&self) -> usize {
        self.diag.len()
    }

    #[inline]
    fn apply_inverse(&self, src: &[f64], dst: &mut [f64]) {
        for ((out, x), d) in dst.iter_mut().zip(src).zip(self.diag.iter()) {
            *out = *x / *d;
        }
    }

    #[inline]
    fn apply_sqrt(&self, src: &[f64], dst: &mut [f64]) {
        for ((out, x), d) in dst.iter_mut().zip(src).zip(self.diag.iter()) {
            *out = *x * d.sqrt();
        }
    }

    #[inline]
    fn log_det(&self) -> f64 {
        self.diag.iter().map(|x| x.ln()).sum()
    }
}

#[derive(Debug)]
pub struct CholeskyFactor {
    dim: usize,
    lower_col_major: OwnedBuffer,
}

impl CholeskyFactor {
    pub fn new_lower(dim: usize, lower_col_major: OwnedBuffer) -> Self {
        assert_eq!(lower_col_major.len(), dim * dim);
        for i in 0..dim {
            let d = lower_col_major[i + i * dim];
            assert!(d.is_finite() && d > 0.0);
        }
        Self {
            dim,
            lower_col_major,
        }
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        self.lower_col_major.as_slice()
    }
}

#[derive(Debug)]
pub struct DenseMetric {
    factor: CholeskyFactor,
}

impl DenseMetric {
    pub fn new(factor: CholeskyFactor) -> Self {
        Self { factor }
    }
    #[inline]
    pub fn factor(&self) -> &CholeskyFactor {
        &self.factor
    }
}

impl Metric for DenseMetric {
    #[inline]
    fn dim(&self) -> usize {
        self.factor.dim
    }

    fn apply_inverse(&self, src: &[f64], dst: &mut [f64]) {
        let n = self.factor.dim;
        let l = self.factor.as_slice();
        #[cfg(feature = "openblas")]
        {
            dense_apply_inverse_openblas(n, l, src, dst);
            return;
        }
        #[cfg(all(not(feature = "openblas"), feature = "faer"))]
        {
            dense_apply_inverse_faer(n, l, src, dst);
            return;
        }
        #[cfg(all(not(feature = "openblas"), not(feature = "faer"), feature = "simd"))]
        {
            dense_apply_inverse_simd(n, l, src, dst);
            return;
        }
        #[cfg(all(
            not(feature = "openblas"),
            not(feature = "faer"),
            not(feature = "simd")
        ))]
        {
            dense_apply_inverse_default(n, l, src, dst);
        }
    }

    fn apply_sqrt(&self, src: &[f64], dst: &mut [f64]) {
        let n = self.factor.dim;
        let l = self.factor.as_slice();
        #[cfg(feature = "openblas")]
        {
            dense_apply_sqrt_openblas(n, l, src, dst);
            return;
        }
        #[cfg(all(not(feature = "openblas"), feature = "faer"))]
        {
            dense_apply_sqrt_faer(n, l, src, dst);
            return;
        }
        #[cfg(all(not(feature = "openblas"), not(feature = "faer"), feature = "simd"))]
        {
            dense_apply_sqrt_simd(n, l, src, dst);
            return;
        }
        #[cfg(all(
            not(feature = "openblas"),
            not(feature = "faer"),
            not(feature = "simd")
        ))]
        {
            dense_apply_sqrt_default(n, l, src, dst);
        }
    }

    #[inline]
    fn log_det(&self) -> f64 {
        let n = self.factor.dim;
        let l = self.factor.as_slice();
        let mut sum = 0.0;
        for i in 0..n {
            sum += l[i + i * n].ln();
        }
        2.0 * sum
    }
}
