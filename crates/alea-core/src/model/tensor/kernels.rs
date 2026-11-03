//! Block dispatch and contiguous pullbacks. The default preserves scalar order
//! and repeated-operand writes; explicit accelerated policies allow reassociation.
use super::{Binary, BinaryOp, Map, MatrixKernel};
use alea_math::gemm::try_gemm;
use alea_math::numeric::CompensatedSum;

// Keep matrix-sized loop bodies out of the shared pointwise/control interpreter.
// These calls amortize over a block, not one call per matrix element.
#[inline(never)]
pub(super) fn matmul_forward(
    a: &[f64],
    b: &[f64],
    out: &mut [f64],
    (m, k, n): (usize, usize, usize),
    kernel: MatrixKernel,
) {
    if try_gemm(kernel, a, b, out, (m, k, n), (false, false), false) {
        return;
    }
    if n == 1 && k != 0 {
        for (left, out) in a.chunks_exact(k).zip(out) {
            let mut sum = CompensatedSum::default();
            for (&a, &b) in left.iter().zip(b) {
                sum.add_product(a, b);
            }
            *out = sum.total();
        }
    } else {
        for row in 0..m {
            for column in 0..n {
                let mut sum = CompensatedSum::default();
                for j in 0..k {
                    sum.add_product(a[row * k + j], b[j * n + column]);
                }
                out[row * n + column] = sum.total();
            }
        }
    }
}

#[inline(never)]
pub(super) fn matmul_reverse(
    (m, k, n): (usize, usize, usize),
    work: Pullback<'_>,
    kernel: MatrixKernel,
) {
    let Pullback {
        a,
        r,
        seeds,
        adjoints,
        starts: (lhs, rhs),
        active: (da, dr),
    } = work;
    // Preserve skipped derivatives and original interleaving for aliased inputs.
    // Each GEMM writes only its active destination, including existing adjoints.
    if kernel != MatrixKernel::Compensated && lhs != rhs && seeds.iter().all(|s| *s != 0.) {
        let left_done = !da
            || try_gemm(
                kernel,
                seeds,
                r,
                &mut adjoints[lhs..lhs + a.len()],
                (m, n, k),
                (false, true),
                true,
            );
        let right_done = !dr
            || try_gemm(
                kernel,
                a,
                seeds,
                &mut adjoints[rhs..rhs + r.len()],
                (k, m, n),
                (true, false),
                true,
            );
        // Unsupported backend dimensions decline before writes. Do not repeat a
        // destination that was already accumulated by a successful GEMM.
        if left_done && right_done {
            return;
        }
        matmul_reverse(
            (m, k, n),
            Pullback {
                a,
                r,
                seeds,
                adjoints,
                starts: (lhs, rhs),
                active: (da && !left_done, dr && !right_done),
            },
            MatrixKernel::Compensated,
        );
        return;
    }
    if n == 1 && k != 0 && lhs != rhs {
        for (row, (left, &seed)) in a.chunks_exact(k).zip(seeds).enumerate() {
            if seed == 0. {
                continue;
            }
            if da {
                for (g, &value) in adjoints[lhs + row * k..lhs + (row + 1) * k]
                    .iter_mut()
                    .zip(r)
                {
                    *g += seed * value;
                }
            }
            if dr {
                for (g, &value) in adjoints[rhs..rhs + k].iter_mut().zip(left) {
                    *g += seed * value;
                }
            }
        }
    } else {
        // Shared storage needs the original interleaved additions. Neither
        // operand may receive a computed derivative when it is inactive.
        for row in 0..m {
            for column in 0..n {
                let seed = seeds[row * n + column];
                if seed == 0. {
                    continue;
                }
                for j in 0..k {
                    if da {
                        adjoints[lhs + row * k + j] += seed * r[j * n + column];
                    }
                    if dr {
                        adjoints[rhs + j * n + column] += seed * a[row * k + j];
                    }
                }
            }
        }
    }
}

pub(super) fn forward(b: &Binary, a: &[f64], r: &[f64], out: &mut [f64]) {
    match b.kind {
        BinaryOp::Add => forward_with(b, a, r, out, |a, r| a + r),
        BinaryOp::Subtract => forward_with(b, a, r, out, |a, r| a - r),
        BinaryOp::Multiply => forward_with(b, a, r, out, |a, r| a * r),
        BinaryOp::Divide => forward_with(b, a, r, out, |a, r| a / r),
    }
}
fn forward_with(b: &Binary, a: &[f64], r: &[f64], out: &mut [f64], f: impl Fn(f64, f64) -> f64) {
    match (&b.left, &b.right) {
        (Map::Identity, Map::Identity) => {
            for ((out, &a), &r) in out.iter_mut().zip(a).zip(r) {
                *out = f(a, r);
            }
        }
        (Map::Scalar, Map::Identity) => {
            let a = a[0];
            for (out, &r) in out.iter_mut().zip(r) {
                *out = f(a, r);
            }
        }
        (Map::Identity, Map::Scalar) => {
            let r = r[0];
            for (out, &a) in out.iter_mut().zip(a) {
                *out = f(a, r);
            }
        }
        _ => {
            for (j, out) in out.iter_mut().enumerate() {
                *out = f(a[b.left.at(j)], r[b.right.at(j)]);
            }
        }
    }
}

pub(super) struct Pullback<'a> {
    pub a: &'a [f64],
    pub r: &'a [f64],
    pub seeds: &'a [f64],
    pub adjoints: &'a mut [f64],
    pub starts: (usize, usize),
    pub active: (bool, bool),
}
pub(super) fn reverse(b: &Binary, work: Pullback<'_>) {
    match b.kind {
        BinaryOp::Add => reverse_with(b, work, |_, _| (1., 1.)),
        BinaryOp::Subtract => reverse_with(b, work, |_, _| (1., -1.)),
        BinaryOp::Multiply => reverse_with(b, work, |a, r| (r, a)),
        BinaryOp::Divide => reverse_with(b, work, |a, r| (1. / r, -(a / r) / r)),
    }
}
fn reverse_with(b: &Binary, work: Pullback<'_>, f: impl Fn(f64, f64) -> (f64, f64)) {
    let Pullback {
        a,
        r,
        seeds,
        adjoints,
        starts: (lhs, rhs),
        active: (da, dr),
    } = work;
    match (&b.left, &b.right) {
        (Map::Identity, Map::Identity) => {
            if lhs == rhs {
                // Reshapes can alias the same storage. Preserve the original
                // interleaved two additions, not seed*(da+db) or 2*seed*x.
                for (((g, &seed), &a), &r) in adjoints[lhs..lhs + a.len()]
                    .iter_mut()
                    .zip(seeds)
                    .zip(a)
                    .zip(r)
                {
                    if seed != 0. {
                        let (a, r) = f(a, r);
                        if da {
                            *g += seed * a;
                        }
                        if dr {
                            *g += seed * r;
                        }
                    }
                }
            } else {
                // Allocated inputs are disjoint (reshape aliases share starts).
                // Each destination's row order remains identical across passes.
                if da {
                    for (((g, &seed), &a), &r) in adjoints[lhs..lhs + a.len()]
                        .iter_mut()
                        .zip(seeds)
                        .zip(a)
                        .zip(r)
                    {
                        if seed != 0. {
                            *g += seed * f(a, r).0;
                        }
                    }
                }
                if dr {
                    for (((g, &seed), &a), &r) in adjoints[rhs..rhs + r.len()]
                        .iter_mut()
                        .zip(seeds)
                        .zip(a)
                        .zip(r)
                    {
                        if seed != 0. {
                            *g += seed * f(a, r).1;
                        }
                    }
                }
            }
        }
        (Map::Scalar, Map::Identity) if da && !dr => {
            let mut g = adjoints[lhs];
            let a = a[0];
            for (&seed, &r) in seeds.iter().zip(r) {
                if seed != 0. {
                    g += seed * f(a, r).0;
                }
            }
            adjoints[lhs] = g;
        }
        (Map::Identity, Map::Scalar) if !da && dr => {
            let mut g = adjoints[rhs];
            let r = r[0];
            for (&seed, &a) in seeds.iter().zip(a) {
                if seed != 0. {
                    g += seed * f(a, r).1;
                }
            }
            adjoints[rhs] = g;
        }
        (Map::Scalar, Map::Identity) if !da && dr => {
            let a = a[0];
            for ((g, &seed), &r) in adjoints[rhs..rhs + r.len()].iter_mut().zip(seeds).zip(r) {
                if seed != 0. {
                    *g += seed * f(a, r).1;
                }
            }
        }
        (Map::Identity, Map::Scalar) if da && !dr => {
            let r = r[0];
            for ((g, &seed), &a) in adjoints[lhs..lhs + a.len()].iter_mut().zip(seeds).zip(a) {
                if seed != 0. {
                    *g += seed * f(a, r).0;
                }
            }
        }
        _ => {
            for (j, &seed) in seeds.iter().enumerate() {
                if seed == 0. {
                    continue;
                }
                let l = b.left.at(j);
                let index = b.right.at(j);
                let (a, r) = f(a[l], r[index]);
                if da {
                    adjoints[lhs + l] += seed * a;
                }
                if dr {
                    adjoints[rhs + index] += seed * r;
                }
            }
        }
    }
}
