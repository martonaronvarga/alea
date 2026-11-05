//! Block dispatch and contiguous pullbacks. The default preserves scalar order
//! and repeated-operand writes; explicit accelerated policies allow reassociation.
use super::{Binary, BinaryOp, Map, MatrixKernel, UnaryOp, execute};
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
    if kernel != MatrixKernel::Compensated
        && (!da || !dr || lhs != rhs)
        && seeds.iter().all(|s| *s != 0.)
    {
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
    if n == 1 && k != 0 && (!da || !dr || lhs != rhs) {
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
            mapped(b, a, r, out, 0, f);
        }
    }
}

// Keep arbitrary-rank maps compact without adding rank/division work to every
// scalar operation. Split at the two operands' innermost run boundaries; each
// inner loop uses only increments. No per-call cursor allocation or reordered
// reduction is needed. `start` permits fusion tiles to cut across map runs.
pub(super) fn for_each_index(
    b: &Binary,
    start: usize,
    len: usize,
    mut f: impl FnMut(usize, usize, usize),
) {
    let (left_run, dl) = b.left.run();
    let (right_run, dr) = b.right.run();
    let end = start + len;
    let mut j = start;
    while j < end {
        let stop = j
            + (end - j)
                .min(left_run - j % left_run)
                .min(right_run - j % right_run);
        let (mut left, mut right) = (b.left.at(j), b.right.at(j));
        while j < stop {
            f(j - start, left, right);
            left += dl;
            right += dr;
            j += 1;
        }
    }
}

pub(super) fn mapped(
    b: &Binary,
    a: &[f64],
    r: &[f64],
    out: &mut [f64],
    start: usize,
    f: impl Fn(f64, f64) -> f64,
) {
    for_each_index(b, start, out.len(), |j, left, right| {
        out[j] = f(a[left], r[right]);
    });
}

pub(super) struct Pullback<'a> {
    pub a: &'a [f64],
    pub r: &'a [f64],
    pub seeds: &'a [f64],
    pub adjoints: &'a mut [f64],
    pub starts: (usize, usize),
    pub active: (bool, bool),
}
// Distinct inclusive live ranges make alias freedom explicit to the compiler.
// Admission/seed skipping and derivative arithmetic remain shared with the
// scalar executor; slice extents are checked once rather than per coordinate.
pub(super) fn unary_reverse(op: UnaryOp, x: &[f64], y: &[f64], seeds: &[f64], scores: &mut [f64]) {
    for (((g, &seed), &x), &y) in scores.iter_mut().zip(seeds).zip(x).zip(y) {
        if seed != 0. {
            *g += seed * execute::derivative(op, x, y);
        }
    }
}

pub(super) fn unary_reverse_range(
    op: UnaryOp,
    x: &[f64],
    y: &[f64],
    adjoints: &mut [f64],
    starts: (usize, usize),
) {
    let (input, output) = starts;
    let n = x.len();
    let (seeds, scores): (&[f64], &mut [f64]) = if input < output {
        let (before, after) = adjoints.split_at_mut(output);
        (&after[..n], &mut before[input..input + n])
    } else {
        let (before, after) = adjoints.split_at_mut(input);
        (&before[output..output + n], &mut after[..n])
    };
    unary_reverse(op, x, y, seeds, scores);
}

// Keep operation/map specialization at both interpreter and tiled call sites.
// Outlining this shared wrapper adds argument traffic to every scalar scan trip.
#[inline(always)]
pub(super) fn reverse(b: &Binary, work: Pullback<'_>) {
    match b.kind {
        BinaryOp::Add => reverse_with(b, work, |_, _| (1., 1.)),
        BinaryOp::Subtract => reverse_with(b, work, |_, _| (1., -1.)),
        BinaryOp::Multiply => reverse_with(b, work, |a, r| (r, a)),
        BinaryOp::Divide => reverse_with(b, work, |a, r| (1. / r, -(a / r) / r)),
    }
}
// Same derivative expressions and repeated-operand addition order as reverse.
// Offsets apply to the semantic index maps, not to scalar/broadcast primals.
pub(super) fn reverse_range(b: &Binary, work: Pullback<'_>, start: usize) {
    let end = start + work.seeds.len();
    // Reuse the existing specialized vector loops: matching maps and the
    // derivative once per element loses the benefit of tiled execution.
    if matches!(
        (&b.left, &b.right),
        (Map::Identity, Map::Identity)
            | (Map::Identity, Map::Scalar)
            | (Map::Scalar, Map::Identity)
    ) {
        let left = matches!(b.left, Map::Identity);
        let right = matches!(b.right, Map::Identity);
        reverse(
            b,
            Pullback {
                a: if left { &work.a[start..end] } else { work.a },
                r: if right { &work.r[start..end] } else { work.r },
                seeds: work.seeds,
                adjoints: work.adjoints,
                starts: (
                    work.starts.0 + if left { start } else { 0 },
                    work.starts.1 + if right { start } else { 0 },
                ),
                active: work.active,
            },
        );
        return;
    }
    let Pullback {
        a,
        r,
        seeds,
        adjoints,
        starts: (lhs, rhs),
        active: (da, dr),
    } = work;
    for_each_index(b, start, seeds.len(), |j, l, index| {
        let seed = seeds[j];
        if seed == 0. {
            return;
        }
        let (left, right) = match b.kind {
            BinaryOp::Add => (1., 1.),
            BinaryOp::Subtract => (1., -1.),
            BinaryOp::Multiply => (r[index], a[l]),
            BinaryOp::Divide => (1. / r[index], -(a[l] / r[index]) / r[index]),
        };
        if da {
            adjoints[lhs + l] += seed * left;
        }
        if dr {
            adjoints[rhs + index] += seed * right;
        }
    });
}
#[inline(always)]
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
            for_each_index(b, 0, seeds.len(), |j, l, index| {
                let seed = seeds[j];
                if seed == 0. {
                    return;
                }
                let (a, r) = f(a[l], r[index]);
                if da {
                    adjoints[lhs + l] += seed * a;
                }
                if dr {
                    adjoints[rhs + index] += seed * r;
                }
            });
        }
    }
}
