"""Independent 120-digit dense 2x2 and batch-scatter Fisher oracles.

Python standard library only. No Alea, Faer, MGS or streaming-Welford calls.
Uses the closed 2x2 principal square root and the C-oriented Riccati solution,
not Alea's F-oriented reduced eigensolve. Emits fixtures; --check verifies bytes.
These are mathematical fixtures, NOT executed nuts-rs warmup traces.
"""
import argparse
import csv
from decimal import Decimal as D, localcontext
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(2)) for j in range(2)] for i in range(2)]


def determinant(a):
    return a[0][0] * a[1][1] - a[0][1] * a[1][0]


def inverse(a):
    det = determinant(a)
    return [[a[1][1] / det, -a[0][1] / det], [-a[1][0] / det, a[0][0] / det]]


def root(a):
    detroot = determinant(a).sqrt()
    denom = (a[0][0] + a[1][1] + 2 * detroot).sqrt()
    return [[(a[i][j] + (detroot if i == j else 0)) / denom for j in range(2)] for i in range(2)]


def centered(rows):
    means = [sum(row[i] for row in rows) / len(rows) for i in range(2)]
    return [[row[i] - means[i] for i in range(2)] for row in rows]


def scatter(rows, ridge):
    return [[sum(row[i] * row[j] for row in rows) + (ridge if i == j else 0) for j in range(2)] for i in range(2)]


def dense(q, s, scales, ridge, threshold, cap):
    x = [[r[i] / scales[i] for i in range(2)] for r in centered(q)]
    y = [[r[i] * scales[i] for i in range(2)] for r in centered(s)]
    c, f = scatter(x, ridge), scatter(y, ridge)
    ch = root(c)
    g = matmul(matmul(ch, inverse(root(matmul(matmul(ch, f), ch)))), ch)
    # Independently verify the untruncated SPD equation in high precision.
    residual = matmul(matmul(g, f), g)
    assert max(abs(residual[i][j] - c[i][j]) for i in range(2) for j in range(2)) < D('1e-90') * max(abs(v) for row in c for v in row)
    gap = ((g[0][0] - g[1][1]) ** 2 + 4 * g[0][1] ** 2).sqrt()
    values = [(g[0][0] + g[1][1] + sign * gap) / 2 for sign in [-1, 1]]
    keep = sorted((i for i in range(2) if values[i] < 1 / threshold or values[i] > threshold), key=lambda i: abs(values[i].ln()), reverse=True)[:cap]
    result = [[D(i == j) for j in range(2)] for i in range(2)]
    for k in keep:
        if gap == 0:
            assert len(keep) == 2  # No ambiguous rank-capped repeated spectrum cases.
            result = g
            break
        other = values[1 - k]
        for i in range(2):
            for j in range(2):
                result[i][j] += (values[k] - 1) * (g[i][j] - (other if i == j else 0)) / (values[k] - other)
    result = [[result[i][j] * scales[i] * scales[j] for j in range(2)] for i in range(2)]
    return [result[0][0], result[0][1], result[1][1], -determinant(result).ln(), len(keep)]


def rows(values):
    return [[D(str(v)) for v in row] for row in values]


def flat(values):
    return ';'.join(str(v) for row in values for v in row)


def fixture_text(header, data):
    output = io.StringIO()
    output.write('# oracle: Python decimal, precision=120; tools/fisher-reference/generate.py\n')
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(data)
    return output.getvalue()


def generate():
    q = rows([[-2, 1], [1, 3], [2, -1], [-1, -3]])
    s = rows([[1, -2], [-3, -1], [-1, 2], [3, 1]])
    cases = []
    for name, scale, threshold, cap in [('full', ['1', '1'], '1', 2), ('scaled', ['0.125', '8'], '1', 2), ('capped', ['1', '1'], '1', 1), ('filtered', ['1', '1'], '2', 2)]:
        cases.append((name, q, s, list(map(D, scale)), D('1e-5'), D(threshold), cap))
    # Same supported geometry despite extremely large, matched raw scatters.
    for exponent in [-100, 100]:
        factor = D(10) ** exponent
        cases.append((f'common_{exponent}', [[v * factor for v in r] for r in q], [[v * factor for v in r] for r in s], [D(1), D(1)], D('1e-5') * factor ** 2, D(1), 2))
    cases.append(('rank_one', rows([[-2, -4], [0, 0], [2, 4]]), rows([[1, 2], [0, 0], [-1, -2]]), [D(1), D(1)], D('1e-5'), D('1.01'), 2))
    fits = [[name, flat(q), flat(s), *scales, ridge, threshold, cap, *dense(q, s, scales, ridge, threshold, cap)] for name, q, s, scales, ridge, threshold, cap in cases]
    trace = []
    observations = rows([[i - 4, i * i % 11] for i in range(13)])
    scores = [[-r[0] / 2 + r[1], r[0] - 2 * r[1]] for r in observations]
    for n in range(1, 14):
        start = 0 if n <= 6 else ((n - 1) // 3 - 1) * 3
        c, f = scatter(centered(observations[start:n]), D('1e-5')), scatter(centered(scores[start:n]), D('1e-5'))
        trace.append([n, *observations[n - 1], *scores[n - 1], n - start, *((c[i][i] / f[i][i]).sqrt().sqrt() for i in range(2))])
    return {
        'crates/alea-math/tests/fixtures/fisher-dense.csv': fixture_text(['case', 'q', 'score', 'scale0', 'scale1', 'ridge', 'threshold', 'cap', 'g00', 'g01', 'g11', 'logdet_mass', 'rank'], fits),
        'crates/alea-mcmc/tests/fixtures/fisher-diagonal.csv': fixture_text(['n', 'q0', 'q1', 's0', 's1', 'count', 'scale0', 'scale1'], trace),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    with localcontext() as context:
        context.prec = 120
        for path, content in generate().items():
            destination = ROOT / path
            if args.check:
                assert destination.read_text() == content, f'stale fixture: {path}'
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(content)
