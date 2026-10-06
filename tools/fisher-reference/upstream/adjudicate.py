"""80-digit dense oracle for conditioning-sensitive upstream fit comparisons.

Uses only emitted input observations/options/scales, never upstream fit outputs.
Denman-Beavers square root of C F and Jacobi rotations are independent of Faer.
Standard library only; --check requires byte-identical regenerated fixtures.
"""
import argparse
import csv
from decimal import Decimal as D, localcontext
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / 'crates/alea-math/tests/fixtures/fisher-nuts-rs.csv'
OUTPUT = ROOT / 'crates/alea-math/tests/fixtures/fisher-nuts-rs-decimal.csv'


def multiply(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def identity(n):
    return [[D(i == j) for j in range(n)] for i in range(n)]


def inverse(matrix):
    n = len(matrix)
    a = [row[:] + eye for row, eye in zip(matrix, identity(n))]
    for j in range(n):
        pivot = max(range(j, n), key=lambda i: abs(a[i][j]))
        a[j], a[pivot] = a[pivot], a[j]
        divisor = a[j][j]
        a[j] = [v / divisor for v in a[j]]
        for i in range(n):
            if i != j:
                factor = a[i][j]
                a[i] = [x - factor * y for x, y in zip(a[i], a[j])]
    return [row[n:] for row in a]


def square_root(matrix):
    n = len(matrix)
    scale = max(sum(abs(v) for v in row) for row in matrix)
    a = [[v / scale for v in row] for row in matrix]
    y, z = [row[:] for row in a], identity(n)
    for _ in range(200):
        zi, yi = inverse(z), inverse(y)
        yn = [[(y[i][j] + zi[i][j]) / 2 for j in range(n)] for i in range(n)]
        z = [[(z[i][j] + yi[i][j]) / 2 for j in range(n)] for i in range(n)]
        error = max(abs(yn[i][j] - y[i][j]) for i in range(n) for j in range(n))
        y = yn
        if error < D('1e-55'):
            break
    else:
        raise ArithmeticError('matrix square root did not converge')
    squared = multiply(y, y)
    assert max(abs(squared[i][j] - a[i][j]) for i in range(n) for j in range(n)) < D('1e-50')
    return [[v * scale.sqrt() for v in row] for row in y]


def eigen(matrix):
    n = len(matrix)
    a, u = [row[:] for row in matrix], identity(n)
    for _ in range(100):
        for p in range(n):
            for q in range(p + 1, n):
                if abs(a[p][q]) < D('1e-60'):
                    continue
                tau = (a[q][q] - a[p][p]) / (2 * a[p][q])
                t = (D(1) if tau >= 0 else D(-1)) / (abs(tau) + (1 + tau * tau).sqrt())
                c = 1 / (1 + t * t).sqrt()
                s = t * c
                a[p][p], a[q][q] = a[p][p] - t * a[p][q], a[q][q] + t * a[p][q]
                a[p][q] = a[q][p] = D(0)
                for k in range(n):
                    if k != p and k != q:
                        ap, aq = a[k][p], a[k][q]
                        a[k][p] = a[p][k] = c * ap - s * aq
                        a[k][q] = a[q][k] = s * ap + c * aq
                    up, uq = u[k][p], u[k][q]
                    u[k][p], u[k][q] = c * up - s * uq, s * up + c * uq
        if max(abs(a[i][j]) for i in range(n) for j in range(i)) < D('1e-55'):
            break
    else:
        raise ArithmeticError('Jacobi iteration did not converge')
    values = [a[i][i] for i in range(n)]
    gram = multiply(list(zip(*u)), u)
    assert max(abs(gram[i][j] - D(i == j)) for i in range(n) for j in range(n)) < D('1e-50')
    reconstructed = multiply([[u[i][j] * values[j] for j in range(n)] for i in range(n)], list(zip(*u)))
    assert max(abs(reconstructed[i][j] - matrix[i][j]) for i in range(n) for j in range(n)) < D('1e-45')
    return values, u


def solve(row):
    n, d = int(row['n']), int(row['dim'])
    scales = list(map(D, row['scales'].split(';')))
    ridge, threshold = D(row['ridge']), D(row['threshold'])
    matrices = []
    for key in ['q', 'score']:
        values = list(map(D, row[key].split(';')))
        means = [sum(values[t * d + i] for t in range(n)) / n for i in range(d)]
        data = [[(values[t * d + i] - means[i]) * (1 / scales[i] if key == 'q' else scales[i]) for t in range(n)] for i in range(d)]
        matrices.append([[sum(x * y for x, y in zip(data[i], data[j])) + ridge * D(i == j) for j in range(d)] for i in range(d)])
    c, f = matrices
    g = multiply(square_root(multiply(c, f)), inverse(f))
    residual = multiply(multiply(g, f), g)
    assert max(abs(residual[i][j] - c[i][j]) for i in range(d) for j in range(d)) < D('1e-45')
    assert max(abs(g[i][j] - g[j][i]) for i in range(d) for j in range(d)) < D('1e-45')
    g = [[(g[i][j] + g[j][i]) / 2 for j in range(d)] for i in range(d)]
    values, u = eigen(g)
    assert min(values) > 0
    keep = [k for k, v in enumerate(values) if v < 1 / threshold or v > threshold]
    output = [(D(i == j) + sum((values[k] - 1) * u[i][k] * u[j][k] for k in keep)) * scales[i] * scales[j] for i in range(d) for j in range(d)]
    logdet = -2 * sum(v.ln() for v in scales) - sum(values[k].ln() for k in keep)
    return [row['case'], len(keep), ';'.join(format(v, '.50e') for v in output), format(logdet, '.50e')]


def generate():
    rows = csv.DictReader(line for line in INPUT.read_text().splitlines() if not line.startswith('#'))
    out = io.StringIO()
    out.write('# oracle: Python Decimal precision=80; Denman-Beavers + Jacobi\n')
    writer = csv.writer(out, lineterminator='\n')
    writer.writerow(['case', 'rank', 'inverse_mass', 'logdet_mass'])
    with localcontext() as context:
        context.prec = 80
        writer.writerows(solve(row) for row in rows)
    return out.getvalue()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = generate()
    if args.check:
        assert OUTPUT.read_text() == result, 'stale dense Decimal fixture'
    else:
        OUTPUT.write_text(result)
