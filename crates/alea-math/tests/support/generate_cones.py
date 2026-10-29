"""Independent exact vertex-enumeration oracle; no elimination or LP package.

Intersect each homogeneous cone with [-1,1]^p. Its compact polytope has a
strict improving direction iff the linear sum-of-margins objective is positive
at a vertex. Enumerate active sets and solve with Fraction Gaussian elimination.
"""
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
from random import Random
import sys


def solve(rows, rhs):
    p = len(rhs)
    a = [[F(x) for x in row] + [F(b)] for row, b in zip(rows, rhs)]
    for j in range(p):
        pivot = next((k for k in range(j, p) if a[k][j]), None)
        if pivot is None:
            return None
        a[j], a[pivot] = a[pivot], a[j]
        scale = a[j][j]
        a[j] = [v / scale for v in a[j]]
        for k in range(p):
            if k != j:
                scale = a[k][j]
                a[k] = [v - scale * w for v, w in zip(a[k], a[j])]
    return [row[-1] for row in a]


def improving(p, inequalities, equalities):
    a = list(inequalities) + list(equalities) + [tuple(-x for x in r) for r in equalities]
    rhs = [0] * len(a)
    for j in range(p):
        for sign in (-1, 1):
            a.append(tuple(sign if k == j else 0 for k in range(p)))
            rhs.append(-1)
    for active in combinations(range(len(a)), p):
        point = solve([a[k] for k in active], [rhs[k] for k in active])
        if point is None:
            continue
        margins = [sum(x * v for x, v in zip(row, point)) for row in a]
        if all(m >= b for m, b in zip(margins, rhs)) and sum(margins[:len(inequalities)]) > 0:
            return True
    return False


def generate():
    rng = Random(20261029)
    lines = ["# p|inequalities (row major)|equalities (row major)|strict direction exists"]
    for p in (1, 2, 3):
        for _ in range(60):
            a = [tuple(rng.randrange(-3, 4) for _ in range(p)) for _ in range(rng.randrange(1, 6))]
            e = [tuple(rng.randrange(-3, 4) for _ in range(p)) for _ in range(rng.randrange(3))]
            encode = lambda rows: ",".join(str(x) for row in rows for x in row)
            lines.append(f"{p}|{encode(a)}|{encode(e)}|{int(improving(p, a, e))}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    target = Path(__file__).with_name("cones-fraction.txt")
    text = generate()
    if "--check" in sys.argv:
        assert target.read_text() == text
    else:
        target.write_text(text)
