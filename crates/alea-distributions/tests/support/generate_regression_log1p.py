"""Independent exact-f64 -> Decimal log1p fixtures for the bounded SIMD kernel."""
import argparse
from decimal import Decimal, localcontext
import math
from pathlib import Path
import struct


def bits(x):
    return struct.unpack('>Q', struct.pack('>d', x))[0]


def fixture():
    points = {0., math.ulp(0.), math.ldexp(1., -1022), 1.}
    points.update(math.ldexp(1., -i) for i in range(0, 1075, 17))
    for i in range(1, 129):
        x = i / 128
        points.update((x, math.nextafter(x, 0.), math.nextafter(x, 1.)))
    state = 17
    for _ in range(256):
        state = (state * 6364136223846793005 + 1) % 2**64
        points.add((state >> 11) / 2**53)
    lines = ['# input-f64-bits, correctly-rounded Decimal(precision=400) log1p bits']
    with localcontext() as context:
        context.prec = 400
        for x in sorted(points):
            value = float((Decimal(1) + Decimal.from_float(x)).ln())
            lines.append(f'{bits(x):016x},{bits(value):016x}')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    path = Path(__file__).with_name('regression-log1p.csv')
    expected = fixture()
    if args.check:
        if path.read_text() != expected:
            raise SystemExit('fixture differs')
    else:
        path.write_text(expected)
