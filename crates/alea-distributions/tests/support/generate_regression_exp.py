"""Independent Decimal80 oracle on exact binary64 inputs, not decimal surrogates."""
from decimal import Decimal, localcontext
from pathlib import Path
import math
import struct


def bits(x):
    return struct.unpack('>Q', struct.pack('>d', x))[0]


points = {0., -32., -32.00000000000001, -40., -100., -700., -740., -745., -746.}
for i in range(129):
    points.add(-32. * i / 128)
for i in range(47):
    x = -i * math.log(2)
    if x >= -32.:
        points.update([x, math.nextafter(x, -math.inf), math.nextafter(x, math.inf)])
# Table bins and nearest-integer switching surfaces, on exact f64 inputs.
for i in range(739):
    for fraction in [0., 0.5]:
        x = -(i + fraction) * math.log(2) / 16
        if x >= -32.:
            points.update([x, math.nextafter(x, -math.inf), math.nextafter(x, math.inf)])
rows = ['# input binary64 bits, nearest binary64 exp(input); Decimal80']
with localcontext() as ctx:
    ctx.prec = 80
    for x in sorted(points):
        expected = float(Decimal.from_float(x).exp())
        rows.append(f'{bits(x):016x},{bits(expected):016x}')
Path(__file__).with_name('regression-exp.csv').write_text('\n'.join(rows) + '\n')
