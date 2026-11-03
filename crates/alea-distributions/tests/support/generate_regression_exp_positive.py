"""Independent Decimal80 reference on exact positive binary64 inputs."""
from decimal import Decimal, localcontext
from pathlib import Path
import math
import struct


def bits(x):
    return struct.unpack('>Q', struct.pack('>d', x))[0]


points = {0., 32., math.nextafter(32., math.inf), 40., 100., 700., 710.}
for i in range(129):
    points.add(32. * i / 128)
for i in range(47):
    x = i * math.log(2)
    if x <= 32.:
        points.update([x, math.nextafter(x, -math.inf), math.nextafter(x, math.inf)])
for i in range(739):
    for fraction in [0., 0.5]:
        x = (i + fraction) * math.log(2) / 16
        if x <= 32.:
            points.update([x, math.nextafter(x, -math.inf), math.nextafter(x, math.inf)])
rows = ['# input binary64 bits, nearest binary64 exp(input); Decimal80']
with localcontext() as ctx:
    ctx.prec = 80
    for x in sorted(points):
        rows.append(f'{bits(x):016x},{bits(float(Decimal.from_float(x).exp())):016x}')
Path(__file__).with_name('regression-exp-positive.csv').write_text('\n'.join(rows) + '\n')
