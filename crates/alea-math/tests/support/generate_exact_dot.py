"""Regenerate/check the independent Fraction dot-sign fixture (stdlib only)."""
from fractions import Fraction
from pathlib import Path
import random
import struct
import sys

rng = random.Random(3029)
def finite_bits():
    while True:
        bits = rng.getrandbits(64)
        if (bits >> 52) & 2047 != 2047:
            return bits

def value(bits):
    return struct.unpack(">d", struct.pack(">Q", bits))[0]

rows = []
for _ in range(200):
    n = rng.randrange(1, 9)
    left, right = [finite_bits() for _ in range(n)], [finite_bits() for _ in range(n)]
    if rng.randrange(2) == 0:
        left += [left[0] ^ (1 << 63)]
        right += [right[0]]
    total = sum((Fraction.from_float(value(x)) * Fraction.from_float(value(y))
                 for x, y in zip(left, right)), Fraction(0))
    rows.append(",".join(f"{x:016x}" for x in left) + "|" +
                ",".join(f"{x:016x}" for x in right) + "|" + str((total > 0) - (total < 0)))
expected = "# Python Fraction.from_float, random.Random(3029), finite IEEE bits\n" + "\n".join(rows) + "\n"
path = Path(__file__).with_name("exact-dot-fraction.txt")
if "--write" in sys.argv:
    path.write_text(expected)
else:
    assert path.read_text() == expected, "fixture differs from independent Fraction oracle"
