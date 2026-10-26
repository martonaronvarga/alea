#!/usr/bin/env python3
"""Independent high precision lookup values, not an interval certificate.

Run with mpmath 1.4.1. Uses direct image/eigenfunction sums, no Alea or Stan
code. Recomputes all entries at 100/160 digits and 24/48 terms, requiring
60-digit agreement before rounding to binary64 fixtures. Inputs are the exact
binary64 values parsed by Rust. No density is exponentiated in binary64.
"""
import csv
import io
import math
from pathlib import Path
import mpmath as mp

OUTPUT = Path(__file__).with_name('wiener_extremes.csv')


def density(rt, a, tau, w, v, sv, terms, lower):
    if lower:
        w, v = 1-w, -v
    t = rt - tau
    tp = t / (a * a)
    q = 1 - w
    if tp <= mp.mpf('.25'):
        raw = mp.fsum((q + 2*k) * mp.exp(-((q + 2*k)**2 - q*q) / (2*tp))
                      for k in range(-terms, terms + 1))
        series = -mp.log(2*mp.pi)/2 - mp.log(tp)*mp.mpf('1.5') - q*q/(2*tp) + mp.log(raw)
    else:
        b = mp.pi**2 * tp/2
        raw = mp.fsum(k * mp.sin(k*mp.pi*q) * mp.exp(-(k*k-1)*b)
                      for k in range(1, terms + 1))
        series = mp.log(mp.pi) - b + mp.log(raw)
    lam = 1 + sv*sv*t
    pref = -2*mp.log(a) - mp.log(lam)/2 + (-v*v*t + 2*a*v*q + a*a*q*q*sv*sv)/(2*lam)
    return pref + series


def evaluate(inputs, digits, terms, lower):
    with mp.workdps(digits):
        args = list(map(mp.mpf, inputs))
        value = density(*args, terms, lower)
        score = []
        for i in range(1, 6):
            def at(x):
                changed = args.copy()
                changed[i] = x
                return density(*changed, terms, lower)
            score.append(mp.diff(at, args[i]))
        return [value, *score]


def main():
    output = io.StringIO()
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(['rt', 'alpha', 'tau', 'beta', 'delta', 's_delta',
                     'log_prob', 'g_alpha', 'g_tau', 'g_beta', 'g_delta', 'g_s_delta', 'boundary'])
    for tp in [1e-8, 1e-4, .01, .1, .25, .3, 1., 10., 100.]:
        for w in [1e-15, 1e-12, 1e-8, .999e-5, 1.001e-5, .001, .1, .5, .9, 1-1e-8, 1-1e-12, math.nextafter(1., 0.)]:
            for v, sv in [(0., 0.), (-30., 5.)]:
                for lower in [False, True]:
                    inputs = [tp, 1., 0., w, v, sv]
                    low = evaluate(inputs, 100, 24, lower)
                    high = evaluate(inputs, 160, 48, lower)
                    with mp.workdps(160):
                        for x, y in zip(low, high):
                            if abs(x-y) > mp.mpf('1e-60') * max(1, abs(y)):
                                raise ValueError(('unresolved reference', inputs, x, y))
                    row = inputs + list(map(float, high))
                    if not all(map(math.isfinite, row)):
                        raise ValueError(('nonfinite reference', inputs))
                    writer.writerow([*map(repr, row), 'lower' if lower else 'upper'])
    OUTPUT.write_text(output.getvalue())


if __name__ == '__main__':
    main()
