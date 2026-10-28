#!/usr/bin/env python3
"""Independent integrated-image enclosures beyond normalized time two."""
import csv, math
from pathlib import Path
from flint import arb
from wiener5_cdf_reference import evaluate


def main():
    path = Path(__file__).with_name('wiener7_long_cdf_balls.csv')
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['rt','a','tau','beta','v','sd','sw','st','lower'] +
                        [f'{v}_{s}' for v in ['log','a','tau','beta','v','sd','sw','st']
                         for s in ['mid','radius']])
        for t in [.1, .5, 1.]:
            for beta, drift, sd, sw in [(.999,-3.,1e-8,0.),(.999,30.,.01,0.),
                                        (.999,1.,1e-8,.001),(.5,-3.,5.,.1)]:
                for lower in [False,True]:
                    params = [t,.1,0.,beta,drift,sd,sw,2*t,lower]
                    values, info = evaluate(params, goal='1e-16')
                    row = params[:8] + [int(lower)]
                    for value in values:
                        midpoint = float(value.mid())
                        row.extend([midpoint, math.nextafter(float(abs(value-arb(midpoint)).upper()),math.inf)])
                    writer.writerow(row)
                    stream.flush()
                    print(params, info, flush=True)


if __name__ == '__main__':
    main()
