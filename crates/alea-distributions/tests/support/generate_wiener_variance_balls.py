#!/usr/bin/env python3
"""Exact-binary64 Arb references for cancellation at large drift variability."""
import csv,math
from pathlib import Path
from flint import arb
from generate_wiener_balls import evaluate


def main():
    path=Path(__file__).with_name('wiener_variance_balls.csv')
    with path.open('w',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(['rt','alpha','tau','beta','delta','sd','lower','precision','terms']+[f'{v}_{s}' for v in ['log','alpha','tau','beta','delta','sd'] for s in ['mid','radius']])
        for t in [1e-8,1e-5,.001,.03]:
            for sd in [350.,1e4,1e6]:
                for lower in [False,True]:
                    inputs=[t,1.6,0.,.3,.5,sd]
                    values,precision,k=evaluate(*inputs,lower)
                    row=inputs+[int(lower),precision,k]
                    for value in values:
                        mid=float(value.mid())
                        row += [mid,math.nextafter(float(abs(value-arb(mid)).upper()),math.inf)]
                    writer.writerow(row)

if __name__=='__main__':main()
