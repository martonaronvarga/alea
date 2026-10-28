#!/usr/bin/env python3
"""Exact time marginalization with Gaussian drift variability: Arb fixtures."""
import csv,math
from pathlib import Path
from flint import arb
from wiener5_cdf_reference import evaluate


def main():
    path=Path(__file__).with_name('wiener7_drift_cdf_balls.csv')
    with path.open('w',newline='') as stream:
        out=csv.writer(stream,lineterminator='\n')
        out.writerow(['rt','a','tau','beta','v','sd','st','lower']+[f'{v}_{s}' for v in ['log','a','tau','beta','v','sd','sw','st'] for s in ['mid','radius']])
        for t in [.003,.03,.3]:
            for v in [-3.,3.]:
                for sd in [.5,350.,1e12]:
                    for width in [.4,1.5]:
                        for lower in [False,True]:
                            params=[t,1.,0.,.3,v,sd,0.,width*t,lower]
                            values,info=evaluate(params,goal='1e-16')
                            row=params[:6]+[params[7],int(lower)]
                            for x in values:
                                m=float(x.mid());row.extend([m,math.nextafter(float(abs(x-arb(m)).upper()),math.inf)])
                            out.writerow(row);stream.flush()
    print(path)


if __name__=='__main__':main()
