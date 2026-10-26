#!/usr/bin/env python3
"""Independent CDF-score enclosures via continuous density/derivative integrals."""
import csv,math
from pathlib import Path
from flint import arb
from wiener7_reference import evaluate

def main():
    p=Path(__file__).with_name('wiener_cdf_gradient_balls.csv')
    with p.open('w',newline='') as stream:
        out=csv.writer(stream,lineterminator='\n');out.writerow(['t','beta','delta','lower','precision']+[f'{v}_{s}' for v in ['log','alpha','tau','beta','delta'] for s in ['mid','radius']])
        for t in [.01,.15,.5]:
            for beta in [1e-8,.5]:
                for v in [-5.,5.]:
                    for lower in [False,True]:
                        values,info=evaluate([t,1.,0.,beta,v,0.,0.,t,lower],goal='1e-15',max_precision=384)
                        values[0]+=arb(t).log();row=[t,beta,v,int(lower),info['precision']]
                        for x in values[:5]:
                            m=float(x.mid());r=math.nextafter(float(abs(x-arb(m)).upper()),math.inf);row.extend([m,r])
                        out.writerow(row);stream.flush()
    print(p)
if __name__=='__main__':main()
