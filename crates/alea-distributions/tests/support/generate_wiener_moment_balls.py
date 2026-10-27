#!/usr/bin/env python3
"""Arb/Acb continuous references for endpoint-scaled Gaussian moments."""
import csv,math
from pathlib import Path
from flint import arb,acb,ctx

def main():
    ctx.prec=160
    path=Path(__file__).with_name('wiener_moment_balls.csv')
    with path.open('w',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(['A','B']+[f'J{r}_{s}' for r in range(4) for s in ['mid','radius']])
        for av,bv in [(0.,50.),(8.,80.),(1e3,3e3),(1e6,3e6),(1e12,3e12),(1e6,1e8)]:
            for sign in [-1,1]:
                a,b=arb(av),arb(sign*bv);l=abs(b)-a;delta=a/l**2;end=min(l,arb(100));row=[av,sign*bv]
                for r in range(4):
                    def integrand(u,analytic):return (sign*(arb('.5')-u/l))**r*(-u-delta*u*u).exp()/l
                    value=acb.integral(integrand,0,end,abs_tol=arb('1e-40'),rel_tol=arb('1e-35'),eval_limit=4000).real
                    if l>end:
                        # Bound the remaining positive real tail by exp(-u).
                        bound=(-end).exp()/l*sum(arb(math.comb(r,m))*(arb('.5')+end/l)**(r-m)*math.factorial(m)/l**m for m in range(r+1))
                        value+=arb(0,bound.upper())
                    if not value.is_finite():raise ArithmeticError('moment integral failed')
                    mid=float(value.mid());row.extend([mid,math.nextafter(float(abs(value-arb(mid)).upper()),math.inf)])
                writer.writerow(row)
if __name__=='__main__':main()
