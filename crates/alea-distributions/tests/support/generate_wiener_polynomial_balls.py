#!/usr/bin/env python3
"""Independent continuous Acb enclosures for moderate StandardNormalTarget moments."""
import csv,math
from pathlib import Path
from flint import arb,acb,ctx


def main():
    ctx.prec=160
    path=Path(__file__).with_name('wiener_polynomial_balls.csv')
    with path.open('w',newline='') as stream:
        out=csv.writer(stream,lineterminator='\n')
        out.writerow(['A','B']+[f'J{r}_{s}' for r in range(4) for s in ['mid','radius']])
        for av,bv in [(0.,0.),(0.,.02),(0.,1.),(0.,8.),(.0001,.001),(.1,1.),(1.,2.),(4.,0.),(4.,4.),(8.,4.),(16.,0.)]:
            for sign in [-1,1]:
                a,b=arb(av),arb(sign*bv);row=[av,sign*bv]
                for r in range(4):
                    def integrand(z,analytic):return z**r*(-a*z*z+b*z).exp()
                    value=acb.integral(integrand,arb('-.5'),arb('.5'),abs_tol=arb('1e-35'),rel_tol=arb('1e-35')).real
                    if not value.is_finite():raise ArithmeticError('moment reference did not converge')
                    m=float(value.mid());row.extend([m,math.nextafter(float(abs(value-arb(m)).upper()),math.inf)])
                out.writerow(row)
    print(path)


if __name__=='__main__':main()
