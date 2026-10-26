#!/usr/bin/env python3
"""Independent Arb enclosures for Wiener5 values and scores (python-flint 0.9).

Finite image/spectral sums use ball arithmetic. Positive geometric majorants
bound every omitted density/time/start derivative term. Precision and term count
increase until all output balls meet the requested relative/absolute width.
Inputs are exact binary64 values, not rounded decimal substitutes. This oracle
certifies its own enclosures; it does not certify Rust's floating-point kernels.
No quadrature or between-trial start/time variability is covered by this oracle.
"""
import csv
from pathlib import Path
from flint import arb, ctx


def moment(p, x, step, c):
    ratio = ((x + step) / x) ** p * (-c * (2*x*step + step*step)).exp()
    if not ratio < 1:
        raise ArithmeticError("nondecreasing tail")
    return x**p * (-c*x*x).exp() / (1-ratio)


def enclosed_sum(u, q, k):
    pi = arb.pi()
    f = du = dq = arb(0)
    if u < arb('0.15'):
        c = 1/(2*u)
        norm = 1/(2*pi*u**3).sqrt()
        for j in range(-k, k+1):
            x = q+2*j
            e = (-c*x*x).exp()
            f += x*e
            du += x*e*(x*x/(2*u*u)-3/(2*u))
            dq += (1-x*x/u)*e
        tails = [sum((moment(p, 2*(k+1)+sgn*q, arb(2), c) for sgn in [-1,1]), arb(0)) for p in range(4)]
        errors = [tails[1], tails[3]/(2*u*u)+3*tails[1]/(2*u), tails[0]+tails[2]/u]
        return tuple(norm*(v+arb(0,b.upper())) for v,b in zip([f,du,dq], errors))
    c = pi*pi*u/2
    for j in range(1,k+1):
        x = pi*j*q
        e = (-c*j*j).exp()
        f += pi*j*x.sin()*e
        du -= pi**3*j**3*x.sin()*e/2
        dq += pi*pi*j*j*x.cos()*e
    errors = [pi*moment(1,arb(k+1),arb(1),c), pi**3*moment(3,arb(k+1),arb(1),c)/2, pi*pi*moment(2,arb(k+1),arb(1),c)]
    return tuple(v+arb(0,b.upper()) for v,b in zip([f,du,dq],errors))


def evaluate(rt, a, tau, beta, delta, sd, lower):
    for precision in [96,192,384,768,1536]:
        ctx.prec = precision
        a,tau,beta,delta,sd,rt = map(arb,[a,tau,beta,delta,sd,rt])
        t = rt-tau
        q = beta if lower else 1-beta
        v = -delta if lower else delta
        for k in [4,8,16,32,64]:
            try:
                f,fu,fq = enclosed_sum(t/a**2,q,k)
                if not f > 0:
                    continue
                su,sq = fu/f,fq/f
                lam = 1+sd*sd*t
                n = -v*v*t+2*a*v*q+a*a*q*q*sd*sd
                pref = -2*a.log()-lam.log()/2+n/(2*lam)
                pa = -2/a+(v*q+a*q*q*sd*sd)/lam
                pt = -sd*sd/(2*lam)+(-v*v*lam-n*sd*sd)/(2*lam*lam)
                pq = (a*v+a*a*q*sd*sd)/lam
                pv = (-v*t+a*q)/lam
                ps = -sd*t/lam+(a*a*q*q*sd*lam-n*sd*t)/(lam*lam)
                out = [pref+f.log(),pa-2*t/a**3*su,-pt-su/a**2,(1 if lower else -1)*(pq+sq),(-1 if lower else 1)*pv,ps]
                if all(x.is_finite() and x.rad() < arb('1e-24')*(1+abs(x.mid())) for x in out):
                    return out,precision,k
            except ArithmeticError:
                pass
    raise ArithmeticError("oracle budget exhausted")


def main():
    path = Path(__file__).with_name('wiener_balls.csv')
    with path.open('w',newline='') as stream:
        writer=csv.writer(stream, lineterminator="\n")
        writer.writerow(['rt','alpha','tau','beta','delta','s_delta','lower','precision','terms']+[f'{v}_{s}' for v in ['log','alpha','tau','beta','delta','sd'] for s in ['mid','radius']])
        for t in [1e-5,.001,.01,.1,.15,.3,1.,10.,100.]:
            for beta in [1e-12,1e-8,.01,.2,.5,.99,1-1e-8,1-1e-12]:
                for lower in [False,True]:
                    for sd in [0.,.7]:
                        inputs=[t,1.,0.,beta,.8,sd]
                        out,precision,k=evaluate(*inputs,lower)
                        row=inputs+[int(lower),precision,k]
                        for x in out:
                            mid=float(x.mid())
                            radius=float((abs(x-arb(mid))).upper())
                            # Outward binary64 rounding for the saved radius.
                            import math
                            row += [mid,math.nextafter(radius,math.inf)]
                        writer.writerow(row)
    print(path)

if __name__=='__main__':
    main()
