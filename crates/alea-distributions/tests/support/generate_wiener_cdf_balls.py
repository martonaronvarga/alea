#!/usr/bin/env python3
"""Arb boundary CDF/tail enclosures. Requires python-flint==0.9.0.
Exact binary64 inputs; integrated image terms or spectral tails plus geometric
majorants. Precision/term count increase until both log probabilities enclose
within 1e-24*(1+abs(midpoint)). No Rust kernel is called.
"""
import csv, math
from pathlib import Path
from flint import arb,ctx
from generate_wiener_balls import moment


def evaluate(t0,beta0,v0,lower):
    for precision in [128,256,512,1024]:
        ctx.prec=precision
        t,beta,v=map(arb,[t0,beta0,v0]);q=beta if lower else 1-beta
        if lower:v=-v
        w=1-q
        mass=w if v==0 else (-2*v*w).expm1()/(-2*v).expm1()
        for k in [8,16,32,64,128]:
            if t<arb('0.15'):
                f=arb(0)
                def hit(x):
                    return ((-v*x).exp()*((x-v*t)/(2*t).sqrt()).erfc()+(v*x).exp()*((x+v*t)/(2*t).sqrt()).erfc())/2
                for j in range(-k,k+1):
                    x=q+2*j
                    f+=(1 if j>=0 else -1)*hit(abs(x))
                errors=[]
                for x in [2*(k+1)+q,2*(k+1)-q]:
                    if not x>abs(v)*t:break
                    first=(-x*x/(2*t)-v*v*t/2).exp()*(t/(2*arb.pi())).sqrt()*(1/(x-v*t)+1/(x+v*t))
                    errors.append(first/(1-(-(4*x+4)/(2*t)).exp()))
                if len(errors)!=2:continue
                f=(f+arb(0,sum(errors,arb(0)).upper()))*(v*q).exp()
                h=mass-f
            else:
                c=arb.pi()**2*t/2
                h=arb(0)
                for j in range(1,k+1):
                    rate=(v*v+(arb.pi()*j)**2)/2
                    h+=arb.pi()*j*(arb.pi()*j*q).sin()*(-rate*t).exp()/rate
                error=arb.pi()*(-v*v*t/2).exp()*moment(1,arb(k+1),arb(1),c)/((v*v+(arb.pi()*(k+1))**2)/2)
                h=(h+arb(0,error.upper()))*(v*q).exp()
                f=mass-h
            if f>0 and h>0:
                out=[f.log(),h.log()]
                if all(x.is_finite() and x.rad()<arb('1e-24')*(1+abs(x.mid())) for x in out):return out,precision,k
    raise ArithmeticError((t0,beta0,v0,lower))


def main():
    path=Path(__file__).with_name('wiener_cdf_balls.csv')
    with path.open('w',newline='') as stream:
        out=csv.writer(stream,lineterminator='\n');out.writerow(['t','beta','delta','lower','precision','terms','log_cdf','cdf_radius','log_tail','tail_radius'])
        for t in [.001,.01,.1,.15,.3,1.,10.,100.]:
            for beta in [.01,.2,.5,.9,.99]:
                for drift in [-20.,-1.,0.,1.,20.]:
                    for lower in [False,True]:
                        values,p,k=evaluate(t,beta,drift,lower);row=[t,beta,drift,int(lower),p,k]
                        for x in values:
                            m=float(x.mid());r=math.nextafter(float(abs(x-arb(m)).upper()),math.inf);row.extend([m,r])
                        out.writerow(row)
    print(path)
if __name__=='__main__':main()
