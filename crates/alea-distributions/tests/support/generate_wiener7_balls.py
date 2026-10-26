#!/usr/bin/env python3
"""Rigorous Arb/Acb Wiener7 start-width fixtures (python-flint==0.9.0).
Integrates finite Wiener5 image densities and derivatives with acb.integral.
Adds uniform analytic tail envelopes over the entire starting-point interval.
Time width is zero; this oracle does not cover clipped or nested time cubature.
"""
import csv,math
from pathlib import Path
from flint import arb,acb,ctx
from generate_wiener_balls import moment


def evaluate(t0,beta0,delta0,sd0,width0,lower):
    for precision in [96,192,384,768]:
        ctx.prec=precision
        t,beta,delta,sd,h=map(arb,[t0,beta0,delta0,sd0,width0])
        qmid=beta if lower else 1-beta;v=-delta if lower else delta
        qsign=1 if lower else -1;vsign=-1 if lower else 1
        lam=1+sd*sd*t
        pref=1/(2*arb.pi()*t**3*lam).sqrt()
        n=lambda q:-v*v*t+2*v*q+q*q*sd*sd
        k=4
        def components(x):
            q=qmid+h*x
            raw=[acb(0) for _ in range(8)]
            nn=n(q)
            for image in range(-k,k+1):
                z=q+2*image
                term=(-z*z/(2*t)+nn/(2*lam)).exp()*pref
                density=z*term
                alpha=term*z*(1-z*z/t+(v*q+q*q*sd*sd)/lam)
                tau=-term*z*(-3/(2*t)-sd*sd/(2*lam)+z*z/(2*t*t)+(-v*v*lam-nn*sd*sd)/(2*lam*lam))
                dq=term*(1+z*(-z/t+(v+q*sd*sd)/lam))
                drift=density*(-v*t+q)/lam
                sigma=density*(-sd*t/lam+q*q*sd/lam-nn*sd*t/(lam*lam))
                values=[density,alpha,tau,qsign*dq,vsign*drift,sigma,x*dq,tau/2]
                raw=[a+b for a,b in zip(raw,values)]
            return raw
        vals=[acb.integral(lambda x,analytic:components(x)[i],arb('-0.5'),arb('0.5'),rel_tol=arb(2)**(-precision+16),abs_tol=arb(2)**(-precision+16),eval_limit=20000).real for i in range(8)]
        qlo,qhi=qmid-h/2,qmid+h/2
        tails=[moment(p,2*(k+1)+qlo,arb(2),1/(2*t))+moment(p,2*(k+1)-qhi,arb(2),1/(2*t)) for p in range(4)]
        nmax=max(n(qlo).upper(),n(qhi).upper())
        scale=pref*(nmax/(2*lam)).exp()
        density=scale*tails[1]
        time=scale*(tails[3]/(2*t*t)+3*tails[1]/(2*t))
        start=scale*(tails[0]+tails[2]/t)
        nb=v*v*t+2*abs(v)*qhi+qhi*qhi*sd*sd
        pa=2+(abs(v)*qhi+qhi*qhi*sd*sd)/lam
        pt=sd*sd/(2*lam)+v*v/(2*lam)+nb*sd*sd/(2*lam*lam)
        pq=(abs(v)+qhi*sd*sd)/lam
        pv=(abs(v)*t+qhi)/lam
        ps=sd*t/lam+qhi*qhi*sd/lam+nb*sd*t/(lam*lam)
        beta_error=pq*density+start
        tau_error=pt*density+time
        errors=[density,pa*density+2*t*time,tau_error,beta_error,pv*density,ps*density,beta_error/2,tau_error/2]
        vals=[value+arb(0,error.upper()) for value,error in zip(vals,errors)]
        if vals[0]>0:
            out=[vals[0].log()]+[value/vals[0] for value in vals[1:]]
            if all(x.is_finite() and x.rad()<arb('1e-22')*(1+abs(x.mid())) for x in out):return out,precision,k
    raise ArithmeticError((t0,beta0,delta0,sd0,width0,lower))


def main():
    path=Path(__file__).with_name('wiener7_balls.csv')
    with path.open('w',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n');writer.writerow(['t','beta','delta','sd','width','lower','precision','terms']+[f'{v}_{s}' for v in ['log','alpha','tau','beta','delta','sd','width','time_width'] for s in ['mid','radius']])
        for t in [.01,.08]:
            for delta in [-2.,2.]:
                for sd in [.5,3.]:
                    for width in [1e-10,.1,.4]:
                        for lower in [False,True]:
                            values,p,k=evaluate(t,.5,delta,sd,width,lower);row=[t,.5,delta,sd,width,int(lower),p,k]
                            for x in values:
                                m=float(x.mid());r=math.nextafter(float(abs(x-arb(m)).upper()),math.inf);row.extend([m,r])
                            writer.writerow(row);stream.flush()
    print(path)
if __name__=='__main__':main()
