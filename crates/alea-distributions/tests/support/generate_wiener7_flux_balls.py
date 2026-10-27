#!/usr/bin/env python3
"""Independent clipped Wiener7 references from integrated spectral CDFs.

No production code or time quadrature is used. Acb integrates the start
coordinate; positive Gaussian moment majorants enclose every spectral tail.
Requires python-flint 0.9.0. Inputs are exact binary64 values.
"""
import csv,math
from pathlib import Path
from flint import arb,acb,ctx
from generate_wiener_balls import moment


def evaluate(params):
    ctx.prec=160
    rt,a,tau,beta,v,h,st,lower=params
    rt,a,tau,beta,v,h,st=map(arb,[rt,a,tau,beta,v,h,st])
    t=rt-tau
    assert 0<t<=st and h>0
    if lower:v=-v
    b=a*v;pi=arb.pi();k=16
    rate_min=(v*v+(pi*(k+1)/a)**2)/2
    qmax=(beta if lower else 1-beta)+h/2
    pref=(b*qmax-v*v*t/2).exp()*pi/(a*a)
    tails=[moment(p,arb(k+1),arb(1),pi*pi*t/(2*a*a)) for p in [1,2,3]]
    tail=pref*tails[0]/rate_min
    errors=[tail,(2/a+abs(v)*qmax)*tail+pref*pi*pi/a**3*(t+1/rate_min)*tails[2]/rate_min,
            pref*tails[0],abs(b)*tail+pref*pi*tails[1]/rate_min,
            (a*qmax+abs(v)*(t+1/rate_min))*tail,arb(0),arb(0)]
    errors[6]=errors[3]/2
    def component(z,analytic,i):
        beta_trial=beta+h*z;q=beta_trial if lower else 1-beta_trial
        if b==0:mass=1-q;mass_b=q*(1-q);mass_beta=acb(1)
        else:
            e=(-2*b).exp();eq=(-2*b*(1-q)).exp();den=1-e
            mass=(1-eq)/den
            mass_b=(2*(1-q)*eq*den-2*e*(1-eq))/(den*den)
            mass_beta=2*b*eq/den
        if i==0:value=mass
        elif i==1:value=v*mass_b
        elif i==2:value=acb(0)
        elif i in (3,6):value=mass_beta*(1 if not lower else -1)
        elif i==4:value=a*mass_b
        else:return acb(0)
        for n in range(1,k+1):
            omega=pi*n;rate=(v*v+(omega/a)**2)/2
            weight=pi*n/(a*a)*(-rate*t+b*q).exp()
            sine=(omega*q).sin();term=weight*sine/rate
            if i==0:value-=term
            elif i==1:value-=term*(-2/a+v*q+omega*omega/a**3*(t+1/rate))
            elif i==2:value-=weight*sine
            elif i in (3,6):value-=(1 if lower else -1)*weight*(b*sine+omega*(omega*q).cos())/rate
            elif i==4:value-=term*(a*q-v*(t+1/rate))
        if i==4 and lower:value=-value
        if i==6:value*=z
        return value
    values=[]
    for i in range(7):
        value=acb.integral(lambda z,analytic:component(z,analytic,i),arb('-.5'),arb('.5'),abs_tol=arb('1e-30'),rel_tol=arb('1e-30'))
        values.append(value.real+arb(0,errors[i].upper()))
    result=[values[0].log()-st.log()]+[x/values[0] for x in values[1:]]+[-1/st]
    assert all(x.is_finite() and x.rad()<arb('1e-20')*(1+abs(x.mid())) for x in result)
    return result


def main():
    path=Path(__file__).with_name('wiener7_flux_balls.csv')
    with path.open('w',newline='') as stream:
        out=csv.writer(stream,lineterminator='\n')
        out.writerow(['rt','a','tau','beta','v','sw','st','lower']+[f'{v}_{s}' for v in ['log','a','tau','beta','v','sd','sw','st'] for s in ['mid','radius']])
        for gap,a,tau,v,st in [(.01,.1,0.,30.,.2),(.1,.3,.05,0.,.2),(.01,.1,1e-6,1.,1.)]:
            for lower in [False,True]:
                params=[tau+gap,a,tau,1-.999 if lower else .999,-v if lower else v,.001,st,lower]
                row=params[:-1]+[int(lower)]
                for x in evaluate(params):
                    m=float(x.mid());row.extend([m,math.nextafter(float(abs(x-arb(m)).upper()),math.inf)])
                out.writerow(row)
    print(path)


if __name__=='__main__':main()
