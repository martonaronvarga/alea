#!/usr/bin/env python3
"""Independent continuous-start spectral Wiener7 ball references."""
import csv,math
from pathlib import Path
from flint import arb,acb,ctx
from generate_wiener_balls import moment


def evaluate(params):
    ctx.prec=128
    t,beta,v,s,h=map(arb,params[:5]);lower=bool(params[5])
    q=beta if lower else 1-beta
    if lower:v=-v
    lam=1+s*s*t;pi=arb.pi();c=pi*pi*t/2;k=16
    g=lambda x:(-v*v*t+2*v*x+x*x*s*s)/(2*lam)
    log_scale=pi.log()-lam.log()/2+g(q)-c
    weights=[(-c*(n*n-1)).exp() for n in range(1,k+1)]
    factor=(max(g(q-h/2),g(q+h/2))-g(q)+c).exp()
    t1,t2,t3=[factor*moment(p,arb(k+1),arb(1),c) for p in [1,2,3]]
    qmax=q+h/2
    beta_error=(abs(v)+qmax*s*s)/lam*t1+pi*t2
    tau_error=(s*s/lam+(abs(v)+qmax*s*s)**2/lam**2)*t1/2+pi*pi*t3/2
    errors=[t1,(2+(abs(v)*qmax+qmax*qmax*s*s)/lam)*t1+pi*pi*t*t3,
            tau_error,beta_error,(qmax+abs(v)*t)/lam*t1,
            s*((abs(v)*t+qmax)**2/lam**2+t/lam)*t1,beta_error/2,tau_error/2]
    values=[]
    for component in range(8):
        def integrand(z,analytic):
            x=q+h*z;f=acb(0);du=acb(0);dq=acb(0)
            for n,w in enumerate(weights,1):
                angle=pi*n*x;sin=angle.sin();cos=angle.cos()
                f+=n*w*sin;du-=pi*pi*n**3*w*sin/2;dq+=pi*n*n*w*cos
            pa=-2+(v*x+x*x*s*s)/lam
            pt=-s*s/(2*lam)-(v+x*s*s)**2/(2*lam*lam)
            pq=(v+x*s*s)/lam
            if component==0:value=f
            elif component==1:value=pa*f-2*t*du
            elif component in (2,7):value=-pt*f-du
            elif component==3:value=(1 if lower else -1)*(pq*f+dq)
            elif component==4:value=(-1 if lower else 1)*(x-v*t)/lam*f
            elif component==5:value=s*((v*t-x)**2/lam**2-t/lam)*f
            else:value=z*(pq*f+dq)
            if component==7:value/=2
            return (g(x)-g(q)).exp()*value
        value=acb.integral(integrand,arb('-.5'),arb('.5'),abs_tol=arb('1e-28'),rel_tol=arb('1e-28'),eval_limit=4000)
        values.append(value.real+arb(0,errors[component].upper()))
    if not values[0]>0:raise ArithmeticError('nonpositive reference density')
    result=[values[0].log()+log_scale]+[x/values[0] for x in values[1:]]
    if not all(x.is_finite() and x.rad()<arb('1e-22')*(1+abs(x.mid())) for x in result):raise ArithmeticError('reference enclosure too wide')
    return result


def main():
    path=Path(__file__).with_name('wiener7_spectral_balls.csv')
    with path.open('w',newline='') as stream:
        out=csv.writer(stream,lineterminator='\n')
        out.writerow(['t','beta','v','sd','sw','lower']+[f'{v}_{s}' for v in ['log','a','tau','beta','v','sd','sw','st'] for s in ['mid','radius']])
        for t in [.15,.5,2.,10.]:
            for v in [-3.,0.,3.]:
                for sd in [.5,2.]:
                    for beta in [.2,.5,.8]:
                        for width in [.1,.9 if beta==.5 else .35]:
                            for lower in [False,True]:
                                params=[t,beta,v,sd,width,lower];row=params[:5]+[int(lower)]
                                for x in evaluate(params):
                                    m=float(x.mid());row.extend([m,math.nextafter(float(abs(x-arb(m)).upper()),math.inf)])
                                out.writerow(row);stream.flush()
    print(path)


if __name__=='__main__':main()
