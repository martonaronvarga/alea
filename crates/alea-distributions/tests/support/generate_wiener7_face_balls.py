#!/usr/bin/env python3
"""Independent narrow-time-width references using continuous Acb quadrature."""
import csv,math
from pathlib import Path
from flint import arb,ctx
from wiener7_reference import evaluate
from generate_wiener_balls import moment


def large_drift():
    """Independent spectral interval integral, avoiding saturated CDF subtraction."""
    ctx.prec=160
    params=[10.,1.,0.,.1,-30.,0.,0.,5.,False]
    t,a,tau,beta,v,sd,sw,h=map(arb,params[:8]);q=1-beta;lo=t-h
    pi=arb.pi();k=8;c=pi*pi*lo/(2*a*a);r1=(v*v+(pi/a)**2)/2
    log_scale=pi.log()-2*a.log()+a*v*q-r1*lo
    values=[arb(0)]*8
    for n in range(1,k+1):
        omega=pi*n;rate=(v*v+(omega/a)**2)/2
        interval=-(-rate*h).expm1()
        weight=n*(-(rate-r1)*lo).exp()*interval/(rate*h)
        sine=(omega*q).sin();cosine=(omega*q).cos()
        derivative=lo+1/rate-h/(rate*h).expm1()
        row=[sine,(-2/a+v*q+omega*omega/a**3*derivative)*sine,rate*sine,
             -(a*v*sine+omega*cosine),(a*q-v*derivative)*sine,arb(0),arb(0),
             (rate/interval-1/h)*sine]
        values=[x+weight*y for x,y in zip(values,row)]
    m1,m2,m3=[c.exp()*moment(p,arb(k+1),arb(1),c) for p in [1,2,3]]
    rmin=(v*v+(pi*(k+1)/a)**2)/2
    f=m1/(rmin*h)
    errors=[f,(2/a+abs(v)*q)*f+pi*pi/a**3*(lo+1/rmin)*m3/(rmin*h),m1/h,
            (a*abs(v)*m1+pi*m2)/(rmin*h),(a*q+abs(v)*(lo+1/rmin))*f,
            arb(0),arb(0),m1/h+f/h]
    values=[x+arb(0,e.upper()) for x,e in zip(values,errors)]
    result=[values[0].log()+log_scale]+[x/values[0] for x in values[1:]]
    return params,result


def main():
    path=Path(__file__).with_name('wiener7_face_balls.csv')
    with path.open('w',newline='') as stream:
        out=csv.writer(stream,lineterminator='\n')
        out.writerow(['rt','a','tau','beta','v','sd','sw','st','lower']+[f'{v}_{s}' for v in ['log','a','tau','beta','v','sd','sw','st'] for s in ['mid','radius']])
        for width in [.01,.001,1e-8]:
            for drift in [0.,3.]:
                for lower in [False,True]:
                    params=[50.,8.,0.,.5,drift,.5,.1,width,lower]
                    values,info=evaluate(params,goal='1e-13',max_precision=192)
                    row=params[:8]+[int(lower)]
                    for x in values:
                        m=float(x.mid());row.extend([m,math.nextafter(float(abs(x-arb(m)).upper()),math.inf)])
                    out.writerow(row);stream.flush()
        params,values=large_drift();row=params[:8]+[int(params[8])]
        for x in values:
            m=float(x.mid());row.extend([m,math.nextafter(float(abs(x-arb(m)).upper()),math.inf)])
        out.writerow(row)
    print(path)


if __name__=='__main__':main()
