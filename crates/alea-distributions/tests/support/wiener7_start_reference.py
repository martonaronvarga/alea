"""Rigorous finite-image start averages for the optional Wiener7 ball oracle.

Complex StandardNormalTarget moments use bounded polynomial expansions or erf/Tricomi U.
The outer time integral remains independent Acb quadrature. No f64 library
error model is assumed. This is an optional reference path, not production Rust.
"""
import math
from flint import arb,acb,acb_series,ctx

COEFFICIENTS=[[1],[arb('.5'),-1],[arb('.25'),-1,1],[arb('.125'),arb('-.75'),arb('1.5'),-1]]


def widen(value,error):
    return value+acb(arb(0,error.upper()),arb(0,error.upper()))


def moments(t,a,v,s,m,h,j,analytic):
    lam=1+s*s*t
    aa=a*a/(2*t*lam)
    az=aa*h*h
    bz=(a*v/lam-2*a*a*j/t-2*aa*m)*h
    def phase(q):return -2*a*a*j*(j+q)/t-(a*q-v*t)**2/(2*t*lam)
    radius=(abs(az)/4+abs(bz)/2).upper()
    if radius<=4:
        # Cauchy's coefficient bound on |z|=1 controls the omitted power
        # series uniformly on |z|<=1/2. FLINT computes the polynomial in C,
        # avoiding nested Python polynomial products at every quadrature node.
        n=ctx.prec+48
        previous_cap=ctx.cap
        try:
            ctx.cap=n
            poly=acb_series([0,bz,-az],prec=n).exp()
        finally:ctx.cap=previous_cap
        out=[sum((poly[q]/(2**(q+r)*(q+r+1)) for q in range(n) if (q+r)%2==0),acb(0)) for r in range(4)]
        error=(abs(az)+abs(bz)).exp()*arb(2)**(1-n)
        scale=phase(m).exp()
        return [scale*widen(x,error/(2**r*(r+1))) for r,x in enumerate(out)]
    sign=1 if bz.real>=0 else -1
    l=sign*bz-az
    lr=l.real.lower()
    if az.real>=0 and lr>0 and abs(az)/lr**2<=arb('.0025'):
        delta=abs(az)/lr**2
        n=8 if delta<arb('1e-5') else 16 if delta<arb('1e-4') else 32
        if ctx.prec>=192:n*=2
        # Exact finite exponential moments avoid an unbounded tail approximation.
        linear=[(-l).hypgeom_1f1(q+1,q+2)/(q+1) for q in range(2*n+4)]
        y=[acb(0) for _ in range(4)];coefficient=acb(1)
        for k in range(n):
            for q in range(4):y[q]+=coefficient*linear[2*k+q]
            coefficient*=-az/(k+1)
        errors=[abs(az)**n*math.factorial(2*n+q)/(math.factorial(n)*lr**(2*n+q+1)) for q in range(4)]
        scale=phase(m+sign*h/2).exp()
        return [scale*widen(sign**r*sum((c*y[q] for q,c in enumerate(COEFFICIENTS[r])),acb(0)),sum(abs(c)*errors[q] for q,c in enumerate(COEFFICIENTS[r]))) for r in range(4)]
    root=az.sqrt(analytic=analytic)
    center=bz/(2*az)
    lo=root*(-arb('.5')-center);hi=root*(arb('.5')-center)
    left=phase(m-h/2).exp();right=phase(m+h/2).exp()
    def erfcx(z):return (z*z).hypgeom_u(arb('.5'),arb('.5'))/arb.pi().sqrt()
    factor=arb.pi().sqrt()/(2*root)
    if lo.real>0:zero=factor*(left*erfcx(lo)-right*erfcx(hi))
    elif hi.real<0:zero=factor*(right*erfcx(-hi)-left*erfcx(-lo))
    else:
        constant=-2*a*v*j+2*a*a*j*j*s*s
        zero=factor*constant.exp()*(hi.erf()-lo.erf())
    out=[zero]
    for r in range(3):
        boundary=arb('.5')**r*right-(-arb('.5'))**r*left
        out.append((bz*out[r]+(r*out[r-1] if r else 0)-boundary)/(2*az))
    return out


def start_point(t,m,h,a,v,s,lower,k,analytic,component,z=0):
    lam=1+s*s*t
    pref=a/(2*arb.pi()).sqrt()/t/t.sqrt(analytic=analytic)/lam.sqrt(analytic=analytic)
    out=acb(0)
    for j in range(-k,k+1):
        mm=moments(t,a,v,s,m,h,j,analytic)
        d=m+2*j;r=a*m-v*t
        density=d*mm[0]+h*mm[1]
        dz=d*mm[1]+h*mm[2]
        dzz=d*mm[2]+h*mm[3]
        residual=r*density+a*h*dz
        residual2=r*r*density+2*r*a*h*dz+a*a*h*h*dzz
        gap=4*j*((j+m)*density+h*dz)
        if component==0:value=density
        elif component==1:
            qr=m*r*density+(a*m+r)*h*dz+a*h*h*dzz
            value=density/a-a*gap/t-qr/(t*lam)
        elif component in (2,7):
            dt=(-3/(2*t)-s*s/(2*lam))*density+a*a*gap/(2*t*t)+v*residual/(t*lam)+(1/t+s*s/lam)*residual2/(2*t*lam)
            value=-dt if component==2 else -z*dt
        elif component in (3,6):
            c0=-2*a*a*j/t-a*r/(t*lam);c1=-a*a*h/(t*lam)
            if component==3:value=(1 if lower else -1)*(mm[0]+c0*density+c1*dz)
            else:value=mm[1]+d*c0*mm[1]+(d*c1+h*c0)*mm[2]+h*c1*mm[3]
        elif component==4:value=(-1 if lower else 1)*residual/lam
        else:value=s*t/lam*(residual2/(t*lam)-density)
        out+=pref*value
    return out
