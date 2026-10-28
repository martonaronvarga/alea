"""Gaussian-drift-averaged integrated images, independently enclosed with Acb.

E[exp(kV) Phi(A+B V)] = exp(kv+k²s²/2)
    Phi((A+B(v+ks²))/sqrt(1+B²s²)).
All derivatives below differentiate this closed form, not a Rust evaluator.
"""
from flint import arb,acb,ctx
from generate_wiener_balls import moment


def point(t,a,q,v,s,lower,k,component):
    lam=1+s*s*t;root=(t*lam).sqrt();total=acb(0)
    for j in range(-k,k+1):
        direction=1 if j>=0 else -1
        d=direction*a*(q+2*j)
        dq=direction*a
        for sign in [1,-1]:
            # Preserve exact cancellations of the common uncertain start
            # coordinate (notably shift=0 in the central forward image).
            shift=a*((1-sign*direction)*q-2*sign*direction*j)
            mean=v+shift*s*s
            arg=(sign*mean*t-d)/root
            pref=(shift*v+shift*shift*s*s/2).exp()
            probability=(-arg/arb(2).sqrt()).erfc()/2
            normal=(-arg*arg/2).exp()/(2*arb.pi()).sqrt()
            z=-arg
            large_tilt=abs(shift*v+shift*shift*s*s/2).upper()>100
            if large_tilt and not z.real>0 and not z.real<0:
                return acb('nan')
            if z.real>0 and large_tilt:
                # Combine the exponentially growing drift tilt with the
                # normal tail before ball evaluation. Their common exponent
                # is exactly the one in the Wiener5 image density.
                residual=a*q-v*t
                energy=-2*a*a*j*(j+q)/t-residual*residual/(2*t*lam)
                u0=(z*z/2).hypgeom_u(arb('.5'),arb('.5'))
                u1=(z*z/2).hypgeom_u(arb('1.5'),arb('1.5'))
                value=energy.exp()*u0/(2*arb.pi().sqrt())
                if component==0:total+=direction*value;continue
                if component==1:
                    de=-4*a*j*(j+q)/t-q*residual/(t*lam)
                    dz=(direction*(q+2*j)+s*s*t*((direction-sign)*q+2*direction*j))/root
                elif component==2:
                    de=-(2*a*a*j*(j+q)/(t*t)+v*residual/(t*lam)+residual*residual/(2*t*lam)*(1/t+s*s/lam))
                    dz=(d+t*(s*s*a*((direction+sign)*q+2*direction*j)+sign*v))/(2*t*lam*root)
                elif component in (3,6):
                    de=-2*a*a*j/t-a*residual/(t*lam)
                    dz=a*(direction+(direction-sign)*s*s*t)/root
                    if not lower:de=-de;dz=-dz
                elif component==4:
                    de=residual/lam;dz=-sign*t/root
                    if lower:de=-de;dz=-dz
                elif component==5:
                    de=s*residual*residual/(lam*lam)
                    dz=2*s*t*a*((direction-sign)*q+2*direction*j)/root-z*s*t/lam
                else:raise ValueError('invalid component')
                # Keep the derivative in numerator form. Dividing by an
                # interval U0 introduces an artificial pole when its enclosure
                # contains zero, although the differentiated CDF is analytic.
                total+=direction*energy.exp()/(2*arb.pi().sqrt())*(u0*de-z*u1*dz/2)
                continue
            if component==0:value=probability
            else:
                if component==1:
                    dk=shift/a
                    dp=dk*mean
                    dz=(sign*dk*s*s*t-d/a)/root
                elif component==2:
                    dp=arb(0)
                    dz=-(sign*mean/root-arg*(1/t+s*s/lam)/2)
                elif component in (3,6):
                    dk=a-sign*dq
                    dp=dk*mean
                    dz=(sign*dk*s*s*t-dq)/root
                    if not lower:dp=-dp;dz=-dz
                elif component==4:
                    dp=shift;dz=sign*t/root
                    if lower:dp=-dp;dz=-dz
                elif component==5:
                    dp=shift*shift*s
                    dz=sign*2*shift*s*t/root-arg*s*t/lam
                else:raise ValueError('invalid component')
                value=dp*probability+dz*normal
            total+=direction*pref*value
    return total


def tails(t,a,v,s,qlo,qhi,k):
    """Integrate absolute density/derivative tails using monotonicity in time.

    t^-r exp(-A/t) increases through t when A>=r*t. The largest r is 5/2;
    every omitted image has A>=a²((2(k+1)-qhi)²-qhi²)/2.
    """
    n=arb(2*(k+1));u=t/(a*a)
    if not (n-qhi)**2-qhi*qhi>=5*u:
        raise ArithmeticError('omitted time tails are not monotone')
    moments=[moment(p,n+qlo,arb(2),1/(2*u))+moment(p,n-qhi,arb(2),1/(2*u)) for p in range(4)]
    factor=a/(2*arb.pi()*t).sqrt()*(qhi*qhi/(2*u)).exp()
    m0,m1,m2,m3=[factor*x for x in moments]
    residual=a*qhi+abs(v)*t
    alpha=(1/a+qhi*residual/t+a*qhi*qhi/t)*m1+a*m3/t
    beta=m0+a*a*m2/t+(a*a*qhi+a*residual)*m1/t
    return [m1,alpha,m1/t,beta,residual*m1,s*(residual*residual+t)*m1,beta/2]


def evaluate(params,goal='1e-14',max_precision=384,progress=None):
    """Continuous start average of two exact Gaussian-drift CDFs.

    Same exact-binary64 parameter order as wiener7_reference.evaluate.
    Time width must be positive. Both clipped and unclipped time are supported.
    """
    for precision in [96,192,384,768]:
        if precision>max_precision:break
        ctx.prec=precision
        rt,a,tau,beta,v,s,h,st=map(arb,params[:8]);lower=bool(params[8])
        if not st>0:raise ValueError('positive time width required')
        if lower:v=-v
        q=beta if lower else 1-beta
        hi=rt-tau;lo=max(arb(0),hi-st)
        if not hi>0 or not q-h/2>0 or not q+h/2<1:raise ValueError('invalid support')
        for k in [0,1,2,4,8,16,32,64]:
            try:
                high_error=tails(hi,a,v,s,q-h/2,q+h/2,k)
                low_error=tails(lo,a,v,s,q-h/2,q+h/2,k) if lo>0 else [arb(0)]*7
            except ArithmeticError:continue
            center=point(hi,a,acb(q),v,s,lower,k,0).real
            if lo>0:center-=point(lo,a,acb(q),v,s,lower,k,0).real
            if not center>0:continue
            if max((x+y).upper() for x,y in zip(high_error,low_error))>arb(goal)*center/8:
                continue
            scale=center.mid();values=[]
            for component in range(7):
                def integrand(x,analytic):
                    trial=acb(q)+(1 if lower else -1)*h*x
                    high=point(hi,a,trial,v,s,lower,k,component)
                    low=point(lo,a,trial,v,s,lower,k,component) if lo>0 else acb(0)
                    value=(high-low)/scale
                    return x*value if component==6 else value
                value=integrand(acb(0),False) if h==0 else acb.integral(integrand,arb('-.5'),arb('.5'),abs_tol=arb(goal)/4096,rel_tol=arb(goal)/4096,eval_limit=4000)
                error=(high_error[component]+low_error[component])/scale
                values.append(value.real+arb(0,error.upper()))
                if progress:progress(precision,k,component,str(values[-1]))
            # The lower time face supplies the time-width derivative.
            if lo>0:
                def face(x,analytic):
                    trial=acb(q)+(1 if lower else -1)*h*x
                    return -point(lo,a,trial,v,s,lower,k,2)/scale
                low_density=face(acb(0),False) if h==0 else acb.integral(face,arb('-.5'),arb('.5'),abs_tol=arb(goal)/4096,rel_tol=arb(goal)/4096,eval_limit=4000)
                low_density=low_density.real+arb(0,(low_error[2]/scale).upper())
            else:low_density=arb(0)
            if not values[0]>0:continue
            result=[(values[0]*scale/st).log()]+[x/values[0] for x in values[1:]]+[low_density/values[0]-1/st]
            if all(x.is_finite() and x.rad()<arb(goal)*(1+abs(x.mid())) for x in result):
                return result,{'precision':precision,'terms':k,'time_integration':'analytic Gaussian-drift CDF','start_integration':'continuous Acb quadrature'}
    raise ArithmeticError('CDF reference work/precision budget exhausted')
