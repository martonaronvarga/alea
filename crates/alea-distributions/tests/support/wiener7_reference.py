#!/usr/bin/env python3
"""Pointwise certified Wiener7 reference with precision escalation (python-flint 0.9).

Finite image densities/derivatives are integrated with Acb on the fixed trial
variability cube. Analytic Gaussian envelopes bound omitted images uniformly
on the full integration rectangle. Clipped t=0 support is removed with an
explicit uniform bound on the omitted time slab, including derivative numerators.
All complex square roots forward Acb's analyticity flag. The result encloses
the mathematical likelihood and scores, not an assumed f64 operation model.
"""
import argparse,json,math
from flint import arb,acb,ctx
from generate_wiener_balls import moment


def point(t,q,a,v,s,x,z,lower,k,analytic,component):
    lam=1+s*s*t
    pref=a/(2*arb.pi()).sqrt()/t/t.sqrt(analytic=analytic)/lam.sqrt(analytic=analytic)
    n=-v*v*t+2*a*v*q+a*a*q*q*s*s
    out=acb(0)
    for j in range(-k,k+1):
        d=q+2*j
        e=(-a*a*d*d/(2*t)+n/(2*lam)).exp()*pref
        f=d*e
        if component==0:value=f
        elif component==1:value=f*(1/a-a*d*d/t+(v*q+a*q*q*s*s)/lam)
        elif component in (2,7):
            dt=f*(-3/(2*t)-s*s/(2*lam)+a*a*d*d/(2*t*t)+(-v*v*lam-n*s*s)/(2*lam*lam))
            value=-dt if component==2 else -z*dt
        elif component in (3,6):
            dq=e*(1+d*(-a*a*d/t+(a*v+a*a*q*s*s)/lam))
            value=(1 if lower else -1)*dq if component==3 else x*dq
        elif component==4:value=(-1 if lower else 1)*f*(-v*t+a*q)/lam
        else:value=f*(-s*t/lam+a*a*q*q*s/lam-n*s*t/(lam*lam))
        out+=value
    return out


def envelopes(a,v,s,qlo,qhi,tlo,thi,k,all_images=False):
    """Bounds each numerator over all q,t in the closed positive rectangle.

    For the clipped slab [0,thi], all_images=True is valid only when
    thi*(1+s²*thi)/a² <= qlo²/8. Gaussian t^-r exp(-a²qlo²/(2t*(1+s²t))) is then increasing for
    every exponent r<=7/2 used here. This also bounds the infinite image tails.
    """
    u=thi/(a*a)
    # Keep the drift-variance cancellation before bounding. For image d=q+2j,
    # -a²d²/(2t)+a²q²s²/(2lambda)
    # =-a²(d²-q²)/(2t)-a²q²/(2t*lambda).
    # Every noncentral image has d²>=q²; the central image is handled separately.
    lam=1+s*s*thi
    shift=(qhi*qhi/(2*u)-qlo*qlo/(2*u*lam)).exp()
    if all_images:
        if not u*lam<=qlo*qlo/8:raise ArithmeticError('unproved time-zero monotonicity')
        moments=[qhi**p*(-qlo*qlo/(2*u*lam)).exp()+shift*(moment(p,2+qlo,arb(2),1/(2*u))+moment(p,2-qhi,arb(2),1/(2*u))) for p in range(4)]
        tlo=thi
    else:
        moments=[shift*(moment(p,2*(k+1)+qlo,arb(2),1/(2*u))+moment(p,2*(k+1)-qhi,arb(2),1/(2*u))) for p in range(4)]
    scale=(a*abs(v)*qhi).exp()*a/(2*arb.pi()).sqrt()/tlo**arb('1.5')
    density=scale*moments[1]
    time=scale*(moments[3]/(2*(tlo/(a*a))**2)+3*moments[1]/(2*tlo/(a*a)))
    start=scale*(moments[0]+moments[2]/(tlo/(a*a)))
    nb=v*v*thi+2*a*abs(v)*qhi+a*a*qhi*qhi*s*s
    pa=2/a+abs(v)*qhi+a*qhi*qhi*s*s
    pt=s*s/2+v*v/2+nb*s*s/2
    pq=a*abs(v)+a*a*qhi*s*s
    pv=abs(v)*thi+a*qhi
    ps=s*thi+a*a*qhi*qhi*s+nb*s*thi
    beta=pq*density+start
    tau=pt*density+time/(a*a)
    return [density,pa*density+2*u/a*time,tau,beta,pv*density,ps*density,beta/2,tau]


def evaluate(params,goal='1e-18',max_precision=768):
    """params=[rt,a,tau,beta,delta,sd,start_width,time_width,lower]."""
    source=list(params)
    if len(source) != 9 or not all(math.isfinite(float(x)) for x in source[:8]) or source[2] < 0:
        raise ValueError("expected finite parameters and nonnegative nondecision time")
    for precision in [96,192,384,768]:
        if precision>max_precision:break
        ctx.prec=precision
        rt,a,tau,beta,v,s,h,st=map(arb,source[:8]);lower=bool(source[8])
        if not (a>0 and rt>tau and beta-h/2>0 and beta+h/2<1 and s>=0 and h>=0 and st>=0):raise ValueError('outside reference support')
        q=beta if lower else 1-beta
        if lower:v=-v
        thi=rt-tau;tmin=thi-st if st>0 else thi
        clipped=st>0 and not tmin>0
        probes=[]
        for qp in [q-h/2,q,q+h/2]:
            for tp in [thi/2,thi]:
                f=point(acb(tp),acb(qp),a,v,s,0,0,lower,8,False,0).real
                if f>0:probes.append(f.log().mid())
        if not probes:continue
        log_scale=max(probes);normalizer=(-log_scale).exp()
        cutoff=arb(0);slab=[arb(0)]*8
        if clipped:
            c=a*a*(q-h/2)**2/16
            cutoff=min((thi/16).lower(),(c/(1+s*c.sqrt())).lower())
            for _ in range(64):
                slab=[e*cutoff/st for e in envelopes(a,v,s,q-h/2,q+h/2,cutoff,cutoff,0,True)]
                if max((e*normalizer).upper() for e in slab)<arb(goal)**2:break
                cutoff/=2
            else:raise ArithmeticError('time-zero slab did not meet budget')
            tmin=cutoff
        for k in [4,8,16,32]:
            try:
                tails=envelopes(a,v,s,q-h/2,q+h/2,tmin,thi,k)
            except (ArithmeticError,ValueError):continue
            zhi=min(arb(1),(thi-tmin)/st) if st>0 else arb(0)
            tol=arb(2)**(-precision+24)
            values=[]
            for component in range(8):
                def outer(z,analytic):
                    t=acb(thi)-st*z if st>0 else acb(thi)
                    def inner(x,inner_analytic):
                        zz=z if st>0 else arb('0.5')
                        return normalizer*point(t,acb(q)+h*x,a,v,s,x,zz,lower,k,analytic or inner_analytic,component)
                    if h==0:return inner(acb(0),analytic)
                    return acb.integral(inner,arb('-0.5'),arb('0.5'),rel_tol=tol,abs_tol=tol,eval_limit=4000)
                if st>0:
                    # Geometric positive-time panels avoid analyticity disks
                    # spanning the essential singularity at t=0.
                    value=acb(0);left=tmin
                    while left<thi:
                        right=min(thi,(left*2).lower())
                        lo=(thi-right)/st;hi=(thi-left)/st
                        value+=acb.integral(outer,lo,hi,rel_tol=tol,abs_tol=tol,eval_limit=4000)
                        left=right
                else:value=outer(acb(0),False)
                error=tails[component]*(zhi if st>0 else 1)+slab[component]
                values.append(value.real+arb(0,(error*normalizer).upper()))
            if values[0]>0:
                result=[values[0].log()+log_scale]+[x/values[0] for x in values[1:]]
                if all(x.is_finite() and x.rad()<arb(goal)*(1+abs(x.mid())) for x in result):
                    return result,{'precision':precision,'terms':k,'clipped':clipped,'cutoff':str(cutoff),'slab_upper':[str(x.upper()) for x in slab]}
    raise ArithmeticError('reference work/precision budget exhausted')


def certify(values,candidate,absolute='1e-9',relative='1e-9'):
    if len(candidate)!=8 or any(not math.isfinite(float(x)) for x in candidate):
        return {'certified':False,'status':'nonfinite_candidate'}
    if len(values) != 8 or any(not x.is_finite() for x in values):
        raise ValueError('expected eight finite reference enclosures')
    if any(not arb(x).is_finite() or arb(x)<0 for x in [absolute,relative]):
        raise ValueError('expected finite nonnegative tolerances')
    differences=[abs(value-arb(float(observed))) for value,observed in zip(values,candidate)]
    upper=[x.upper() for x in differences]
    lower=[max(arb(0),x.lower()) for x in differences]
    tol_low=[arb(absolute)+arb(relative)*abs(value).lower() for value in values]
    tol_high=[arb(absolute)+arb(relative)*abs(value).upper() for value in values]
    passed=all(e<=t for e,t in zip(upper,tol_low))
    rejected=any(e>t for e,t in zip(lower,tol_high))
    return {'certified':passed,'status':'certified' if passed else 'rejected' if rejected else 'unresolved',
            'error_upper':[str(x) for x in upper],'error_lower':[str(x) for x in lower],
            'tolerance_lower':[str(x) for x in tol_low],'candidate':candidate,
            'enclosures':[str(x) for x in values]}


def audit(params,candidate,absolute='1e-9',relative='1e-9'):
    if len(candidate)!=8 or any(not math.isfinite(float(x)) for x in candidate):
        return {'certified':False,'status':'nonfinite_candidate'}
    tolerances=[arb(absolute),arb(relative)]
    if any(not x.is_finite() or x<0 for x in tolerances):
        raise ValueError('expected finite nonnegative tolerances')
    positive=[x for x in tolerances if x>0]
    # Spend precision on the requested comparison. Certification below still
    # uses the actual enclosure, independently of this work-budget choice.
    goal=min(positive)/1024 if positive else arb('1e-30')
    values,metadata=evaluate(params,goal=goal)
    result=certify(values,candidate,absolute,relative)
    result.update(metadata=metadata,params=params)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('input',help='JSON with params and candidate arrays');args=parser.parse_args()
    data=json.loads(args.input)
    try:result=audit(data['params'],data['candidate'],data.get('absolute','1e-9'),data.get('relative','1e-9'))
    except ArithmeticError as e:result={'certified':False,'status':'reference_exhausted','message':str(e)}
    except ValueError as e:result={'certified':False,'status':'invalid_parameters','message':str(e)}
    print(json.dumps(result,indent=2))
