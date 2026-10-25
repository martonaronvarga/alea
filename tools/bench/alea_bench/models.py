"""Independent model measures and generative checks; no sampler implementation."""
import math
import random


def density(case, q, xp=None):
    """Unnormalized log density with respect to the declared unconstrained dq.

    Python math is the independent finite-difference oracle; JAX supplies xp for
    an independently differentiated reference. Constants in Wiener data are kept.
    """
    k, d, x = case['kind'], case['dimension'], case['data']
    exp, log = (math.exp, math.log) if xp is None else (xp.exp, xp.log)
    softplus = (lambda z: max(z, 0) + math.log1p(math.exp(-abs(z)))) if xp is None else (lambda z: xp.logaddexp(z, 0.0))
    if k == 0:
        return -.5 * sum((q[i] / x[i]) ** 2 for i in range(d))
    if k == 1:
        return -.5 * q[0] ** 2 - .5 * sum((q[i] - x[0] * q[i-1]) ** 2 for i in range(1, d)) / (1-x[0]**2)
    if k == 2:
        return -.5 * sum(q[i]**2 + (q[i+1] - x[0]*(q[i]**2-1))**2 for i in range(0, d, 2))
    if k == 3:
        return -.5 * (q[0]/x[0])**2 - .5*(d-1)*q[0] - .5*sum(q[i]**2 for i in range(1,d))*exp(-q[0])
    if k == 4:
        value = -.5*sum(v*v for v in q)
        for n in range(case['rows']):
            row = x[n*(d+1):(n+1)*(d+1)]
            eta = sum(q[i]*row[i] for i in range(d))
            value += row[d]*eta-softplus(eta)
        return value
    if k in (5,9):
        tau = exp(q[1])
        value = -.5*(q[0]/.7)**2-.5*((q[1]+.7)/.3)**2-.5*(q[2]**2+q[3]**2)
        for n in range(case['rows']):
            group, y = x[2*n:2*n+2]
            eta = q[0]+tau*q[2+int(group)]
            value += -.5*((y-eta)/.7)**2 if k == 5 else y*1.2*eta-softplus(1.2*eta)
        return value
    if k == 6:
        phi, sigma = x[:2]
        return (-.5*q[0]**2*(1-phi*phi)/sigma**2
                -.5*sum((q[i]-phi*q[i-1])**2 for i in range(1,d))/sigma**2
                -.5*sum(q[i]+x[i+2]**2*exp(-q[i]) for i in range(d)))
    if k == 7:
        return -.5*sum(q[i]**2+4*(x[i]-q[i])**2 for i in range(d))
    if k == 8:
        return -.5*q[0]**2 + sum(x[3*n+2]+.6*x[3*n+1]*q[0]-.5*q[0]**2*(x[3*n]-.15) for n in range(case['rows']))
    if k in (10,11):
        tau = exp(q[1])
        value = -.5*(q[0]/5)**2+q[1]-log(1+(tau/5)**2)
        for i in range(8):
            y,sigma=x[2*i:2*i+2]
            if k == 10:
                value -= q[1]+.5*((q[i+2]-q[0])/tau)**2+.5*((y-q[i+2])/sigma)**2
            else:
                value -= .5*q[i+2]**2+.5*((y-q[0]-tau*q[i+2])/sigma)**2
        return value
    raise ValueError('unknown density')


def expected(case):
    """Independent analytic marginal mean/variance where known; absent otherwise."""
    k, d, x = case['kind'], case['dimension'], case['data']
    if k == 0: return [(0.,s*s) for s in x]
    if k == 1: return [(0.,1.)]*d
    if k == 2: return [(0.,1. if i%2 == 0 else 1+2*x[0]**2) for i in range(d)]
    if k == 3: return [(0.,x[0]**2)]+[(0.,math.exp(.5*x[0]**2))]*(d-1)
    if k == 7: return [(.8*y,.2) for y in x]
    if k == 8:
        precision=1+sum(x[3*n]-.15 for n in range(case['rows']))
        return [(.6*sum(x[3*n+1] for n in range(case['rows']))/precision,1/precision)]
    return None


def generate(case, seed):
    """Fresh prior truth and observations, outside log densities/Jacobians."""
    rng=random.Random(seed); k=case['kind']; data=[]
    result=dict(case)
    if k == 7:
        truth=[rng.gauss(0,1) for _ in range(case['dimension'])]
        data=[q+rng.gauss(0,.5) for q in truth]
    elif k in (5,9):
        truth=[rng.gauss(0,.7),rng.gauss(-.7,.3),rng.gauss(0,1),rng.gauss(0,1)]
        for i in range(case['rows']):
            group=i%2; eta=truth[0]+math.exp(truth[1])*truth[2+group]
            event=eta+rng.gauss(0,.7) if k==5 else float(rng.random()<1/(1+math.exp(-1.2*eta)))
            data.extend((group,event))
    else: raise ValueError('SBC generator unavailable for this measure')
    result['data']=data
    return result,truth


def predictive(case, draws, seed):
    """Posterior predictive coverage on independently generated held-out events."""
    rng=random.Random(seed);k=case['kind']; replicated=[]
    for q in draws:
        if k==7: replicated.append(q[0]+rng.gauss(0,.5))
        elif k==5: replicated.append(q[0]+math.exp(q[1])*q[2]+rng.gauss(0,.7))
        elif k==9:
            eta=q[0]+math.exp(q[1])*q[2]
            replicated.append(float(rng.random()<1/(1+math.exp(-1.2*eta))))
        else: raise ValueError('predictive measure unavailable')
    return replicated
