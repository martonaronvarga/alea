"""Independent JAX densities and the predeclared harder CPU workloads."""
import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ['JAX_ENABLE_X64']='true'
import jax
import jax.numpy as jnp
import numpy as np

CASES=[('gaussian32',32,.08),('gaussian128',128,.10),('banana32',32,.12),('logistic32x512',32,.02),('funnel10_width3',10,.04)]
WARMUP,DRAWS,DEPTH=1024,2048,9
SEEDS=range(701,705)

def data(case):
    d=CASES[case][1];n=512 if case==3 else 0
    x=np.array([[((17*i+13*j+7*i*j)%101)/50.-1. for j in range(d)] for i in range(n)]).reshape(n,d)
    y=np.array([int((37*i)%101<50) for i in range(n)],dtype=int)
    return dict(which=case,D=d,N=n,X=x.tolist(),y=y.tolist())

def density(case):
    if case<=1:
        rho=.95 if case==0 else .9
        return lambda q: -.5*(q[0]**2+jnp.sum((q[1:]-rho*q[:-1])**2)/(1-rho*rho))
    if case==2:
        return lambda q: -.5*jnp.sum(q[::2]**2+(q[1::2]-.8*(q[::2]**2-1))**2)
    if case==3:
        dataset=data(case);x=jnp.array(dataset['X']);y=jnp.array(dataset['y'])
        return lambda q: -.5*jnp.sum(q*q)+jnp.sum(y*(x@q)-jax.nn.softplus(x@q))
    if case==4:
        return lambda q: -.5*q[0]**2/9-.5*(q.size-1)*q[0]-.5*jnp.sum(q[1:]**2)*jnp.exp(-q[0])
    raise ValueError('unknown case')

def initial(d,seed):
    return np.array([(.1 if seed%2 else -.1)*(-1 if i%2 else 1) for i in range(d)])

def generative(case,q):
    z=q.copy()
    if case<=1:
        rho=.95 if case==0 else .9
        z[...,1:]=(q[...,1:]-rho*q[...,:-1])/np.sqrt(1-rho*rho)
    elif case==2:z[...,1::2]-=.8*(q[...,::2]**2-1)
    elif case==4:
        z[...,0]/=3
        z[...,1:]*=np.exp(-.5*q[...,0,None])
    return z
