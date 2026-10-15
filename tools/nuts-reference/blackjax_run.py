"""Pinned BlackJAX NUTS retained coordinates for the predeclared M5 protocol."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'blackjax-reference'))
from generate import correlated, banana, logistic, funnel, VERSIONS
from importlib.metadata import version
import blackjax
import jax
import jax.numpy as jnp
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    for package,expected in VERSIONS.items():
        assert version(package)==expected
    assert jax.config.x64_enabled and jax.default_backend()=='cpu'
    with args.output.open('x') as out:
        out.write('target,seed,draw,q0,q1,divergent,steps,acceptance\n')
        for t,density in enumerate((correlated,banana,logistic,funnel)):
            kernel=blackjax.nuts(density,0.06 if t==3 else 0.15,jnp.ones(2),max_num_doublings=8)
            @jax.jit
            def run(seed):
                sign=jnp.where(seed%2==0,-1.0,1.0)
                state=kernel.init(jnp.array([sign,-sign]))
                def step(state,key):
                    state,info=kernel.step(key,state)
                    return state,jnp.concatenate((state.position,jnp.array([info.is_divergent,info.num_integration_steps,info.acceptance_rate])))
                return jax.lax.scan(step,state,jax.random.split(jax.random.key(seed),9216))[1]
            for seed in range(501,505):
                values=np.asarray(run(seed))[1024:]
                rows=np.column_stack((np.full(8192,t),np.full(8192,seed),np.arange(8192),values))
                np.savetxt(out,rows,delimiter=',',fmt=['%d','%d','%d','%.17e','%.17e','%d','%d','%.17e'])
            print(f'BlackJAX completed target {t}',flush=True)

if __name__=='__main__':main()
