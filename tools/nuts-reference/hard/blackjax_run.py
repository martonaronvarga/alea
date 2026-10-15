"""Hard-target CPU runner with separately compiled and synchronized scans."""
import csv
import json
import sys
import time
from pathlib import Path
from importlib.metadata import version
from models import jax,jnp,np,CASES,WARMUP,DRAWS,DEPTH,SEEDS,density,initial
import blackjax

def main():
    case=int(sys.argv[1]);outpath=Path(sys.argv[2]);timing=Path(sys.argv[3])
    assert version('blackjax')=='1.5' and version('jax')=='0.9.2' and version('jaxlib')=='0.9.2'
    assert jax.config.x64_enabled and jax.default_backend()=='cpu'
    jax.config.update('jax_enable_compilation_cache',False)
    _,d,step=CASES[case]
    kernel=blackjax.nuts(density(case),step,jnp.ones(d),max_num_doublings=DEPTH)
    def run(state,keys):
        def advance(state,key):
            state,info=kernel.step(key,state)
            return state,jnp.concatenate((jnp.array([info.is_divergent,info.num_integration_steps]),state.position))
        return jax.lax.scan(advance,state,keys)
    state=kernel.init(jnp.array(initial(d,701)));keys=jax.random.split(jax.random.key(701),WARMUP+DRAWS)
    jax.block_until_ready((state,keys));start=time.perf_counter()
    warm=jax.jit(run).lower(state,keys[:WARMUP]).compile()
    sample=jax.jit(run).lower(state,keys[WARMUP:]).compile()
    compilation=time.perf_counter()-start
    with outpath.open('x') as out,timing.open('x') as times:
        writer=csv.writer(times);writer.writerow(('seed','warmup_seconds','sampling_seconds','warmup_steps'))
        out.write('seed,draw,divergent,steps,'+','.join(f'q{i}' for i in range(d))+'\n')
        for seed in SEEDS:
            state=kernel.init(jnp.array(initial(d,seed)));keys=jax.random.split(jax.random.key(seed),WARMUP+DRAWS)
            wk,sk=keys[:WARMUP],keys[WARMUP:];jax.block_until_ready((state,wk,sk))
            start=time.perf_counter();state,discarded=jax.block_until_ready(warm(state,wk));wt=time.perf_counter()-start
            start=time.perf_counter();_,draws=jax.block_until_ready(sample(state,sk));st=time.perf_counter()-start
            writer.writerow((seed,wt,st,int(np.asarray(discarded)[:,1].sum())))
            rows=np.column_stack((np.full(DRAWS,seed),np.arange(DRAWS),np.asarray(draws)))
            np.savetxt(out,rows,delimiter=',',fmt=['%d']*4+['%.17e']*d)
    with timing.with_suffix('.compile.json').open('x') as out:json.dump(dict(seconds=compilation),out)

if __name__=='__main__':main()
