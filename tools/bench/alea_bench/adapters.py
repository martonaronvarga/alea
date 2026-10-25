"""Worker entry points for independently pinned external engines."""
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from . import models

HEADER=['chain','draw','divergent','saturated','steps','acceptance','energy','energy_error']
PHASES=['chain','setup_seconds','warmup_seconds','retained_seconds','setup_gradient_calls','warmup_gradient_calls','retained_gradient_calls','warmup_steps','retained_steps','velocity_actions','momentum_actions']


def blackjax(case,policy,directory):
    started=time.perf_counter()
    import importlib.metadata
    import jax
    import jax.numpy as jnp
    import numpy as np
    import blackjax
    versions={name:importlib.metadata.version(name) for name in ('blackjax','jax','jaxlib')}
    if versions!={'blackjax':'1.5','jax':'0.9.2','jaxlib':'0.9.2'}:raise RuntimeError(f'unpinned JAX engine: {versions}')
    if not jax.config.x64_enabled or jax.default_backend()!='cpu':raise RuntimeError('requires float64 CPU JAX')
    jax.config.update('jax_enable_compilation_cache',False)
    kernel=blackjax.nuts(lambda q:models.density(case,q,jnp),case['step'],jnp.ones(case['dimension']),max_num_doublings=policy['depth'])
    def advance(state,keys):
        def step(state,key):
            state,info=kernel.step(key,state)
            row=jnp.concatenate((jnp.array([info.is_divergent,info.num_integration_steps,info.acceptance_rate,info.num_trajectory_expansions]),state.position))
            return state,row
        return jax.lax.scan(step,state,keys)
    initial=jnp.array(case['initial'],dtype=jnp.float64);state=kernel.init(initial)
    keys=jax.random.split(jax.random.key(policy['seeds'][0]),max(policy['warmup'],1)+policy['draws'])
    jax.block_until_ready((state,keys));compile_start=time.perf_counter()
    warm=jax.jit(advance).lower(state,keys[:policy['warmup']]).compile() if policy['warmup'] else None
    sample=jax.jit(advance).lower(state,keys[-policy['draws']:]).compile()
    compilation=time.perf_counter()-compile_start
    with (directory/'draws.csv').open('x') as raw,(directory/'phases.csv').open('x') as phases:
        rows=csv.writer(raw);rows.writerow(HEADER+[f'q{i}' for i in range(case['dimension'])]);times=csv.writer(phases);times.writerow(PHASES)
        for chain,seed in enumerate(policy['seeds']):
            start=time.perf_counter();state=kernel.init(initial);keys=jax.random.split(jax.random.key(seed),policy['warmup']+policy['draws']);jax.block_until_ready((state,keys));setup=time.perf_counter()-start
            start=time.perf_counter()
            if warm is not None:state,discarded=jax.block_until_ready(warm(state,keys[:policy['warmup']]))
            wt=time.perf_counter()-start;warm_steps=int(np.asarray(discarded)[:,1].sum()) if warm else 0
            start=time.perf_counter();state,draws=jax.block_until_ready(sample(state,keys[policy['warmup']:]))
            retained=time.perf_counter()-start;draws=np.asarray(draws)
            times.writerow([chain,setup,wt,retained,'','','',warm_steps,int(draws[:,1].sum()),'',''])
            for i,row in enumerate(draws):rows.writerow([chain,i,int(row[0]),int(row[3]>=policy['depth']),int(row[1]),row[2],'','',*row[4:]])
    (directory/'engine.json').write_text(json.dumps({'versions':versions,'jit_seconds':compilation,'cold_seconds':compile_start-started,'energy':'unavailable: BlackJAX proposal energy differs from initial Hamiltonian','actual_threads':threads()},allow_nan=False)+'\n')


def threads():
    try:return int(next(s.split(':')[1] for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('Threads:')))
    except (OSError,StopIteration):return None


def stan(case,policy,directory,binary):
    data=directory/'stan-data.json';data.write_text(json.dumps({'kind':case['kind'],'D':case['dimension'],'N':case['rows'],'K':len(case['data']),'x':case['data']}))
    init=directory/'stan-init.json';init.write_text(json.dumps({'q':case['initial']}))
    with (directory/'draws.csv').open('x') as raw,(directory/'phases.csv').open('x') as phases:
        rows=csv.writer(raw);rows.writerow(HEADER+[f'q{i}' for i in range(case['dimension'])]);times=csv.writer(phases);times.writerow(PHASES)
        for chain,seed in enumerate(policy['seeds']):
            output=directory/f'stan-{chain}.csv'
            command=[binary,'sample',f'num_warmup={policy["warmup"]}',f'num_samples={policy["draws"]}','save_warmup=1','adapt','engaged=0','algorithm=hmc','engine=nuts',f'max_depth={policy["depth"]}','metric=unit_e',f'stepsize={case["step"]}','stepsize_jitter=0', 'data',f'file={data}','random',f'seed={seed}',f'init={init}','output',f'file={output}','sig_figs=17','refresh=0']
            subprocess.run(command,check=True)
            lines=output.read_text().splitlines();samples=list(csv.DictReader(line for line in lines if not line.startswith('#')))
            if len(samples)!=policy['warmup']+policy['draws']:raise RuntimeError('Stan lost warmup/retained rows')
            text='\n'.join(lines);wt=float(re.search(r'([\d.eE+-]+) seconds \(Warm-up\)',text)[1]);rt=float(re.search(r'([\d.eE+-]+) seconds \(Sampling\)',text)[1])
            times.writerow([chain,'',wt,rt,'','','',sum(int(row['n_leapfrog__']) for row in samples[:policy['warmup']]),sum(int(row['n_leapfrog__']) for row in samples[policy['warmup']:]),'',''])
            for i,row in enumerate(samples[policy['warmup']:]):rows.writerow([chain,i,row['divergent__'],int(int(row['treedepth__'])>=policy['depth']),row['n_leapfrog__'],row['accept_stat__'],row['energy__'],'',*[row[f'q.{j+1}'] for j in range(case['dimension'])]])
    (directory/'engine.json').write_text(json.dumps({'versions':{'stan':'2.38.0'},'jit_seconds':0.,'cold_seconds':None,'energy':'Stan energy__ at reported transition; native timings include CSV writes','actual_threads':threads()})+'\n')


def main():
    engine,case_path,policy_path,directory,*extra=sys.argv[1:]
    case=json.loads(Path(case_path).read_text());policy=json.loads(Path(policy_path).read_text());directory=Path(directory).resolve()
    if engine=='blackjax':blackjax(case,policy,directory)
    elif engine=='stan':stan(case,policy,directory,extra[0])
    else:raise ValueError('unsupported external adapter')

if __name__=='__main__':main()
