"""Auditable native Rust primals for the pinned twenty-model PosteriorDB corpus.

These are hand translations, not a Stan compiler. Coordinates follow Stan 2.38,
including Jacobians. Only parameter-independent density constants may differ.
Data are immutable build inputs; no Stan evaluation enters a native derivative.
"""
import json
import math
from pathlib import Path

from .manifest import ROOT, file_hash, write_new

HELPERS = r"""
fn sp(z: f64) -> f64 { if z > 0.0 { z + (-z).exp().ln_1p() } else { z.exp().ln_1p() } }
fn ls(z: f64) -> f64 { -sp(-z) }
fn sig(z: f64) -> f64 { ls(z).exp() }
fn norm(event: f64, location: f64, logscale: f64) -> f64 { -logscale - 0.5*((event-location)*(-logscale).exp()).powi(2) }
fn lse(a: f64, b: f64) -> f64 { if a > b { a + (b-a).exp().ln_1p() } else { b + (a-b).exp().ln_1p() } }
"""


def literal(value):
    if isinstance(value, list):
        return '[' + ','.join(literal(x) for x in value) + ']'
    if not math.isfinite(value):
        raise ValueError('nonfinite model data')
    return repr(float(value)) + '_f64'


def rust_port(case, data):
    """Return (unconstrained dimension, scalar Rust body) for a named model."""
    model = case.split('-', 1)[1]
    declarations = []

    def array(name, values):
        def rust_type(v):
            return f'[{rust_type(v[0])}; {len(v)}]' if isinstance(v, list) else 'f64'
        declarations.append(f'#[allow(non_upper_case_globals)] const {name}: {rust_type(values)} = {literal(values)};')

    if model in ('Rate_1_model', 'Rate_2_model', 'Rate_5_model'):
        pairs = [(data['k'], data['n'])] if model == 'Rate_1_model' else [(data['k1'], data['n1']), (data['k2'], data['n2'])]
        if model == 'Rate_5_model':
            pairs = [(sum(k for k, _ in pairs), sum(n for _, n in pairs))]
        body = ' + '.join(f'{k+1}.0*ls(x[{i}]) + {n-k+1}.0*ls(-x[{i}])' for i, (k, n) in enumerate(pairs))
        dimension = len(pairs)
    elif model in ('earn_height', 'logearn_interaction_z', 'kidscore_momhs', 'blr', 'wells_dist100_model', 'dogs_log'):
        prior = ''
        domain = ''
        logistic = model in ('wells_dist100_model', 'dogs_log')
        if model == 'earn_height':
            design = [[1., h] for h in data['height']]; response = data['earn']
        elif model == 'logearn_interaction_z':
            heights = data['height']; mean = math.fsum(heights) / len(heights)
            sd = math.sqrt(math.fsum((h-mean)**2 for h in heights)/(len(heights)-1))
            design = [[1., (h-mean)/sd, m, m*(h-mean)/sd] for h, m in zip(heights, data['male'], strict=True)]
            response = [math.log(y) for y in data['earn']]
        elif model == 'kidscore_momhs':
            design = [[1., h] for h in data['mom_hs']]; response = data['kid_score']
            prior = 'lp -= sp(2.0*(x[p]-2.5_f64.ln()));'
        elif model == 'blr':
            design = data['X']; response = data['y']
            prior = 'for j in 0..p { lp -= 0.5*(x[j]/10.0).powi(2); } lp -= 0.5*(x[p].exp()/10.0).powi(2);'
        elif model == 'wells_dist100_model':
            design = [[1., h/100.] for h in data['dist']]; response = data['switched']
        else:
            design = []; response = []
            for dog in data['y']:
                avoid = shock = 0
                for y in dog:
                    design.append([avoid, shock]); response.append(y)
                    avoid += 1-y; shock += y
            domain = 'if x[0] <= -100.0 || x[0] >= 0.0 || x[1] <= 0.0 || x[1] >= 100.0 { return f64::NAN; }'
        p = len(design[0]); dimension = p + (not logistic)
        array('design', design); array('response', response)
        likelihood = 'lp += response[i]*mu-sp(mu);' if logistic else 'lp += norm(response[i],mu,x[p]);'
        mu = ' + '.join(f'design[i][{j}]*x[{j}]' for j in range(p))
        p_declaration = '' if logistic else f'let p = {p};'
        body = f'{domain} {p_declaration} let mut lp = ' + ('0.0' if logistic else 'x[p]') + f'; {prior} for i in 0..response.len() {{ let mu = {mu}; {likelihood} }} lp'
    elif model in ('eight_schools_centered', 'eight_schools_noncentered'):
        n = data['J']; dimension = n+2
        array('y', data['y']); array('sigma', data['sigma'])
        theta = 'x[i]*tau+mu' if model.endswith('noncentered') else 'x[i]'
        prior = 'lp -= 0.5*x[i]*x[i];' if model.endswith('noncentered') else 'lp += norm(x[i],mu,lt);'
        tau = 'let tau=lt.exp();' if model.endswith('noncentered') else ''
        body = f'let mu=x[{n}]; let lt=x[{n+1}]; {tau} let mut lp=-0.5*(mu/5.0).powi(2)+lt-sp(2.0*(lt-5.0_f64.ln())); for i in 0..{n} {{ {prior} lp += norm(y[i],{theta},sigma[i].ln()); }} lp'
    elif model == 'arK':
        k = data['K']; dimension = k+2; array('y', data['y'])
        mu = 'x[0]' + ''.join(f'+x[{j+1}]*y[i-{j+1}]' for j in range(k))
        body = f'let mut lp=x[{k+1}]-sp(2.0*(x[{k+1}]-2.5_f64.ln())); for j in 0..{k+1} {{ lp -= 0.5*(x[j]/10.0).powi(2); }} for i in {k}..y.len() {{ let mu={mu}; lp += norm(y[i],mu,x[{k+1}]); }} lp'
    elif model == 'arma11':
        dimension = 4; array('y', data['y'])
        body = 'let mut lp=x[3]-sp(2.0*(x[3]-2.5_f64.ln()))-0.5*(x[0]/10.0).powi(2)-0.5*(x[1]/2.0).powi(2)-0.5*(x[2]/2.0).powi(2); let mut err=y[0]-x[0]-x[1]*x[0]; lp += norm(err,0.0,x[3]); for i in 1..y.len() { err=y[i]-x[0]-x[1]*y[i-1]-x[2]*err; lp += norm(err,0.0,x[3]); } lp'
    elif model == 'garch11':
        dimension = 4; array('y', data['y'])
        body = f'let a0=x[1].exp(); let a1=sig(x[2]); let b1=sig(-x[2])*sig(x[3]); let mut variance={literal(data["sigma1"]**2)}; let mut lp=x[1]+ls(x[2])+2.0*ls(-x[2])+ls(x[3])+ls(-x[3]); for i in 0..y.len() {{ if i>0 {{ variance=a0+a1*(y[i-1]-x[0]).powi(2)+b1*variance; }} lp += norm(y[i],x[0],0.5*variance.ln()); }} lp'
    elif model in ('low_dim_gauss_mix_collapse', 'normal_mixture'):
        array('y', data['y'])
        if model == 'normal_mixture':
            dimension = 3
            body = 'let a=ls(x[0]); let b=ls(-x[0]); let mut lp=a+b-0.5*(x[1]/10.0).powi(2)-0.5*(x[2]/10.0).powi(2); for v in y { lp += lse(a+norm(v,x[1],0.0),b+norm(v,x[2],0.0)); } lp'
        else:
            dimension = 5
            body = 'let a=ls(x[4]); let b=ls(-x[4]); let mut lp=5.0*(a+b)+x[2]+x[3]; for i in 0..2 { lp -= 0.5*(x[i]/2.0).powi(2)+0.5*(x[i+2].exp()/2.0).powi(2); } for v in y { lp += lse(a+norm(v,x[0],x[2]),b+norm(v,x[1],x[3])); } lp'
    elif model == 'seeds_stanified_model':
        n = data['I']; dimension = n+5
        for name, key in (('n','n'),('trials','N'),('x1','x1'),('x2','x2')): array(name, data[key])
        body = f'let lt=x[{n+4}]; let mut lp=lt-sp(2.0*lt); for j in 0..4 {{ lp -= 0.5*x[j]*x[j]; }} for i in 0..{n} {{ lp += norm(x[i+4],0.0,lt); let z=x[0]+x[1]*x1[i]+x[3]*x2[i]+x[2]*x1[i]*x2[i]+x[i+4]; lp += n[i]*z-trials[i]*sp(z); }} lp'
    elif model == 'surgical_model':
        n = data['N']; dimension = n+2
        array('success', data['r']); array('trials', data['n'])
        body = f'let lv=x[1]; let mut lp=-0.5*(x[0]/1000.0).powi(2)-0.001*lv-0.001*(-lv).exp(); for i in 0..{n} {{ let z=x[i+2]; lp += norm(z,x[0],0.5*lv)+success[i]*z-trials[i]*sp(z); }} lp'
    elif model == 'radon_variable_intercept_noncentered':
        j = data['J']; dimension = j+4
        for name in ('county_idx', 'floor_measure', 'log_radon'): array(name, data[name])
        body = f'let beta=x[{j}]; let mean=x[{j+1}]; let la=x[{j+2}]; let ly=x[{j+3}]; let mut lp=la+ly-0.5*(beta/10.0).powi(2)-0.5*(mean/10.0).powi(2)-0.5*(2.0*la).exp()-0.5*(2.0*ly).exp(); for i in 0..{j} {{ lp -= 0.5*x[i]*x[i]; }} for i in 0..log_radon.len() {{ let mu=mean+la.exp()*x[county_idx[i] as usize-1]+floor_measure[i]*beta; lp += norm(log_radon[i],mu,ly); }} lp'
    elif model == 'hmm_example':
        if data['K'] != 2: raise ValueError('this port requires K=2')
        dimension = 4; array('y', data['y'])
        body = 'let z0=std::f64::consts::SQRT_2*x[0]; let z1=std::f64::consts::SQRT_2*x[1]; let m0=x[2].exp(); let m1=m0+x[3].exp(); let mut lp=x[2]+x[3]+ls(z0)+ls(-z0)+ls(z1)+ls(-z1)+norm(m0,3.0,0.0)+norm(m1,10.0,0.0); let mut a=norm(y[0],m0,0.0); let mut b=norm(y[0],m1,0.0); for i in 1..y.len() { let c=lse(a+ls(z0),b+ls(z1))+norm(y[i],m0,0.0); b=lse(a+ls(-z0),b+ls(-z1))+norm(y[i],m1,0.0); a=c; } lp += lse(a,b); lp'
    else:
        raise ValueError(f'no native Rust translation for {case}')
    return dimension, HELPERS + '\n'.join(declarations) + '\n' + body


def generate(output):
    output.mkdir(parents=True, exist_ok=False)
    corpus = json.loads((ROOT/'tools/bench/corpus/pdb-v3.json').read_text())
    records = []
    for i, case in enumerate(corpus['cases']):
        data = ROOT/case['directory']/'data.json'
        dimension, body = rust_port(case['id'], json.loads(data.read_text()))
        path = output/f'model_{i}.rs'; path.write_text(body)
        records.append(dict(case=case['id'], module=f'model_{i}', dimension=dimension,
                            primal=path.name, primal_sha256=file_hash(path),
                            data_sha256=file_hash(data), stan_sha256=case['files']['model.stan']))
    write_new(output/'models.json', dict(schema_version=1, records=records,
        coordinate_system='Stan 2.38 unconstrained, Jacobian once; density constants may differ',
        modeling='ModelBuilder identity coordinates + explicit custom Rust primal; no analytic VJP execution'))
    return records
