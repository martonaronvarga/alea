"""Native float64 JAX densities; independently checked against Stan before timing."""
import math


def density(case, data):
    import jax
    import jax.numpy as j
    from .pdb_external import native_density
    original = native_density(case, data, j)
    if original is not None:
        return original
    model = case.split('-', 1)[1]
    sp = j.logaddexp
    ls = lambda z: -sp(0., -z)
    norm = lambda y, m, l: -l-.5*((y-m)*j.exp(-l))**2
    if model == 'Rate_5_model':
        return lambda q: (data['k1']+data['k2']+1)*ls(q[0])+(data['n1']+data['n2']-data['k1']-data['k2']+1)*ls(-q[0])
    if model in ('earn_height', 'logearn_interaction_z', 'blr', 'dogs_log'):
        if model == 'earn_height':
            design = [[1., h] for h in data['height']]; response = data['earn']
        elif model == 'logearn_interaction_z':
            h = data['height']; mean = math.fsum(h)/len(h)
            sd = math.sqrt(math.fsum((v-mean)**2 for v in h)/(len(h)-1))
            design = [[1., (v-mean)/sd, m, m*(v-mean)/sd] for v, m in zip(h, data['male'], strict=True)]
            response = [math.log(v) for v in data['earn']]
        elif model == 'blr':
            design = data['X']; response = data['y']
        else:
            design = []; response = []
            for dog in data['y']:
                avoid = shock = 0
                for y in dog:
                    design.append([avoid, shock]); response.append(y)
                    avoid += 1-y; shock += y
        x = j.asarray(design); y = j.asarray(response); p = x.shape[1]
        def regression(q):
            if model == 'dogs_log':
                z = x@q
                valid = (q[0] > -100.) & (q[0] < 0.) & (q[1] > 0.) & (q[1] < 100.)
                return j.where(valid, j.sum(y*z-j.logaddexp(0., z)), -j.inf)
            lp = q[p]+j.sum(norm(y, x@q[:p], q[p]))
            if model == 'blr': lp -= .5*j.sum((q[:p]/10.)**2)+.5*(j.exp(q[p])/10.)**2
            return lp
        return regression
    if model == 'arK':
        k = data['K']; y = j.asarray(data['y'])
        x = j.asarray([[1.]+[data['y'][i-r-1] for r in range(k)] for i in range(k, len(data['y']))])
        return lambda q: q[-1]-sp(0., 2*(q[-1]-math.log(2.5)))-.5*j.sum((q[:-1]/10.)**2)+j.sum(norm(y[k:], x@q[:-1], q[-1]))
    if model == 'arma11':
        y = j.asarray(data['y'])
        def arma(q):
            first = y[0]-q[0]-q[1]*q[0]
            def step(previous, pair):
                lag, current = pair
                error = current-q[0]-q[1]*lag-q[2]*previous
                return error, norm(error, 0., q[3])
            _, values = jax.lax.scan(step, first, (y[:-1], y[1:]))
            return q[3]-sp(0., 2*(q[3]-math.log(2.5)))-.5*((q[0]/10.)**2+(q[1]/2.)**2+(q[2]/2.)**2)+norm(first, 0., q[3])+j.sum(values)
        return arma
    if model == 'garch11':
        y = j.asarray(data['y']); initial = data['sigma1']**2
        def garch(q):
            a0 = j.exp(q[1]); a1 = jax.nn.sigmoid(q[2]); b1 = jax.nn.sigmoid(-q[2])*jax.nn.sigmoid(q[3])
            def step(variance, pair):
                lag, current = pair
                variance = a0+a1*(lag-q[0])**2+b1*variance
                return variance, norm(current, q[0], .5*j.log(variance))
            _, values = jax.lax.scan(step, initial, (y[:-1], y[1:]))
            return q[1]+ls(q[2])+2*ls(-q[2])+ls(q[3])+ls(-q[3])+norm(y[0], q[0], .5*math.log(initial))+j.sum(values)
        return garch
    if model in ('normal_mixture', 'low_dim_gauss_mix_collapse'):
        y = j.asarray(data['y'])
        if model == 'normal_mixture':
            return lambda q: ls(q[0])+ls(-q[0])-.5*j.sum((q[1:]/10.)**2)+j.sum(j.logaddexp(ls(q[0])+norm(y, q[1], 0.), ls(-q[0])+norm(y, q[2], 0.)))
        return lambda q: 5*(ls(q[4])+ls(-q[4]))+q[2]+q[3]-.5*j.sum((q[:2]/2.)**2)-.5*j.sum((j.exp(q[2:4])/2.)**2)+j.sum(j.logaddexp(ls(q[4])+norm(y, q[0], q[2]), ls(-q[4])+norm(y, q[1], q[3])))
    if model == 'seeds_stanified_model':
        n, trials, x1, x2 = (j.asarray(data[k]) for k in ('n', 'N', 'x1', 'x2'))
        def seeds(q):
            z = q[0]+q[1]*x1+q[3]*x2+q[2]*x1*x2+q[4:-1]
            return q[-1]-sp(0., 2*q[-1])-.5*j.sum(q[:4]**2)+j.sum(norm(q[4:-1], 0., q[-1]))+j.sum(n*z-trials*sp(0., z))
        return seeds
    if model == 'surgical_model':
        success = j.asarray(data['r']); trials = j.asarray(data['n'])
        return lambda q: -.5*(q[0]/1000.)**2-.001*q[1]-.001*j.exp(-q[1])+j.sum(norm(q[2:], q[0], .5*q[1])+success*q[2:]-trials*sp(0., q[2:]))
    if model == 'radon_variable_intercept_noncentered':
        n = data['J']; county = j.asarray(data['county_idx'])-1
        floor = j.asarray(data['floor_measure']); y = j.asarray(data['log_radon'])
        def radon(q):
            mu = q[n+1]+j.exp(q[n+2])*q[:n][county]+q[n]*floor
            return q[n+2]+q[n+3]-.5*((q[n]/10.)**2+(q[n+1]/10.)**2+j.exp(2*q[n+2])+j.exp(2*q[n+3])+j.sum(q[:n]**2))+j.sum(norm(y, mu, q[n+3]))
        return radon
    if model == 'hmm_example':
        if data['K'] != 2: raise ValueError('native HMM requires K=2')
        y = j.asarray(data['y'])
        def hmm(q):
            mu = j.stack((j.exp(q[2]), j.exp(q[2])+j.exp(q[3])))
            z = math.sqrt(2.)*q[:2]
            transitions = j.stack((j.stack((ls(z[0]), ls(-z[0]))), j.stack((ls(z[1]), ls(-z[1])))))
            def step(previous, observation):
                return jax.scipy.special.logsumexp(previous[:, None]+transitions, axis=0)+norm(observation, mu, 0.), None
            last, _ = jax.lax.scan(step, norm(y[0], mu, 0.), y[1:])
            return q[2]+q[3]+j.sum(ls(z)+ls(-z))+norm(mu[0], 3., 0.)+norm(mu[1], 10., 0.)+jax.scipy.special.logsumexp(last)
        return hmm
    raise ValueError(f'no native JAX translation for {case}; no callback fallback')
