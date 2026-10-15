"""Independent JAX AD checks for the shared Rust analytic benchmark targets."""
import subprocess
import sys
from models import CASES,jax,jnp,np,density

for case,(_,d,_) in enumerate(CASES):
    rows=subprocess.check_output([sys.argv[1],str(case),'--probe'],text=True).splitlines()
    for offset,row in zip([-.4,.1,.9],rows,strict=True):
        q=jnp.array([offset+.03*(i%7) for i in range(d)])
        lp,g=jax.value_and_grad(density(case))(q)
        actual=np.fromstring(row,sep=',');expected=np.r_[np.asarray(lp),np.asarray(g)]
        np.testing.assert_allclose(actual,expected,rtol=2e-11,atol=2e-10)
    print(f'{CASES[case][0]}: independent JAX density/score agrees',flush=True)
