//! Application checks are independent of the sampler's own target evaluation.
use alea::target::LogDensityGradient;
use alea_downstream_conformance::gaussian_hierarchy::{CASES, DATA, model, report};
use std::{
    io::Write,
    process::{Command, Stdio},
};

fn oracle(q: &[f64; 4]) -> f64 {
    // Independent normalized test density. Not a model/backend implementation.
    let normal = |x: f64, m: f64| -0.5 * (x - m).powi(2) - 0.5 * std::f64::consts::TAU.ln();
    let tau = q[1].exp();
    let mut lp = normal(q[0], 0.) + normal(tau, 0.) + std::f64::consts::LN_2 + q[1];
    for (g, z) in q[2..].iter().enumerate() {
        lp += normal(*z, 0.);
        for y in DATA[g] {
            lp += normal(y, q[0] + tau * z);
        }
    }
    lp
}

#[test]
fn normalized_posterior_and_scores_include_exactly_one_scale_jacobian() {
    let target = model().unwrap().into_target().unwrap();
    for q in [
        [0., 0., 0., 0.],
        [-0.5, -4., 1.2, -0.7],
        [1., 1.5, -0.2, 0.4],
        [0.2, -0.3, 0.5, -0.7],
    ] {
        let mut gradient = [0.; 4];
        let lp = target.logp_grad(&q, &mut gradient).unwrap();
        assert!((lp - oracle(&q)).abs() < 1e-11 * (1. + lp.abs()));
        for i in 0..4 {
            let (mut plus, mut minus) = (q, q);
            plus[i] += 1e-5;
            minus[i] -= 1e-5;
            let fd = (oracle(&plus) - oracle(&minus)) / 2e-5;
            assert!(
                (gradient[i] - fd).abs() < 1e-7 * (1. + fd.abs()),
                "coordinate{i}: {} vs {fd}",
                gradient[i]
            );
        }
    }
}

#[test]
fn actual_native_four_chain_posterior_and_controls_pass_independent_reference() {
    let python = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("python");
    for case in CASES {
        let r = report(case).unwrap();
        let mut child = Command::new("python3").env("PYTHONPATH",&python).arg("-c").arg(r#"
import copy, json, sys
from analysis import unique_object, reject_constant
from gaussian_hierarchy import validate
r=json.load(sys.stdin,object_pairs_hook=unique_object,parse_constant=reject_constant)
s=validate(r)
assert s['planned_chains']==4
assert s['qualified']==(r['case']=='complete')
if r['case']=='complete':
    # Mutate actual native output, not a synthetic passing sampler fixture.
    for mutation in range(6):
        bad=copy.deepcopy(r)
        if mutation==0: bad['summaries']['tau']['mean']+=1
        if mutation==1: bad['summaries']['mu']['mcse_mean']*=10
        if mutation==2: bad['chains'].pop()
        if mutation==3: bad['plan']['seed'][0]=84
        if mutation==4: bad['csv']=bad['csv'].replace('\r\n','\n')
        if mutation==5: bad['manifest']['predictive_units'][0]['integrated_out']=['z0']
        try: validate(bad)
        except ValueError: pass
        else: raise AssertionError(f'accepted corrupted native report {mutation}')
else:
    assert s['completed_chains']==0
    if r['case']=='initial_transform':
        assert all('transform rounded to the boundary' in ' '.join(c['error_chain']) for c in r['chains'])
print(json.dumps(s))
"#).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(&serde_json::to_vec(&r).unwrap())
            .unwrap();
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "case={case}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        println!("{}", String::from_utf8_lossy(&out.stdout));
    }
    assert!(report("unknown").is_err());
    assert_eq!(report("cancelled").unwrap(), report("cancelled").unwrap());
}
