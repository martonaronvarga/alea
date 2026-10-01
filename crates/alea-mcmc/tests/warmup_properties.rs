#![cfg(not(miri))]
use alea_mcmc::adapt::{MetricKind, OnlineCovariance};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig {cases: 128, ..ProptestConfig::default()})]
    #[test]
    fn streaming_covariance_agrees_with_centered_batch(
        points in prop::collection::vec((-100.0_f64..100.0,-100.0_f64..100.0),2..128),
        offset in -1e5_f64..1e5,
    ) {
        let mut e=OnlineCovariance::new(2,MetricKind::Dense).unwrap();
        let data: Vec<_>=points.iter().map(|&(x,y)|[x+offset,y+offset]).collect();
        for x in &data {e.update(x).unwrap();}
        let n=data.len() as f64;
        let mean:[f64;2]=std::array::from_fn(|i|data.iter().map(|x|x[i]).sum::<f64>()/n);
        let actual=e.regularized_covariance().unwrap();
        for col in 0..2 {
            for row in 0..2 {
                let sample=data.iter().map(|x|(x[row]-mean[row])*(x[col]-mean[col])).sum::<f64>()/(n-1.0);
                let expected=n/(n+5.0)*sample+if row==col {0.005/(n+5.0)} else {0.0};
                prop_assert!((actual[row+col*2]-expected).abs()<1e-7*(1.0+expected.abs()));
            }
        }
        prop_assert!(e.metric().is_ok());
    }
}
