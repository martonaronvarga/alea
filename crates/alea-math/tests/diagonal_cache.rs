use alea_math::{
    buffer::OwnedBuffer,
    metric::{DiagonalMetric, EuclideanMetric},
};

#[test]
fn cached_diagonal_actions_preserve_scalar_bits_extremes_shapes_and_allocations() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<DiagonalMetric>();
    for n in 0..=65 {
        let masses: Vec<_> = (0..n)
            .map(|i| {
                [
                    f64::from_bits(1),
                    f64::MIN_POSITIVE,
                    1e-200,
                    0.3,
                    1.,
                    7.,
                    1e200,
                    f64::MAX,
                ][i % 8]
            })
            .collect();
        let metric = DiagonalMetric::new(OwnedBuffer::from_fn(n, |i| masses[i])).unwrap();
        assert_eq!(metric.as_slice(), masses);
        assert_eq!(metric, metric.clone());
        assert_eq!(
            metric.log_det().to_bits(),
            masses.iter().map(|x| x.ln()).sum::<f64>().to_bits()
        );
        let src: Vec<_> = (0..n)
            .map(|i| if i % 2 == 0 { 1e-100 } else { -1e100 })
            .collect();
        let mut out = vec![0.; n];
        metric.velocity(&src, &mut out).unwrap();
        for ((actual, x), mass) in out.iter().zip(&src).zip(&masses) {
            assert_eq!(actual.to_bits(), (x / mass).to_bits());
        }
        metric.sample_momentum(&src, &mut out).unwrap();
        for ((actual, x), mass) in out.iter().zip(&src).zip(&masses) {
            assert_eq!(actual.to_bits(), (x * mass.sqrt()).to_bits());
        }
        let allocation = allocation_counter::measure(|| {
            metric.velocity(&src, &mut out).unwrap();
            metric.sample_momentum(&src, &mut out).unwrap();
            std::hint::black_box(metric.log_det());
        });
        assert_eq!(allocation.count_total, 0);
        let mut wrong = vec![17.; n + 1];
        assert!(metric.velocity(&src, &mut wrong).is_err());
        assert!(metric.sample_momentum(&src, &mut wrong).is_err());
        assert!(wrong.iter().all(|x| *x == 17.));
    }
}
