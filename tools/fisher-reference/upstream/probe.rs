// Appended to the pinned upstream low_rank.rs in a temporary checkout only.
// Calls the original private compute_update; no fitting code is replaced.
impl LowRankMassMatrixStrategy {
    pub fn emit_alea_reference() {
        fn vector(values: impl Iterator<Item = f64>) -> String {
            values
                .map(|v| format!("{v:.17e}"))
                .collect::<Vec<_>>()
                .join(";")
        }
        println!(
            "# nuts-rs a762aae513bf7bb5ccb27ffc9a938a923fc837ec compute_update; default features disabled"
        );
        println!("case,dim,n,q,score,scales,ridge,threshold,rank,inverse_mass,logdet_mass");
        for dim in [2, 8, 16] {
            for n in [4, 8, 12] {
                let draws = Mat::from_fn(dim, n, |j, i| {
                    (((i + 1) * (j + 3) + i * i) % 17) as f64 / 4.0 - 2.0
                });
                let grads = Mat::from_fn(dim, n, |j, i| {
                    -0.7 * draws[(j, i)]
                        + 0.3 * draws[((j + 1) % dim, i)]
                        + ((i * (j + 1) + 3) % 7) as f64 / 10.0
                });
                for threshold in [1.01, 2.0] {
                    let ridge = 1e-3;
                    let strategy = Self::new(
                        dim,
                        LowRankSettings {
                            gamma: ridge,
                            eigval_cutoff: threshold,
                            ..LowRankSettings::default()
                        },
                    );
                    let (scales, _, values, basis, _) = strategy
                        .compute_update(draws.clone(), grads.clone())
                        .expect("upstream fit failed");
                    let g = Mat::from_fn(dim, dim, |i, j| {
                        let correction: f64 = (0..values.nrows())
                            .map(|k| basis[(i, k)] * (values[k] - 1.0) * basis[(j, k)])
                            .sum();
                        scales[i] * scales[j] * (f64::from(i == j) + correction)
                    });
                    let logdet = -2.0 * scales.iter().map(|s| s.ln()).sum::<f64>()
                        - values.iter().map(|v| v.ln()).sum::<f64>();
                    assert!(logdet.is_finite());
                    println!(
                        "d{dim}_n{n}_t{threshold},{dim},{n},{},{},{},{ridge:.17e},{threshold:.17e},{},{},{logdet:.17e}",
                        vector((0..n * dim).map(|k| draws[(k % dim, k / dim)])),
                        vector((0..n * dim).map(|k| grads[(k % dim, k / dim)])),
                        vector(scales.iter().copied()),
                        values.nrows(),
                        vector((0..dim * dim).map(|k| g[(k / dim, k % dim)]))
                    );
                }
            }
        }
    }
}
