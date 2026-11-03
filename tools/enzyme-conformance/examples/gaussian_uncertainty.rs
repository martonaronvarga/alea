//! Qualification consumer, not a general uncertainty/interval implementation.
//! Run with the pinned autodiff shell and --release --features faer.
#[path = "../models.rs"]
mod models;
use alea::{
    capability::PotentialHessianVector,
    distributions::modeling::regression::{
        RegressionData, diagnostics::GaussianIntervalOptions, normal_regression,
    },
    math::information::InformationOptions,
    model::{assembly::ModelBuilder, prediction::NamedOutput},
    optimize::{
        LbfgsOptions, NamedFit, ObjectiveDescription, ObjectiveKind, OptimizationTermination,
        information::observed_information, maximize,
    },
    target::evaluate,
};
use std::{error::Error, io::Write};

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(actual.is_finite() && expected.is_finite());
    assert!(
        (actual - expected).abs() <= tolerance * (1. + expected.abs()),
        "actual={actual:e}, expected={expected:e}"
    );
}

fn qualify<T: PotentialHessianVector>(
    name: &str,
    x_scale: f64,
    y_scale: f64,
    target: T,
    output: &mut impl Write,
) -> Result<(), Box<dyn Error>> {
    let (design, response, offsets) = models::uncertainty_data(x_scale, y_scale);
    let data =
        RegressionData::new(2, design.clone(), response.clone())?.with_offsets(offsets.clone())?;
    let assessment = data.gaussian_assessment(1e-12, 1e-12)?;
    let mut interval_checks = Vec::new();
    for level in [0.90, 0.95, 0.99] {
        let result = data.gaussian_coefficient_intervals(GaussianIntervalOptions {
            confidence_level: level,
            relative_rank_tolerance: 1e-12,
            relative_residual_tolerance: 1e-12,
            quantile_relative_tolerance: 1e-8,
        })?;
        assert_eq!(
            result.assessment.least_squares.coefficients,
            assessment.least_squares.coefficients
        );
        interval_checks.push([
            level,
            result.critical_value,
            result.bounds[0][0],
            result.bounds[0][1],
            result.bounds[1][0],
            result.bounds[1][1],
        ]);
    }
    assert!(assessment.issues.is_empty());
    assert_eq!(assessment.least_squares.numerical_rank, 2);
    assert_eq!(assessment.least_squares.residual_degrees_of_freedom, 4);
    let mut builder = ModelBuilder::new();
    let a = builder.real("intercept")?.expr();
    let b = builder.real("slope")?.expr();
    let sigma = builder.positive("sigma")?;
    let term = normal_regression(&mut builder, &[a, b], &sigma, data)?;
    builder.likelihood("responses", term)?;
    let model = builder.compile()?.select_likelihood()?;
    let names = NamedOutput::new(model.coordinates(), model.layout().clone(), ())?;
    let batch = model.into_objective()?;
    let description = ObjectiveDescription {
        kind: ObjectiveKind::Likelihood,
        measure: "Lebesgue in two unconstrained coefficients and positive sigma".into(),
        model: "independent homoskedastic fixed-design Gaussian, no priors/weights".into(),
        data: name.into(),
        source: "native batch and generated scalar-native qualification".into(),
    };
    let options = LbfgsOptions::new(500, 5000)?.with_score_tolerance(1e-6)?;
    let start = [0., 0., y_scale.ln()];
    let batch_fit = NamedFit::new(
        names,
        maximize(&batch, &start, description.clone(), options.clone())?,
    )?;
    let (_, source_model) = models::all()?
        .into_iter()
        .find(|(model_name, _)| *model_name == format!("uncertainty_{name}"))
        .ok_or("missing generated model metadata")?;
    let generated_names = NamedOutput::new(
        source_model.coordinates(),
        source_model.layout().clone(),
        (),
    )?;
    let generated_fit = NamedFit::new(
        generated_names,
        maximize(&target, &start, description, options)?,
    )?;
    for fit in [&batch_fit, &generated_fit] {
        assert_eq!(
            fit.parameters()
                .coordinates()
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>(),
            ["intercept", "slope", "sigma"]
        );
        assert_eq!(
            fit.result().termination,
            OptimizationTermination::ScoreTolerance
        );
        close(
            fit.estimate("intercept").ok_or("missing intercept")?,
            1.5 * y_scale,
            2e-6,
        );
        close(
            fit.estimate("slope").ok_or("missing slope")?,
            -0.75 * y_scale / x_scale,
            2e-6,
        );
        close(
            fit.estimate("sigma").ok_or("missing sigma")?,
            2_f64.sqrt() * y_scale,
            2e-6,
        );
    }
    // Compare independent native execution paths at the SAME coordinates.
    let q = &generated_fit
        .result()
        .candidate
        .as_ref()
        .ok_or("missing candidate")?
        .position;
    let mut bg = [0.; 3];
    let mut gg = [0.; 3];
    close(
        evaluate(&batch, q, &mut bg)?,
        evaluate(&target, q, &mut gg)?,
        1e-12,
    );
    for (actual, expected) in bg.into_iter().zip(gg) {
        close(actual, expected, 1e-11);
    }
    // Independent model-level diagnostic, not a derivative backend: compare
    // generated potential HVPs with central differences of NATIVE BATCH scores.
    for j in 0..3 {
        let step = 1e-5 * (1. + q[j].abs());
        let mut plus = q.clone();
        let mut minus = q.clone();
        plus[j] += step;
        minus[j] -= step;
        let mut gp = [0.; 3];
        let mut gm = [0.; 3];
        evaluate(&batch, &plus, &mut gp)?;
        evaluate(&batch, &minus, &mut gm)?;
        let mut basis = [0.; 3];
        basis[j] = 1.;
        let mut hvp = [0.; 3];
        target.potential_hvp(q, &basis, &mut hvp)?;
        for i in 0..3 {
            close(hvp[i], -(gp[i] - gm[i]) / (2. * step), 3e-7);
        }
    }
    let info = observed_information(
        &target,
        &generated_fit,
        InformationOptions {
            relative_eigenvalue_cutoff: 1e-12,
            relative_symmetry_tolerance: 64. * f64::EPSILON,
        },
    )?;
    let covariance = &info.constrained_covariance().values;
    let mle_variance = assessment
        .scale_candidate
        .ok_or("missing positive ML scale")?
        .powi(2);
    let df = assessment.least_squares.residual_degrees_of_freedom;
    let residual_variance = assessment.least_squares.residual_norm.powi(2) / df as f64;
    close(mle_variance, 2. * y_scale * y_scale, 1e-12);
    close(residual_variance, 3. * y_scale * y_scale, 1e-12);
    let expected = [
        38. / 105.,
        -6. / (105. * x_scale),
        -6. / (105. * x_scale),
        12. / (105. * x_scale * x_scale),
    ]
    .map(|x| x * y_scale * y_scale);
    let coefficient_covariance = [covariance[0], covariance[1], covariance[3], covariance[4]];
    for (actual, reference) in coefficient_covariance.into_iter().zip(expected) {
        close(actual, reference, 3e-6);
    }
    // This correction applies ONLY to the coefficient block in this full-rank,
    // fixed-design Gaussian model; never multiply sigma's joint variance by n/df.
    let corrected_covariance = coefficient_covariance.map(|x| x * 6. / df as f64);
    close(covariance[8], mle_variance / 12., 3e-6);
    for i in [2, 5, 6, 7] {
        close(covariance[i], 0., 3e-6);
    }
    for (actual, expected) in batch_fit.estimates().ok_or("missing batch estimates")?[..2]
        .iter()
        .zip(&assessment.least_squares.coefficients)
    {
        close(*actual, *expected, 2e-6);
    }
    // All emitted floating values above are explicitly admitted as finite.
    writeln!(
        output,
        "{{\"case\":{name:?},\"rows\":6,\"columns\":2,\"rank\":2,\"df\":{df},\"design\":{design:?},\"response\":{response:?},\"offsets\":{offsets:?},\"batch_estimates\":{:?},\"generated_estimates\":{:?},\"svd_coefficients\":{:?},\"mle_variance\":{mle_variance:?},\"residual_variance\":{residual_variance:?},\"joint_covariance\":{covariance:?},\"coefficient_covariance\":{coefficient_covariance:?},\"corrected_covariance\":{corrected_covariance:?},\"intervals\":{interval_checks:?}}}",
        batch_fit.estimates().ok_or("missing batch estimates")?,
        generated_fit
            .estimates()
            .ok_or("missing generated estimates")?,
        assessment.least_squares.coefficients
    )?;
    Ok(())
}

fn run(output: &mut impl Write) -> Result<(), Box<dyn Error>> {
    use alea_enzyme_conformance::{uncertainty_base, uncertainty_large, uncertainty_small};
    qualify(
        "base",
        1.,
        1.,
        uncertainty_base::composition::objective(),
        output,
    )?;
    qualify(
        "large",
        8.,
        4.,
        uncertainty_large::composition::objective(),
        output,
    )?;
    qualify(
        "small",
        0.125,
        0.25,
        uncertainty_small::composition::objective(),
        output,
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    run(&mut std::io::stdout().lock())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_batch_generated_curvature_and_svd_agree_without_changing_scale_definition() {
        let mut output = Vec::new();
        run(&mut output).unwrap();
        assert_eq!(output.iter().filter(|&&b| b == b'\n').count(), 3);
    }

    #[test]
    fn rank_deficiency_perfect_fit_and_no_residual_df_do_not_qualify_uncertainty() {
        use alea::distributions::modeling::regression::diagnostics::GaussianIssue;
        let cases = [
            (
                vec![1., 1., 1., 1., 1., 1.],
                vec![0., 1., 2.],
                GaussianIssue::NumericalRankDeficiency,
            ),
            (
                vec![1., 0., 1., 1., 1., 2.],
                vec![1., 2., 3.],
                GaussianIssue::VanishingResidual,
            ),
            (
                vec![1., 0., 1., 1.],
                vec![1., 2.],
                GaussianIssue::VanishingResidual,
            ),
        ];
        for (x, y, issue) in cases {
            let saturated = y.len() == 2;
            let a = RegressionData::new(2, x, y)
                .unwrap()
                .gaussian_assessment(1e-12, 1e-12)
                .unwrap();
            assert!(a.issues.contains(&issue));
            if saturated {
                assert_eq!(a.least_squares.residual_degrees_of_freedom, 0);
            }
            if issue == GaussianIssue::VanishingResidual {
                assert!(a.scale_candidate.is_none());
            }
        }
    }
}
