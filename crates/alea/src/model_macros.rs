/// Builds a model with ordinary Rust control flow and typed, fallible declarations.
///
/// Parameter methods receive their binding's name automatically. Arguments after
/// the name are passed unchanged, e.g. `beta: real_vector(&["intercept", "slope"])`.
/// The body must return `()`; `?` propagates [`crate::model::assembly::AssemblyError`]
/// from the macro's own closure, even in a caller that does not return `Result`.
/// No factor, prior, Jacobian, observation loop or parameterization is inferred.
///
/// ```
/// let model = alea::model! { |b|
///     parameters { mean: real(), scale: positive() }
///     {
///         let prior = b.multiply(-1.0, scale)?;
///         b.prior("scale prior", prior)?;
///         let square = b.multiply(mean, mean)?;
///         let mean_prior = b.multiply(-0.5, square)?;
///         b.prior("mean prior up to constant", mean_prior)?;
///     }
/// };
/// assert!(model.is_ok());
/// ```
///
/// For explicit declarations/custom charts use `model!(|b| { ... })`, or use
/// `ModelBuilder` directly. This macro only constructs/compiles a graph; use
/// explicit statistical selection and `into_target`/`into_objective` afterwards.
#[macro_export]
macro_rules! model {
    (|$builder:ident| parameters {
        $($name:ident : $method:ident($($argument:expr),* $(,)?)),* $(,)?
    } $body:block) => {{
        (|| -> ::core::result::Result<$crate::model::assembly::NativeModel, $crate::model::assembly::AssemblyError> {
            let mut $builder = $crate::model::assembly::ModelBuilder::new();
            $(let $name = $builder.$method(::core::stringify!($name) $(, $argument)*)?;)*
            let _: () = $body;
            $builder.compile()
        })()
    }};
    (|$builder:ident| $body:block) => {{
        $crate::model! { |$builder| parameters {} $body }
    }};
}
