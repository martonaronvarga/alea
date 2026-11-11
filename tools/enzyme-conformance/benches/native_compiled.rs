//! Same mathematical models through native kernels and actual generated AD.
//! Requires pinned nightly/autodiff flags and release fat LTO, as do the tests.
#[allow(dead_code)]
#[path = "../models.rs"]
mod models;
use alea::{
    capability::{PotentialHessianDerivative, PotentialHessianVector},
    model::assembly::NativeModel,
    target::{LogDensityGradient, TargetSpace},
};
use alea_enzyme_conformance as generated;
use criterion::{
    BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main, measurement::WallTime,
};
use std::hint::black_box;

// The pre-I1.1 exporter omits the separate admission flag entirely.
const HAS_TENSOR_VALIDATION: bool =
    tensor_validation_flag(include_str!(concat!(env!("OUT_DIR"), "/polynomial.rs"))).is_some();

const fn tensor_validation_flag(source: &str) -> Option<bool> {
    let prefix = b"pub const SOURCE_TENSOR_SCORE_VALIDATION: bool = ";
    let source = source.as_bytes();
    let mut i = 0;
    while i + prefix.len() < source.len() {
        let mut j = 0;
        while j < prefix.len() && source[i + j] == prefix[j] {
            j += 1;
        }
        if j == prefix.len() {
            return Some(source[i + j] == b't');
        }
        i += 1;
    }
    None
}

// The benchmark uses the generated target through its first-order interface.
struct FirstOrder<'a, T>(&'a T);

impl<T: TargetSpace> TargetSpace for FirstOrder<'_, T> {
    type Error = T::Error;

    fn dimension(&self) -> usize {
        self.0.dimension()
    }
}

impl<T: LogDensityGradient> LogDensityGradient for FirstOrder<'_, T> {
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        self.0.logp_grad(q, g)
    }
}

// Tool-local receipts, not another fixture registry or a production planner.
// Inspect the existing fixtures/exporter; never infer execution from Cargo flags.
#[derive(Clone, Copy)]
struct Route {
    first: &'static str,
    compiler: &'static str,
    checkpointed: bool,
    tensor_validation: bool,
}
impl Route {
    fn inspect(model: &NativeModel) -> Self {
        let tensor = model.graph().any(|node| {
            node.active
                && node
                    .primitive
                    .is_some_and(|p| p.signature().name == "tensor_region")
        });
        let compiler = if tensor { "L" } else { "C" };
        let explicit = model
            .source_first_order(alea::model::assembly::SourceMode::Density)
            .expect("baseline source planning");
        Self {
            first: if explicit.is_some() { "H" } else { compiler },
            compiler,
            tensor_validation: HAS_TENSOR_VALIDATION && tensor && explicit.is_none(),
            checkpointed: explicit.is_some_and(|first| first.checkpointed),
        }
    }
    fn emit(self, fixture: &str, requested: &str, order: &str, status: &str, reason: &str) {
        let resolved = if requested == "N" {
            if matches!(order, "hvp" | "third") {
                "none"
            } else {
                "N"
            }
        } else if matches!(order, "value" | "hvp" | "third") {
            self.compiler
        } else {
            self.first
        };
        let reason = if requested != "N" && order != "value" && self.tensor_validation {
            format!(
                "{reason}; extra explicit tensor score validation, compiler derivative unchanged"
            )
        } else {
            reason.to_owned()
        };
        let admission = if requested == "N" {
            "N"
        } else if order == "value" {
            "none"
        } else if self.tensor_validation {
            "H"
        } else {
            self.first
        };
        eprintln!(
            "ALEA_BASELINE_ROUTE_V1\t{fixture}\t{requested}\t{resolved}\t{order}\tfixture-default\trust\t{admission}\t{}\t{status}\t{reason}",
            requested != "N" && self.checkpointed
        );
    }
    fn inventory(self, fixture: &str) {
        for requested in ["N", "C", "L", "H"] {
            for order in ["value", "first", "hvp", "third"] {
                let (status, reason) = if requested == "N" && matches!(order, "hvp" | "third") {
                    (
                        "unsupported",
                        "generic native NativeModel has no curvature trait",
                    )
                } else if requested != "N" && requested != self.first {
                    (
                        "unsupported",
                        "current exporter cannot honor this pure first-order route; no forced-route control",
                    )
                } else {
                    (
                        "not-run",
                        "planning only; no execution receipt; value-only is not isolated",
                    )
                };
                self.emit(fixture, requested, order, status, reason);
            }
        }
    }
}
macro_rules! flags {
    ($module:ident) => {
        (
            generated::$module::SOURCE_EXPLICIT_FIRST_ORDER,
            generated::$module::SOURCE_CHECKPOINTED,
            tensor_validation_flag(include_str!(concat!(
                env!("OUT_DIR"),
                "/",
                stringify!($module),
                ".rs"
            )))
            .unwrap_or(false),
            include_str!(concat!(env!("OUT_DIR"), "/", stringify!($module), ".rs")),
        )
    };
}

fn compare<T: PotentialHessianDerivative>(
    c: &mut Criterion,
    name: &str,
    native: NativeModel,
    compiled: T,
    fixture: &str,
    flags: (bool, bool, bool, &str),
) {
    let route = Route::inspect(&native);
    assert_eq!(
        (flags.0, flags.1, flags.2),
        (
            route.first == "H",
            route.checkpointed,
            route.tensor_validation
        )
    );
    use benchmark_support::fixed_work::{self, Fixture};
    let primal = native
        .source_primal(alea::model::assembly::SourceMode::Density)
        .unwrap();
    assert!(
        flags.3.contains(&primal.source),
        "generated fixture does not contain the native model/data/chart"
    );
    let model_id = fixed_work::identity(primal.source.as_bytes());
    let coordinates = fixed_work::identity(format!("{:?}", native.coordinates()).as_bytes());
    let native = native.into_target().unwrap();
    let n = native.dimension();
    assert_eq!(n, compiled.dimension());
    let bank = fixed_work::states(n);
    let identity = Fixture {
        name: fixture,
        model: model_id,
        coordinates,
        query: "density/current-factor-selection",
        algorithm: "fused-logp-grad",
        arithmetic: "fixture-default",
        provider: "rust",
    };
    let native_answers =
        fixed_work::run(&identity, "N", &bank, None, |q, g| native.logp_grad(q, g)).unwrap();
    let first_order = FirstOrder(&compiled);
    let compiled_answers = fixed_work::run(&identity, route.first, &bank, None, |q, g| {
        first_order.logp_grad(q, g)
    })
    .unwrap();
    for (a, b) in native_answers.iter().zip(&compiled_answers) {
        assert!((a - b).abs() <= 2e-10 * (1. + a.abs().max(b.abs())));
    }
    let q = &bank[0];
    let mut group = c.benchmark_group(format!("native_compiled/{name}"));
    group.throughput(Throughput::Elements(1));
    compare_first(
        &mut group,
        &native,
        &FirstOrder(&compiled),
        q,
        route,
        fixture,
    );
    compare_hvp(&mut group, &compiled, q, route, fixture);
    compare_third(&mut group, &compiled, q, route, fixture);
    group.finish();
}

fn compare_first<N: LogDensityGradient, T: LogDensityGradient>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    native: &N,
    compiled: &T,
    q: &[f64],
    route: Route,
    fixture: &str,
) {
    let mut a = vec![0.; q.len()];
    let mut b = a.clone();
    let va = native
        .logp_grad(q, &mut a)
        .inspect_err(|_| {
            route.emit(
                fixture,
                "N",
                "first",
                "failed",
                "native preflight returned an error",
            );
        })
        .unwrap();
    route.emit(
        fixture,
        "N",
        "first",
        "supported",
        "native fused preflight executed; not a value-only timing",
    );
    let vb = compiled
        .logp_grad(q, &mut b)
        .inspect_err(|_| {
            route.emit(
                fixture,
                route.first,
                "first",
                "failed",
                "compiled preflight returned an error",
            );
        })
        .unwrap();
    for (x, y) in std::iter::once((&va, &vb)).chain(a.iter().zip(&b)) {
        let agrees = (x - y).abs() <= 2e-10 * (1. + x.abs().max(y.abs()));
        if !agrees {
            route.emit(
                fixture,
                route.first,
                "first",
                "failed",
                "native/compiled value or gradient disagreement",
            );
            panic!("native/compiled disagreement: {x} != {y}");
        }
    }
    route.emit(fixture, route.first, "first", "supported", "compiled fused preflight executed and matched native; independent oracles live in conformance tests");
    group.bench_function("native_reused", |b| {
        b.iter(|| black_box(native.logp_grad(black_box(q), black_box(&mut a)).unwrap()))
    });
    group.bench_function("compiled_reused", |b| {
        b.iter(|| black_box(compiled.logp_grad(black_box(q), black_box(&mut a)).unwrap()))
    });
}

fn compare_hvp<T: PotentialHessianVector>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    compiled: &T,
    q: &[f64],
    route: Route,
    fixture: &str,
) {
    let direction = vec![0.2; q.len()];
    let mut a = vec![0.; q.len()];
    compiled
        .potential_hvp(q, &direction, &mut a)
        .inspect_err(|_| {
            route.emit(
                fixture,
                route.first,
                "hvp",
                "failed",
                "compiled HVP preflight returned an error",
            );
        })
        .unwrap();
    route.emit(fixture, route.first, "hvp", "supported", "compiled HVP preflight executed; original compiler reverse plus first-order admission, not checkpointed curvature");
    group.bench_function("compiled_potential_hvp", |b| {
        b.iter(|| {
            compiled
                .potential_hvp(black_box(q), black_box(&direction), black_box(&mut a))
                .unwrap();
            black_box(&a);
        })
    });
}

fn compare_third<T: PotentialHessianDerivative>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    compiled: &T,
    q: &[f64],
    route: Route,
    fixture: &str,
) {
    let n = q.len();
    let mut h = vec![0.; n * n];
    let mut d = vec![0.; n * n * n];
    compiled
        .potential_hessian_and_derivatives(q, &mut h, &mut d)
        .inspect_err(|_| {
            route.emit(
                fixture,
                route.first,
                "third",
                "failed",
                "compiled third preflight returned an error",
            )
        })
        .unwrap();
    route.emit(fixture, route.first, "third", "supported", "compiled Hessian/third preflight executed; original compiler reverse plus first-order admission, not checkpointed curvature");
    group.bench_function("compiled_potential_hessian_third", |b| {
        b.iter(|| {
            compiled
                .potential_hessian_and_derivatives(
                    black_box(q),
                    black_box(&mut h),
                    black_box(&mut d),
                )
                .unwrap();
            black_box((&h, &d));
        })
    });
}
fn native_compiled(c: &mut Criterion) {
    // All existing definitions appear, even when this benchmark does not run
    // them. Export availability is NEVER a supported execution status.
    for (name, model) in models::all().unwrap() {
        Route::inspect(&model).inventory(name);
    }
    // Do not label native-backed source pullbacks as pure compiler AD.
    macro_rules! route {
        ($($module:ident),* $(,)?) => {$(
            eprintln!("ALEA_SOURCE_ROUTE {} explicit_first_order={} checkpointed={} tensor_score_validation={}",
                stringify!($module), generated::$module::SOURCE_EXPLICIT_FIRST_ORDER,
                generated::$module::SOURCE_CHECKPOINTED,
                flags!($module).2);
        )*};
    }
    route!(
        polynomial,
        native_normal_large,
        native_logit_large,
        native_poisson_large,
        normal_vector,
        normal_vector_observed,
        tensor_checkpoint_long,
        tensor_retained_long
    );
    const {
        assert!(!generated::polynomial::SOURCE_EXPLICIT_FIRST_ORDER);
    }
    compare(
        c,
        "polynomial_compiler_ad",
        models::polynomial().unwrap(),
        generated::polynomial::target(),
        "polynomial",
        flags!(polynomial),
    );
    compare(
        c,
        "normal129",
        models::native_batch_rows(0, 129).unwrap(),
        generated::native_normal_large::target(),
        "native_normal_large",
        flags!(native_normal_large),
    );
    compare(
        c,
        "logit129",
        models::native_batch_rows(1, 129).unwrap(),
        generated::native_logit_large::target(),
        "native_logit_large",
        flags!(native_logit_large),
    );
    compare(
        c,
        "poisson129",
        models::native_batch_rows(2, 129).unwrap(),
        generated::native_poisson_large::target(),
        "native_poisson_large",
        flags!(native_poisson_large),
    );
    compare(
        c,
        "normal_vector",
        models::normal_vector(false).unwrap(),
        generated::normal_vector::target(),
        "normal_vector",
        flags!(normal_vector),
    );
    compare(
        c,
        "normal_vector_observed",
        models::normal_vector(true).unwrap(),
        generated::normal_vector_observed::target(),
        "normal_vector_observed",
        flags!(normal_vector_observed),
    );
    let mut all = models::all().unwrap();
    let name = "tensor_checkpoint_long";
    let i = all.iter().position(|(n, _)| *n == name).unwrap();
    compare(
        c,
        name,
        all.swap_remove(i).1,
        generated::tensor_checkpoint_long::target(),
        name,
        flags!(tensor_checkpoint_long),
    );
    let name = "tensor_retained_long";
    let i = all.iter().position(|(n, _)| *n == name).unwrap();
    compare(
        c,
        name,
        all.swap_remove(i).1,
        generated::tensor_retained_long::target(),
        name,
        flags!(tensor_retained_long),
    );
}
criterion_group! {name=benches; config=benchmark_support::criterion(); targets=native_compiled}
#[path = "../../../crates/benches/support.rs"]
mod benchmark_support;
criterion_main!(benches);
