//! Same mathematical models through native kernels and actual generated AD.
//! Requires pinned nightly/autodiff flags and release fat LTO, as do the tests.
#[allow(dead_code)]
#[path = "../models.rs"]
mod models;
use alea::{
    capability::PotentialHessianDerivative,
    model::assembly::CompiledModel,
    target::{LogDensityGradient, TargetSpace},
};
use alea_enzyme_conformance as generated;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

// Tool-local receipts, not another fixture registry or a production planner.
// Inspect the existing fixtures/exporter; never infer execution from Cargo flags.
#[derive(Clone, Copy)]
struct Route {
    first: &'static str,
    compiler: &'static str,
    checkpointed: bool,
}
impl Route {
    fn inspect(model: &CompiledModel) -> Self {
        let tensor = model.graph().any(|node| {
            node.active
                && node.primitive.is_some_and(|p| {
                    p.descriptor::<alea::model::tensor::TensorFactor>()
                        .is_some()
                })
        });
        let compiler = if tensor { "L" } else { "C" };
        let explicit = model
            .rust_first_order_gradient(true)
            .expect("baseline source planning");
        Self {
            first: if explicit.is_some() { "H" } else { compiler },
            compiler,
            checkpointed: explicit.is_some_and(|(_, _, checkpointed)| checkpointed),
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
        let admission = if requested == "N" {
            "N"
        } else if order == "value" {
            "none"
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
                        "generic native CompiledModel has no curvature trait",
                    )
                } else if requested != "N" && requested != self.first {
                    (
                        "unsupported",
                        "current exporter cannot honor this pure first-order route; no forced-route control",
                    )
                } else {
                    (
                        "not-run",
                        "planning only; no execution receipt; value-only is not isolated and compiled bench requires PotentialHessianDerivative",
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
        )
    };
}

fn compare<T: PotentialHessianDerivative>(
    c: &mut Criterion,
    name: &str,
    native: CompiledModel,
    compiled: T,
    fixture: &str,
    flags: (bool, bool),
) {
    let route = Route::inspect(&native);
    assert_eq!(flags, (route.first == "H", route.checkpointed));
    let native = native.into_target().unwrap();
    let n = native.dimension();
    assert_eq!(n, compiled.dimension());
    let q: Vec<_> = (0..n).map(|i| 0.1 + i as f64 * 0.01).collect();
    let mut a = vec![0.; n];
    let mut b = a.clone();
    let va = native
        .logp_grad(&q, &mut a)
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
        .logp_grad(&q, &mut b)
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
    let mut group = c.benchmark_group(format!("native_compiled/{name}"));
    group.throughput(Throughput::Elements(1)); // Complete value AND gradient.
    group.bench_function("native_reused", |b| {
        b.iter(|| black_box(native.logp_grad(black_box(&q), black_box(&mut a)).unwrap()))
    });
    group.bench_function("compiled_reused", |b| {
        b.iter(|| {
            black_box(
                compiled
                    .logp_grad(black_box(&q), black_box(&mut a))
                    .unwrap(),
            )
        })
    });
    let direction = vec![0.2; n];
    compiled
        .potential_hvp(&q, &direction, &mut a)
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
                .potential_hvp(black_box(&q), black_box(&direction), black_box(&mut a))
                .unwrap();
            black_box(&a);
        })
    });
    let mut h = vec![0.; n * n];
    let mut d = vec![0.; n * n * n];
    compiled
        .potential_hessian_and_derivatives(&q, &mut h, &mut d)
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
                    black_box(&q),
                    black_box(&mut h),
                    black_box(&mut d),
                )
                .unwrap();
            black_box((&h, &d));
        })
    });
    group.finish();
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
            eprintln!("ALEA_SOURCE_ROUTE {} explicit_first_order={} checkpointed={}",
                stringify!($module), generated::$module::SOURCE_EXPLICIT_FIRST_ORDER,
                generated::$module::SOURCE_CHECKPOINTED);
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
