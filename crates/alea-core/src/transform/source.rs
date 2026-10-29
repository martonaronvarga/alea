//! Scalar source lowering avoids byte-copy/type-analysis ambiguities in nested AD.
use super::{Kind, Transform};
use std::fmt::Write;

impl Transform {
    pub(crate) fn rust_validation(&self, density: bool) -> Option<String> {
        let Kind::Matrix {
            n,
            correlation,
            covariance: true,
        } = self.kind
        else {
            return None;
        };
        Some(format!(
            r#"
            let mut x = [0.0_f64; {output}];
            let mut scratch = [0.0_f64; {scratch}];
            if ::alea::transform::Transform::{}({n}).expect("validated transform")
                .{method}(q, &mut x, &mut scratch).is_err() {{ return false; }}
        "#,
            if correlation {
                "correlation"
            } else {
                "covariance"
            },
            method = if density {
                "constrain"
            } else {
                "constrain_values"
            },
            output = self.output,
            scratch = self.scratch
        ))
    }

    pub(crate) fn rust_primal(&self, input: usize, output: usize, density: bool) -> String {
        let mut s = String::new();
        for i in 0..self.output {
            writeln!(s, "let x{};", output + i).unwrap();
        }
        // Keep every transform's temporaries local, while publishing its scalar
        // coordinates. Rust proves each coordinate is assigned exactly once.
        if density {
            s.push_str("jacobian += { let mut jac = 0.0_f64;\n");
        } else {
            s.push_str("{\n");
        }
        match self.kind {
            Kind::Identity => {
                for i in 0..self.input {
                    writeln!(s, "x{} = q[{}];", output + i, input + i).unwrap();
                }
            }
            Kind::Lower(bound) | Kind::Upper(bound) => {
                let op = if matches!(self.kind, Kind::Lower(_)) {
                    "+"
                } else {
                    "-"
                };
                for i in 0..self.input {
                    let x = output + i;
                    let q = input + i;
                    writeln!(s, "x{x} = {bound:?}_f64 {op} q[{q}].exp();\nif x{x} == {bound:?}_f64 {{ return f64::NAN; }}").unwrap();
                    jacobian(&mut s, &format!("q[{q}]"), density);
                }
            }
            Kind::Interval {
                lower,
                upper,
                width,
            } => {
                for i in 0..self.input {
                    let x = output + i;
                    sigmoid(&mut s, &format!("q[{}]", input + i), i, density);
                    writeln!(s, "x{x} = if u{i} >= 0.0 {{ {upper:?}_f64 - {width:?}_f64*c{i} }} else {{ {lower:?}_f64 + {width:?}_f64*v{i} }};\nif x{x} <= {lower:?}_f64 || x{x} >= {upper:?}_f64 {{ return f64::NAN; }}").unwrap();
                    jacobian(&mut s, &format!("{width:?}_f64.ln() + lp{i}"), density);
                }
            }
            Kind::Ordered { positive } => {
                for i in 0..self.input {
                    let x = output + i;
                    let q = input + i;
                    if i == 0 && !positive {
                        writeln!(s, "x{x} = q[{q}];").unwrap();
                    } else {
                        let previous = if i == 0 {
                            "0.0_f64".into()
                        } else {
                            format!("x{}", x - 1)
                        };
                        writeln!(s, "x{x} = {previous} + q[{q}].exp();\nif x{x} <= {previous} {{ return f64::NAN; }}").unwrap();
                        jacobian(&mut s, &format!("q[{q}]"), density);
                    }
                }
            }
            Kind::Simplex => {
                s.push_str("let r0 = 1.0_f64;\n");
                for i in 0..self.input {
                    let x = output + i;
                    let next = i + 1;
                    let remaining = self.input - i;
                    sigmoid(
                        &mut s,
                        &format!("q[{}] - {remaining}.0_f64.ln()", input + i),
                        i,
                        density,
                    );
                    jacobian(&mut s, &format!("r{i}.ln() + lp{i}"), density);
                    writeln!(s, "x{x} = r{i}*v{i};\nlet r{next} = r{i}*c{i};\nif x{x} <= 0.0 || r{next} <= 0.0 || v{i} == 1.0 {{ return f64::NAN; }}").unwrap();
                }
                writeln!(s, "x{} = r{};", output + self.input, self.input).unwrap();
            }
            Kind::UnitVector => {
                s.push_str("let mut r2 = -0.0_f64;\n");
                for i in 0..self.input {
                    let q = input + i;
                    writeln!(s, "r2 += q[{q}]*q[{q}];").unwrap();
                }
                s.push_str("let d = 1.0+r2;\n");
                for i in 0..self.input {
                    writeln!(s, "x{} = (2.0/d)*q[{}];", output + i, input + i).unwrap();
                }
                writeln!(
                    s,
                    "x{} = 1.0-2.0/d;\nif !d.is_finite() || x{} == 1.0 {{ return f64::NAN; }}",
                    output + self.input,
                    output + self.input
                )
                .unwrap();
                jacobian(
                    &mut s,
                    &format!("{}.0 * (::std::f64::consts::LN_2-d.ln())", self.input),
                    density,
                );
            }
            Kind::Matrix {
                n,
                correlation,
                covariance,
            } => matrix_primal(&mut s, n, correlation, covariance, input, output, density),
        }
        for i in 0..self.output {
            writeln!(s, "if !x{}.is_finite() {{ return f64::NAN; }}", output + i).unwrap();
        }
        // An identity/zero-size chart has no Jacobian update.
        if density && (matches!(self.kind, Kind::Identity) || self.input == 0) {
            s = s.replace("let mut jac =", "let jac =");
        }
        if density {
            s.push_str("jac };\n");
        } else {
            s.push_str("};\n");
        }
        s
    }
}

fn jacobian(s: &mut String, expression: &str, density: bool) {
    if density {
        writeln!(s, "jac += {expression};").unwrap();
    }
}

fn sigmoid(s: &mut String, expression: &str, i: usize, density: bool) {
    // Keep complements stable; objective primals do not evaluate unused log terms.
    if density {
        // Each branch is smooth through zero; no abs/max cusp enters higher AD.
        writeln!(s, "let u{i} = {expression};\nlet (v{i}, c{i}, lp{i}) = if u{i} >= 0.0 {{ let t = (-u{i}).exp(); (1.0/(1.0+t), t/(1.0+t), -u{i}-2.0*t.ln_1p()) }} else {{ let t = u{i}.exp(); (t/(1.0+t), 1.0/(1.0+t), u{i}-2.0*t.ln_1p()) }};").unwrap();
    } else {
        writeln!(s, "let u{i} = {expression};\nlet (v{i}, c{i}) = if u{i} >= 0.0 {{ let t = (-u{i}).exp(); (1.0/(1.0+t), t/(1.0+t)) }} else {{ let t = u{i}.exp(); (t/(1.0+t), 1.0/(1.0+t)) }};").unwrap();
    }
}

fn matrix_primal(
    s: &mut String,
    n: usize,
    correlation: bool,
    covariance: bool,
    input: usize,
    output: usize,
    density: bool,
) {
    if covariance && !correlation {
        jacobian(s, &format!("{n}.0 * ::std::f64::consts::LN_2"), density);
    }
    let mut k = input;
    for i in 0..n {
        if correlation {
            writeln!(s, "let r{i}_0 = 1.0_f64;").unwrap();
        }
        for j in 0..i {
            if correlation {
                let next = j + 1;
                let coefficient = if covariance { n - j } else { i - j + 1 };
                writeln!(s, "let z{i}_{j} = q[{k}].tanh();\nlet s{i}_{j} = (1.0-z{i}_{j})*(1.0+z{i}_{j});\nif s{i}_{j} <= 0.0 {{ return f64::NAN; }}\nlet l{i}_{j} = r{i}_{j}*z{i}_{j};\nlet r{i}_{next} = r{i}_{j}*s{i}_{j}.sqrt();").unwrap();
                jacobian(
                    s,
                    &format!("0.5 * {coefficient}.0 * s{i}_{j}.ln()"),
                    density,
                );
            } else {
                writeln!(s, "let l{i}_{j} = q[{k}];").unwrap();
            }
            k += 1;
        }
        if correlation {
            writeln!(s, "let l{i}_{i} = r{i}_{i};").unwrap();
        } else {
            let coefficient = if covariance { n + 1 - i } else { 1 };
            writeln!(s, "let l{i}_{i} = q[{k}].exp();").unwrap();
            jacobian(s, &format!("{coefficient}.0 * q[{k}]"), density);
            k += 1;
        }
        writeln!(
            s,
            "if l{i}_{i} <= 0.0 || !l{i}_{i}.is_finite() {{ return f64::NAN; }}"
        )
        .unwrap();
    }
    for i in 0..n {
        for j in 0..n {
            if covariance {
                if j > i {
                    continue;
                }
                let a = output + i * n + j;
                let b = output + j * n + i;
                if correlation && i == j {
                    writeln!(s, "x{a} = 1.0_f64;").unwrap();
                } else {
                    writeln!(s, "let mut v{i}_{j} = -0.0_f64;").unwrap();
                    for c in 0..=j {
                        writeln!(s, "v{i}_{j} += l{i}_{c}*l{j}_{c};").unwrap();
                    }
                    writeln!(s, "x{a} = v{i}_{j};").unwrap();
                    if a != b {
                        writeln!(s, "x{b} = v{i}_{j};").unwrap();
                    }
                }
            } else {
                let value = if j <= i {
                    format!("l{i}_{j}")
                } else {
                    "0.0_f64".into()
                };
                writeln!(s, "x{} = {value};", output + i * n + j).unwrap();
            }
        }
    }
}
