//! Scalar source lowering avoids byte-copy/type-analysis ambiguities in nested AD.
use super::{Kind, Transform};
use std::fmt::Write;

impl Transform {
    pub(crate) fn rust_validation(&self) -> Option<String> {
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
            if ::alea_core::transform::Transform::{}({n}).expect("validated transform")
                .constrain(q, &mut x, &mut scratch).is_err() {{ return false; }}
        "#,
            if correlation {
                "correlation"
            } else {
                "covariance"
            },
            output = self.output,
            scratch = self.scratch
        ))
    }

    pub(crate) fn rust_primal(&self, input: usize, output: usize) -> String {
        let mut s = String::new();
        for i in 0..self.output {
            writeln!(s, "let x{};", output + i).unwrap();
        }
        // Keep every transform's temporaries local, while publishing its scalar
        // coordinates. Rust proves each coordinate is assigned exactly once.
        s.push_str("jacobian += { let mut jac = 0.0_f64;\n");
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
                    writeln!(s, "x{x} = {bound:?}_f64 {op} q[{q}].exp();\nif x{x} == {bound:?}_f64 {{ return f64::NAN; }}\njac += q[{q}];").unwrap();
                }
            }
            Kind::Interval {
                lower,
                upper,
                width,
            } => {
                for i in 0..self.input {
                    let x = output + i;
                    sigmoid(&mut s, &format!("q[{}]", input + i), i);
                    writeln!(s, "x{x} = if u{i} >= 0.0 {{ {upper:?}_f64 - {width:?}_f64*c{i} }} else {{ {lower:?}_f64 + {width:?}_f64*v{i} }};\nif x{x} <= {lower:?}_f64 || x{x} >= {upper:?}_f64 {{ return f64::NAN; }}\njac += {width:?}_f64.ln() + lp{i};").unwrap();
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
                        writeln!(s, "x{x} = {previous} + q[{q}].exp();\nif x{x} <= {previous} {{ return f64::NAN; }}\njac += q[{q}];").unwrap();
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
                    );
                    writeln!(s, "jac += r{i}.ln() + lp{i};\nx{x} = r{i}*v{i};\nlet r{next} = r{i}*c{i};\nif x{x} <= 0.0 || r{next} <= 0.0 || v{i} == 1.0 {{ return f64::NAN; }}").unwrap();
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
                writeln!(s, "x{} = 1.0-2.0/d;\nif !d.is_finite() || x{} == 1.0 {{ return f64::NAN; }}\njac += {}.0 * (::std::f64::consts::LN_2-d.ln());", output+self.input, output+self.input, self.input).unwrap();
            }
            Kind::Matrix {
                n,
                correlation,
                covariance,
            } => matrix_primal(&mut s, n, correlation, covariance, input, output),
        }
        for i in 0..self.output {
            writeln!(s, "if !x{}.is_finite() {{ return f64::NAN; }}", output + i).unwrap();
        }
        // An identity/zero-size chart has no Jacobian update.
        if matches!(self.kind, Kind::Identity) || self.input == 0 {
            s = s.replace("let mut jac =", "let jac =");
        }
        s.push_str("jac };\n");
        s
    }
}

fn sigmoid(s: &mut String, expression: &str, i: usize) {
    // Each branch is smooth through zero; no abs/max cusp enters higher AD.
    writeln!(s, "let u{i} = {expression};\nlet (v{i}, c{i}, lp{i}) = if u{i} >= 0.0 {{ let t = (-u{i}).exp(); (1.0/(1.0+t), t/(1.0+t), -u{i}-2.0*t.ln_1p()) }} else {{ let t = u{i}.exp(); (t/(1.0+t), 1.0/(1.0+t), u{i}-2.0*t.ln_1p()) }};").unwrap();
}

fn matrix_primal(
    s: &mut String,
    n: usize,
    correlation: bool,
    covariance: bool,
    input: usize,
    output: usize,
) {
    if covariance && !correlation {
        writeln!(s, "jac += {n}.0 * ::std::f64::consts::LN_2;").unwrap();
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
                writeln!(s, "let z{i}_{j} = q[{k}].tanh();\nlet s{i}_{j} = (1.0-z{i}_{j})*(1.0+z{i}_{j});\nif s{i}_{j} <= 0.0 {{ return f64::NAN; }}\nlet l{i}_{j} = r{i}_{j}*z{i}_{j};\nlet r{i}_{next} = r{i}_{j}*s{i}_{j}.sqrt();\njac += 0.5 * {coefficient}.0 * s{i}_{j}.ln();").unwrap();
            } else {
                writeln!(s, "let l{i}_{j} = q[{k}];").unwrap();
            }
            k += 1;
        }
        if correlation {
            writeln!(s, "let l{i}_{i} = r{i}_{i};").unwrap();
        } else {
            let coefficient = if covariance { n + 1 - i } else { 1 };
            writeln!(
                s,
                "let l{i}_{i} = q[{k}].exp();\njac += {coefficient}.0 * q[{k}];"
            )
            .unwrap();
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
