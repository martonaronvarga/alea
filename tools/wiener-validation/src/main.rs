//! Deterministic replay of the Stan Math Wiener support grid through public Alea.
use alea::distributions::wiener::{
    Boundary, Wiener4, Wiener4Params, Wiener5, Wiener5Params, Wiener7, Wiener7Params,
    WienerObservation, WienerOptions,
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    error::Error,
    fs::{self, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
    time::Instant,
};

type Case = [f64; 9];
fn axes(full: bool) -> [&'static [f64]; 9] {
    if full {
        [
            &[1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1., 3., 10., 50.],
            &[0.1, 0.3, 0.7, 1., 2., 4., 8.],
            &[0., 1e-6, 0.05, 0.2, 1., 5.],
            &[
                1e-4, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999,
            ],
            &[-30., -10., -3., -1., 0., 1., 3., 10., 30.],
            &[0., 1e-8, 0.01, 0.1, 0.5, 1., 5.],
            &[0., 1e-8, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.9],
            &[0., 1e-8, 0.001, 0.01, 0.05, 0.2, 1., 5.],
            &[1e-4, 1e-6, 1e-8],
        ]
    } else {
        [
            &[1e-3, 0.05, 0.5, 3.],
            &[0.5, 1., 3.],
            &[0., 0.2],
            &[0.1, 0.5, 0.9],
            &[-3., 0., 3.],
            &[0., 0.3, 1.],
            &[0., 0.05, 0.18],
            &[0., 0.05, 0.5],
            &[1e-4, 1e-6],
        ]
    }
}
fn walk(axes: &[&[f64]; 9], level: usize, case: &mut Case, visit: &mut impl FnMut(Case)) {
    if level == 9 {
        visit(*case);
        return;
    }
    for &value in axes[level] {
        case[level] = value;
        // This pruning preserves the original valid-case index exactly.
        if level == 6 && (case[3] - value / 2. <= 0. || case[3] + value / 2. >= 1.) {
            continue;
        }
        walk(axes, level + 1, case, visit);
    }
}
fn evaluate(c: Case, fused: bool) -> Result<(f64, Vec<f64>), String> {
    let [gap, a, tau, beta, v, sd, sw, st, eps] = c;
    let obs = WienerObservation {
        rt: tau + gap,
        boundary: Boundary::Upper,
    };
    let options = WienerOptions::new(eps);
    let result = if sd == 0. && sw == 0. && st == 0. {
        let p = Wiener4Params::from_array([a, tau, beta, v]);
        if fused {
            Wiener4.try_fused(&obs, &p, options)
        } else {
            Wiener4.try_log_prob(&obs, &p, options)
        }
        .map(|e| {
            (
                e.log_prob,
                if fused {
                    e.grad_array().to_vec()
                } else {
                    vec![]
                },
            )
        })
    } else if sw == 0. && st == 0. {
        let p = Wiener5Params::from_array([a, tau, beta, v, sd]);
        if fused {
            Wiener5.try_fused(&obs, &p, options)
        } else {
            Wiener5.try_log_prob(&obs, &p, options)
        }
        .map(|e| {
            (
                e.log_prob,
                if fused {
                    e.grad_array().to_vec()
                } else {
                    vec![]
                },
            )
        })
    } else {
        let p =
            Wiener7Params::with_params(a, tau, beta, v, sw, st, sd).map_err(|e| e.to_string())?;
        if fused {
            Wiener7.try_fused(&obs, &p, options)
        } else {
            Wiener7.try_log_prob(&obs, &p, options)
        }
        .map(|e| {
            (
                e.log_prob,
                if fused {
                    e.grad_array().to_vec()
                } else {
                    vec![]
                },
            )
        })
    };
    let (lp, g) = result.map_err(|e| e.to_string())?;
    if !lp.is_finite() || g.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite successful evaluation".into());
    }
    Ok((lp, g))
}
fn new_file(path: &Path) -> std::io::Result<BufWriter<std::fs::File>> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map(BufWriter::new)
}
fn replay(input: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    let input = BufReader::new(std::fs::File::open(input)?);
    let mut output = new_file(output)?;
    let mut counts = BTreeMap::<String, usize>::new();
    for line in input.lines() {
        let row: serde_json::Value = serde_json::from_str(&line?)?;
        let case: Case = serde_json::from_value(row["case"].clone())?;
        let index = row["index"].as_u64().ok_or("missing input index")?;
        let kind = row["kind"].as_str().ok_or("missing evaluation kind")?;
        let fused = match kind {
            "density" => false,
            "fused" => true,
            _ => return Err("unknown evaluation kind".into()),
        };
        let result = match evaluate(case, fused) {
            Ok((lp, scores)) => {
                *counts.entry("finite".into()).or_default() += 1;
                json!({"index":index,"case":case,"kind":kind,"status":"finite","log_prob":lp,"scores":scores})
            }
            Err(error) => {
                *counts.entry("error".into()).or_default() += 1;
                json!({"index":index,"case":case,"kind":kind,"status":"error","error":error})
            }
        };
        writeln!(output, "{result}")?;
    }
    output.flush()?;
    println!(
        "{}",
        json!({"counts":counts,"scope":"replay of supplied inputs only"})
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 4 && args[1] == "replay" {
        return replay(Path::new(&args[2]), Path::new(&args[3]));
    }
    if args.len() != 6 {
        return Err("usage: alea-wiener-validation smoke|full SHARD SHARDS LIMIT OUTPUT_DIRECTORY (LIMIT=0: all)".into());
    }
    let full = match args[1].as_str() {
        "full" => true,
        "smoke" => false,
        _ => return Err("unknown profile".into()),
    };
    let shard: usize = args[2].parse()?;
    let shards: usize = args[3].parse()?;
    let limit: usize = args[4].parse()?;
    if shards == 0 || shard >= shards {
        return Err("invalid shard index/count".into());
    }
    let output = Path::new(&args[5]);
    fs::create_dir(output)?;
    let mut failures = new_file(&output.join("failures.jsonl"))?;
    let mut progress = new_file(&output.join("progress.jsonl"))?;
    let mut summary = new_file(&output.join("summary.json"))?;
    let started = Instant::now();
    let mut global = 0usize;
    let mut processed = 0usize;
    let mut counts = BTreeMap::<String, usize>::new();
    let mut io_error = None;
    let mut visit = |c: Case| {
        let index = global;
        global += 1;
        if index % shards != shard || (limit > 0 && processed >= limit) || io_error.is_some() {
            return;
        }
        processed += 1;
        for fused in [false, true] {
            let kind = if fused { "fused" } else { "density" };
            match evaluate(c, fused) {
                Ok(_) => {
                    *counts.entry(format!("{kind}_finite")).or_default() += 1;
                }
                Err(error) => {
                    *counts.entry(format!("{kind}_error")).or_default() += 1;
                    if let Err(e) = writeln!(
                        failures,
                        "{}",
                        json!({"index":index,"case":c,"kind":kind,"error":error})
                    ) {
                        io_error = Some(e);
                    }
                }
            }
        }
        if processed.is_multiple_of(1000) {
            let record = json!({"processed":processed,"last_index":index,"seconds":started.elapsed().as_secs_f64(),"counts":counts});
            if let Err(e) = writeln!(progress, "{record}")
                .and_then(|()| progress.flush())
                .and_then(|()| failures.flush())
            {
                io_error = Some(e);
            }
        }
    };
    walk(&axes(full), 0, &mut [0.; 9], &mut visit);
    if let Some(e) = io_error {
        return Err(e.into());
    }
    let assigned = global / shards + usize::from(shard < global % shards);
    let record = json!({"profile":args[1],"shard":shard,"shards":shards,"limit":limit,"global_cases":global,"assigned":assigned,"processed":processed,"complete":processed==assigned,"counts":counts,"seconds":started.elapsed().as_secs_f64(),"scope":"finite-output coverage; not an accuracy certificate; both density and fused evaluated independently"});
    writeln!(summary, "{record}")?;
    summary.flush()?;
    failures.flush()?;
    progress.flush()?;
    println!("{record}");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_grid_denominator_matches_original_atlasz_report() {
        let a = axes(true);
        let valid_starts = a[3]
            .iter()
            .flat_map(|w| a[6].iter().map(move |sw| (w, sw)))
            .filter(|(w, sw)| **w - **sw / 2. > 0. && **w + **sw / 2. < 1.)
            .count();
        let count = valid_starts
            * [0, 1, 2, 4, 5, 7, 8]
                .map(|i| a[i].len())
                .iter()
                .product::<usize>();
        assert_eq!(count, 42_611_184);
    }
    #[test]
    fn enumeration_preserves_valid_order_and_shard_partition() {
        let mut count = 0;
        let mut shards = [0; 7];
        let mut last = None;
        walk(&axes(false), 0, &mut [0.; 9], &mut |c| {
            assert!(c[3] - c[6] / 2. > 0. && c[3] + c[6] / 2. < 1.);
            if count == 0 {
                assert_eq!(c, [0.001, 0.5, 0., 0.1, -3., 0., 0., 0., 1e-4]);
            }
            shards[count % 7] += 1;
            count += 1;
            last = Some(c);
        });
        assert_eq!(shards.iter().sum::<usize>(), count);
        assert!(shards.iter().max().unwrap() - shards.iter().min().unwrap() <= 1);
        assert_eq!(last.unwrap(), [3., 3., 0.2, 0.9, 3., 1., 0.18, 0.5, 1e-6]);
    }
}
