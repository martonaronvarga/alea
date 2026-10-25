//! Same validated modern diagnostics on unconstrained and constrained coordinates.
use alea_runtime::diagnostics::modern;
use serde_json::{Value, json};
use std::{
    error::Error,
    fs,
    io::{BufRead, BufReader},
    path::Path,
};
fn summarize(
    path: &Path,
    chains: usize,
    draws: usize,
    skip: usize,
) -> Result<Value, Box<dyn Error>> {
    let mut lines = BufReader::new(fs::File::open(path)?).lines();
    let header = lines.next().ok_or("missing CSV header")??;
    let names: Vec<_> = header.split(',').skip(skip).map(str::to_owned).collect();
    let mut data = vec![vec![Vec::with_capacity(draws); names.len()]; chains];
    let mut counts = vec![0; chains];
    let mut divergences = 0;
    let mut depth = 0;
    let mut steps = 0;
    for line in lines {
        let line = line?;
        let row: Vec<_> = line.split(',').collect();
        if row.len() != skip + names.len() {
            return Err("ragged CSV".into());
        }
        let c: usize = row[0].parse()?;
        let i: usize = row[1].parse()?;
        if c >= chains || i != counts[c] || i >= draws {
            return Err("lost/duplicate/out-of-range draw".into());
        }
        counts[c] += 1;
        if skip == 8 {
            divergences += row[2].parse::<usize>()?;
            depth += row[3].parse::<usize>()?;
            steps += row[4].parse::<usize>()?;
        }
        for (k, s) in row[skip..].iter().enumerate() {
            let v: f64 = s.parse()?;
            if !v.is_finite() {
                return Err("nonfinite retained coordinate".into());
            }
            data[c][k].push(v);
        }
    }
    if counts.iter().any(|n| *n != draws) {
        return Err("incomplete chain set".into());
    }
    let mut result = vec![];
    for (k, name) in names.iter().enumerate() {
        let views: Vec<_> = data.iter().map(|c| c[k].as_slice()).collect();
        let mut count = 0.;
        let mut mean = 0.;
        let mut m2 = 0.;
        for value in views.iter().flat_map(|chain| chain.iter()) {
            count += 1.;
            let delta = value - mean;
            mean += delta / count;
            m2 += delta * (value - mean);
        }
        let sd = (m2 / (count - 1.)).sqrt();
        let summary = match modern::summarize(&views) {
            Ok(s) => {
                json!({"mean":s.mean,"sd":sd,"mcse":s.mcse_mean,"rhat":s.rhat,"bulk_ess":s.bulk_ess,"tail_ess":s.tail_ess,"error":null})
            }
            Err(e) => {
                json!({"mean":mean,"sd":sd,"mcse":null,"rhat":null,"bulk_ess":null,"tail_ess":null,"error":e.to_string()})
            }
        };
        result.push(json!({"name":name,"summary":summary}));
    }
    Ok(
        json!({"draws":chains*draws,"divergences":if skip==8{Some(divergences)}else{None},"depth_saturations":if skip==8{Some(depth)}else{None},"integration_steps":if skip==8{Some(steps)}else{None},"coordinates":result}),
    )
}
pub fn run(config: &Path) -> Result<(), Box<dyn Error>> {
    let v: Value = serde_json::from_slice(&fs::read(config)?)?;
    let dir = Path::new(v["directory"].as_str().ok_or("missing directory")?);
    let chains = v["seeds"].as_array().ok_or("missing seeds")?.len();
    let draws = v["draws"].as_u64().ok_or("missing draws")? as usize;
    let out = json!({"schema_version":2,"unconstrained":summarize(&dir.join("draws.csv"),chains,draws,8)?,"constrained":summarize(&dir.join("constrained.csv"),chains,draws,2)?});
    super::write_json(&dir.join("metrics.json"), &out)
}
