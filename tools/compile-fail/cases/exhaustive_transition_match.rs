use alea_mcmc::reference::NutsTermination;

fn classify(termination: NutsTermination) -> usize {
    match termination {
        NutsTermination::UTurn => 0,
        NutsTermination::MaxDepth => 1,
        NutsTermination::Divergence(_) => 2,
    }
}

fn main() {
    let _ = classify;
}
