#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStructure {
    SmoothContinuous,
    StiffHierarchical,
    DiscreteLatentChain,
    NonlinearStateSpace,
    LargeFactorGraph,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceMethod {
    HmcNuts,
    RmhmcChees,
    ForwardBackwardGibbs,
    ParticleMcmc,
    LoopyBpVariational,
}

#[derive(Debug, Clone, Copy)]
pub struct InferenceChoice {
    pub method: InferenceMethod,
    pub mixing_time: &'static str,
    pub gradient_cost: &'static str,
    pub scalability: &'static str,
}

pub fn choose_inference(structure: ModelStructure) -> InferenceChoice {
    match structure {
        ModelStructure::SmoothContinuous => InferenceChoice {
            method: InferenceMethod::HmcNuts,
            mixing_time: "Fast geometric exploration in moderate/high dimensions.",
            gradient_cost: "High: gradient per leapfrog step.",
            scalability: "Good with vectorized gradient evaluation.",
        },
        ModelStructure::StiffHierarchical => InferenceChoice {
            method: InferenceMethod::RmhmcChees,
            mixing_time: "Improved in stiff posterior geometries via metric adaptation.",
            gradient_cost: "Very high: metric/curvature terms increase per-step cost.",
            scalability: "Moderate, bottlenecked by dense linear algebra.",
        },
        ModelStructure::DiscreteLatentChain => InferenceChoice {
            method: InferenceMethod::ForwardBackwardGibbs,
            mixing_time: "Rapid mixing for chain-structured latent blocks.",
            gradient_cost: "Low: dynamic programming and conditional sampling.",
            scalability: "Excellent in sequence length with linear-time passes.",
        },
        ModelStructure::NonlinearStateSpace => InferenceChoice {
            method: InferenceMethod::ParticleMcmc,
            mixing_time: "Robust under multimodal/nonlinear state transitions.",
            gradient_cost: "Model-dependent; gradients often unnecessary.",
            scalability: "Parallel over particles, with higher constant factors.",
        },
        ModelStructure::LargeFactorGraph => InferenceChoice {
            method: InferenceMethod::LoopyBpVariational,
            mixing_time: "Approximate fixed-point convergence on sparse graphs.",
            gradient_cost: "Low to moderate depending on variational family.",
            scalability: "Best for large sparse factor graphs.",
        },
    }
}
