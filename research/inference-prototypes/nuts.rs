#[derive(Debug, Clone, Copy)]
pub struct NUTSConfig {
    pub max_tree_depth: usize,
    pub target_accept_rate: f64,
}

impl Default for NUTSConfig {
    fn default() -> Self {
        Self {
            max_tree_depth: 10,
            target_accept_rate: 0.8,
        }
    }
}

#[derive(Debug)]
pub struct NUTS {
    pub config: NUTSConfig,
}

impl NUTS {
    pub fn new(config: NUTSConfig) -> Self {
        Self { config }
    }
}
