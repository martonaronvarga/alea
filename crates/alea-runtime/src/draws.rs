//! Optional draw storage: allocated once, filled without per-transition allocation.
use alea_math::buffer::OwnedBuffer;
use alea_mcmc::MarkovChain;

#[derive(Debug, thiserror::Error)]
pub enum DrawError<E: std::error::Error + 'static> {
    #[error("draw storage size exceeds addressable memory")]
    Size,
    #[error("chain dimension changed from {expected} to {actual}")]
    Dimension { expected: usize, actual: usize },
    #[error("chain transition failed: {0}")]
    Chain(#[source] E),
}

/// Row-major retained coordinates. Rejections are included, warmup is not implicit.
#[derive(Debug)]
pub struct Draws {
    data: OwnedBuffer,
    dimension: usize,
    rows: usize,
}
impl Draws {
    pub fn len(&self) -> usize {
        self.rows
    }
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    pub fn row(&self, index: usize) -> Option<&[f64]> {
        (index < self.rows)
            .then(|| &self.data[index * self.dimension..(index + 1) * self.dimension])
    }
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
}

/// Checks overflow and allocates before advancing the chain or RNG.
/// Returns errors at the last completed transition; it does not roll back the chain.
pub fn collect_draws<C: MarkovChain, R: rand::Rng + ?Sized>(
    chain: &mut C,
    rows: usize,
    rng: &mut R,
) -> Result<Draws, DrawError<C::Error>> {
    let dimension = chain.position().len();
    let length = rows
        .checked_mul(dimension)
        .filter(|&n| n <= (isize::MAX as usize - 64) / size_of::<f64>())
        .ok_or(DrawError::Size)?;
    let mut data = OwnedBuffer::new(length);
    for row in 0..rows {
        if chain.position().len() != dimension {
            return Err(DrawError::Dimension {
                expected: dimension,
                actual: chain.position().len(),
            });
        }
        chain.step(rng).map_err(DrawError::Chain)?;
        if chain.position().len() != dimension {
            return Err(DrawError::Dimension {
                expected: dimension,
                actual: chain.position().len(),
            });
        }
        data[row * dimension..(row + 1) * dimension].copy_from_slice(chain.position());
    }
    Ok(Draws {
        data,
        dimension,
        rows,
    })
}
