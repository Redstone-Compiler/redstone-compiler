use std::mem;

use rand::rngs::StdRng;
use rand::seq::IteratorRandom;
use rand::SeedableRng;

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
pub enum SamplingPolicy {
    #[default]
    None,
    Take(usize),
    Random(usize),
}

impl SamplingPolicy {
    pub fn sample<T>(self, src: Vec<T>) -> Vec<T> {
        self.sample_with_seed(src, 42)
    }

    pub fn sample_with_seed<T>(self, src: Vec<T>, seed: u64) -> Vec<T> {
        match self {
            SamplingPolicy::None => src,
            SamplingPolicy::Take(count) => src.into_iter().take(count).collect(),
            SamplingPolicy::Random(count) => src
                .into_iter()
                .choose_multiple(&mut Self::placer_rng(seed), count),
        }
    }

    pub fn sample_with_taking<T: Clone + Default>(self, src: &mut [T]) -> Vec<T> {
        self.sample_with_taking_seed(src, 42)
    }

    pub fn sample_with_taking_seed<T: Clone + Default>(self, src: &mut [T], seed: u64) -> Vec<T> {
        match self {
            SamplingPolicy::None => src.to_vec(),
            SamplingPolicy::Take(count) => src.iter_mut().take(count).map(mem::take).collect(),
            SamplingPolicy::Random(count) => src
                .iter_mut()
                .choose_multiple(&mut Self::placer_rng(seed), count)
                .into_iter()
                .map(mem::take)
                .collect(),
        }
    }

    fn placer_rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }
}
