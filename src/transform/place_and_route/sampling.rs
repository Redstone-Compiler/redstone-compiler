use std::mem;

use rand::rngs::StdRng;
use rand::seq::IteratorRandom;
use rand::SeedableRng;

#[derive(Default, Copy, Clone)]
pub enum SamplingPolicy {
    #[default]
    None,
    Take(usize),
    Random(usize),
}

impl SamplingPolicy {
    pub fn sample<T>(self, src: Vec<T>) -> Vec<T> {
        match self {
            SamplingPolicy::None => src,
            SamplingPolicy::Take(count) => src.into_iter().take(count).collect(),
            SamplingPolicy::Random(count) => src
                .into_iter()
                .choose_multiple(&mut Self::placer_rng(), count),
        }
    }

    pub fn sample_with_taking<T: Clone + Default>(self, src: &mut Vec<T>) -> Vec<T> {
        match self {
            SamplingPolicy::None => src.to_vec(),
            SamplingPolicy::Take(count) => src.iter_mut().take(count).map(mem::take).collect(),
            SamplingPolicy::Random(count) => src
                .iter_mut()
                .choose_multiple(&mut Self::placer_rng(), count)
                .into_iter()
                .map(mem::take)
                .collect(),
        }
    }

    fn placer_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }
}
