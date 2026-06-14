use crate::candidate_pipeline::{PipelineCandidate, PipelineQuery};
use crate::util;
use std::any::{Any, type_name_of_val};
use std::hash::Hash;
use tonic::async_trait;
use tracing::warn;
use xai_stats_receiver::global_stats_receiver;

// Hydrators run in parallel and update candidate fields
#[async_trait]
pub trait Hydrator<Q, C>: Any + Send + Sync
where
    Q: PipelineQuery,
    C: PipelineCandidate,
{
    /// Decide if this hydrator should run for the given query
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    /// Hydrate candidates by performing async operations.
    /// Returns candidates with this hydrator's fields populated.
    ///
    /// IMPORTANT: The returned vector must have the same candidates in the same order as the input.
    /// Dropping candidates in a hydrator is not allowed - use a filter stage instead.
    async fn hydrate(&self, query: &Q, candidates: &[C]) -> Vec<Result<C, String>>;

    #[xai_stats_macro::receive_stats(latency=Bucket50To500, size=Bucket500To2500)]
    #[tracing::instrument(skip_all, name = "hydrator", fields(name = self.name()))]
    async fn run(&self, query: &Q, candidates: &[C]) -> Vec<Result<C, String>> {
        let hydrated = self.hydrate(query, candidates).await;
        let expected_len = candidates.len();
        if hydrated.len() == expected_len {
            hydrated
        } else {
            let message = format!(
                "Hydrator length_mismatch expected={} got={}",
                expected_len,
                hydrated.len()
            );
            warn!(
                "Skipped: length_mismatch expected={} got={}",
                expected_len,
                hydrated.len()
            );
            vec![Err(message); expected_len]
        }
    }

    /// Update a single candidate with the hydrated fields.
    /// Only the fields this hydrator is responsible for should be copied.
    fn update(&self, candidate: &mut C, hydrated: C);

    /// Update all successfully hydrated candidates with the fields from `hydrated`.
    /// Default implementation iterates and calls `update` for each pair.
    fn update_all(&self, candidates: &mut [C], hydrated: Vec<Result<C, String>>) {
        for (candidate, hydrated) in candidates.iter_mut().zip(hydrated) {
            if let Ok(hydrated) = hydrated {
                self.update(candidate, hydrated);
            }
        }
    }

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}

const CACHE_HIT_SCOPE: [(&str, &str); 1] = [("requests", "cache_hit")];
const CACHE_MISS_SCOPE: [(&str, &str); 1] = [("requests", "cache_miss")];

#[async_trait]
pub trait CacheStore<K, V>: Send + Sync {
    async fn get(&self, key: &K) -> Option<V>;
    async fn insert(&self, key: K, value: V);
}

#[async_trait]
pub trait CachedHydrator<Q, C>: Any + Send + Sync
where
    Q: PipelineQuery,
    C: PipelineCandidate,
{
    type CacheKey: Eq + Hash + Send + Sync + 'static;
    type CacheValue: Clone + Send + Sync + 'static;

    fn enable(&self, _query: &Q) -> bool {
        true
    }

    fn cache_store(&self) -> &dyn CacheStore<Self::CacheKey, Self::CacheValue>;
    fn cache_key(&self, candidate: &C) -> Self::CacheKey;
    fn cache_value(&self, hydrated: &C) -> Self::CacheValue;

    fn hydrate_from_cache(&self, value: Self::CacheValue) -> C;
    async fn hydrate_from_client(&self, query: &Q, candidates: &[C]) -> Vec<Result<C, String>>;

    fn update(&self, candidate: &mut C, hydrated: C);

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }

    fn stat_cache(&self, cache_hits: usize, cache_misses: usize) {
        if let Some(receiver) = global_stats_receiver() {
            let metric_name = format!("{}.cache", self.name());
            if cache_hits > 0 {
                receiver.incr(metric_name.as_str(), &CACHE_HIT_SCOPE, cache_hits as u64);
            }
            if cache_misses > 0 {
                receiver.incr(metric_name.as_str(), &CACHE_MISS_SCOPE, cache_misses as u64);
            }
        }
    }
}

#[async_trait]
impl<Q, C, T> Hydrator<Q, C> for T
where
    Q: PipelineQuery,
    C: PipelineCandidate,
    T: CachedHydrator<Q, C> + ?Sized,
{
    fn enable(&self, query: &Q) -> bool {
        CachedHydrator::enable(self, query)
    }

    async fn hydrate(&self, query: &Q, candidates: &[C]) -> Vec<Result<C, String>> {
        let mut results = vec![None; candidates.len()];
        let mut missing_candidates = Vec::new();
        let mut missing_keys = Vec::new();
        let mut missing_indices = Vec::new();
        let mut cache_hits = 0usize;
        let mut cache_misses = 0usize;

        for (index, candidate) in candidates.iter().enumerate() {
            let key = self.cache_key(candidate);
            match self.cache_store().get(&key).await {
                Some(value) => {
                    results[index] = Some(Ok(self.hydrate_from_cache(value)));
                    cache_hits += 1;
                }
                None => {
                    missing_candidates.push(candidate.clone());
                    missing_keys.push(key);
                    missing_indices.push(index);
                    cache_misses += 1;
                }
            }
        }

        self.stat_cache(cache_hits, cache_misses);

        if !missing_candidates.is_empty() {
            let hydrated_missing = self.hydrate_from_client(query, &missing_candidates).await;
            if hydrated_missing.len() != missing_candidates.len() {
                let message = format!(
                    "CachedHydrator length_mismatch expected={} got={}",
                    missing_candidates.len(),
                    hydrated_missing.len()
                );
                return vec![Err(message); candidates.len()];
            }

            for ((index, key), hydrated) in missing_indices
                .into_iter()
                .zip(missing_keys.into_iter())
                .zip(hydrated_missing.into_iter())
            {
                if let Ok(ref hydrated_candidate) = hydrated {
                    let value = self.cache_value(hydrated_candidate);
                    self.cache_store().insert(key, value).await;
                }
                results[index] = Some(hydrated);
            }
        }

        results
            .into_iter()
            .map(|result| {
                result.unwrap_or_else(|| Err("Missing hydration result for candidate".to_string()))
            })
            .collect()
    }

    fn update(&self, candidate: &mut C, hydrated: C) {
        CachedHydrator::update(self, candidate, hydrated);
    }
}
