use crate::candidate_pipeline::{PipelineCandidate, PipelineQuery};
use crate::util;
use std::any::type_name_of_val;
use tonic::async_trait;
use tracing::warn;

/// Scorers update candidate fields (like a score field) and run sequentially
#[async_trait]
pub trait Scorer<Q, C>: Send + Sync
where
    Q: PipelineQuery,
    C: PipelineCandidate,
{
    /// Decide if this scorer should run for the given query
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    #[xai_stats_macro::receive_stats]
    #[tracing::instrument(skip_all, name = "scorer", fields(name = self.name()))]
    async fn run(&self, query: &Q, candidates: &[C]) -> Vec<Result<C, String>> {
        let scored = self.score(query, candidates).await;
        let expected_len = candidates.len();
        if scored.len() == expected_len {
            scored
        } else {
            let message = format!(
                "Scorer length_mismatch expected={} got={}",
                expected_len,
                scored.len()
            );
            warn!(
                "Skipped: length_mismatch expected={} got={}",
                expected_len,
                scored.len()
            );
            vec![Err(message); expected_len]
        }
    }

    /// Score candidates by performing async operations.
    /// Returns candidates with this scorer's fields populated.
    ///
    /// IMPORTANT: The returned vector must have the same candidates in the same order as the input.
    /// Dropping candidates in a hydrator is not allowed - use a filter stage instead.
    async fn score(&self, query: &Q, candidates: &[C]) -> Vec<Result<C, String>>;

    /// Update a single candidate with the scored fields.
    /// Only the fields this scorer is responsible for should be copied.
    fn update(&self, candidate: &mut C, scored: C);

    /// Update all successfully scored candidates with the fields from `scored`.
    /// Default implementation iterates and calls `update` for each pair.
    fn update_all(&self, candidates: &mut [C], scored: Vec<Result<C, String>>) {
        for (candidate, scored) in candidates.iter_mut().zip(scored) {
            if let Ok(scored) = scored {
                self.update(candidate, scored);
            }
        }
    }

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}
