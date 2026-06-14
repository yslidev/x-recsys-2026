use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::collections::HashSet;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

/// Filters out subscription-only posts from authors the viewer is not subscribed to.
pub struct IneligibleSubscriptionFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for IneligibleSubscriptionFilter {
    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let subscribed_user_ids: HashSet<u64> = query
            .user_features
            .subscribed_user_ids
            .iter()
            .map(|id| *id as u64)
            .collect();

        let (kept, removed): (Vec<_>, Vec<_>) =
            candidates
                .into_iter()
                .partition(|candidate| match candidate.subscription_author_id {
                    Some(author_id) => subscribed_user_ids.contains(&author_id),
                    None => true,
                });

        FilterResult { kept, removed }
    }
}
