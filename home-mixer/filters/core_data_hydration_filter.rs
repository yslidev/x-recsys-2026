use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct CoreDataHydrationFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for CoreDataHydrationFilter {
    fn filter(
        &self,
        _query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let (kept, removed) = candidates.into_iter().partition(|c| c.author_id != 0);
        FilterResult { kept, removed }
    }
}
