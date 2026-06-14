use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

/// Filter that removes tweets where the author is the viewer.
pub struct SelfTweetFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for SelfTweetFilter {
    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let viewer_id = query.user_id;
        let (kept, removed): (Vec<_>, Vec<_>) = candidates
            .into_iter()
            .partition(|c| c.author_id != viewer_id);

        FilterResult { kept, removed }
    }
}
