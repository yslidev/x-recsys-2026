use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::util::candidates_util::get_related_post_ids;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct PreviouslySeenPostsBackupFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for PreviouslySeenPostsBackupFilter {
    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        if query.impressed_post_ids.is_empty() {
            return FilterResult {
                kept: candidates,
                removed: Vec::new(),
            };
        }

        let (removed, kept): (Vec<_>, Vec<_>) = candidates.into_iter().partition(|c| {
            get_related_post_ids(c)
                .iter()
                .any(|id| query.impressed_post_ids.contains(id))
        });

        FilterResult { kept, removed }
    }
}
