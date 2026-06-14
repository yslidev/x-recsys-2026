use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct VideoFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for VideoFilter {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.exclude_videos
    }

    fn filter(
        &self,
        _query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let (kept, removed): (Vec<_>, Vec<_>) = candidates
            .into_iter()
            .partition(|c| c.min_video_duration_ms.is_none());

        FilterResult { kept, removed }
    }
}
