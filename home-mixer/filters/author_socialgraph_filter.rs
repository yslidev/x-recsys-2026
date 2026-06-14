use std::collections::HashSet;

use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

// Remove candidates that are blocked or muted by the viewer
pub struct AuthorSocialgraphFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for AuthorSocialgraphFilter {
    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let viewer_blocked_user_ids: HashSet<i64> = query
            .user_features
            .blocked_user_ids
            .iter()
            .copied()
            .collect();
        let viewer_muted_user_ids: HashSet<i64> =
            query.user_features.muted_user_ids.iter().copied().collect();

        let mut kept: Vec<PostCandidate> = Vec::new();
        let mut removed: Vec<PostCandidate> = Vec::new();

        for candidate in candidates {
            let author_id = candidate.author_id as i64;
            let muted = viewer_muted_user_ids.contains(&author_id);
            let blocked = viewer_blocked_user_ids.contains(&author_id);
            let author_blocks_viewer = candidate.author_blocks_viewer.unwrap_or(false);

            let quoted_author_blocks_viewer =
                candidate.quoted_author_blocks_viewer.unwrap_or(false);
            let viewer_blocks_quoted_author = candidate
                .quoted_user_id
                .map(|uid| viewer_blocked_user_ids.contains(&(uid as i64)))
                .unwrap_or(false);

            let viewer_blocks_retweeted_user = candidate
                .retweeted_user_id
                .map(|uid| viewer_blocked_user_ids.contains(&(uid as i64)))
                .unwrap_or(false);

            if muted
                || blocked
                || author_blocks_viewer
                || quoted_author_blocks_viewer
                || viewer_blocks_quoted_author
                || viewer_blocks_retweeted_user
            {
                removed.push(candidate);
            } else {
                kept.push(candidate);
            }
        }

        FilterResult { kept, removed }
    }
}
