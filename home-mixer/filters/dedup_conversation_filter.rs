use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::collections::HashMap;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct DedupConversationFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for DedupConversationFilter {
    fn filter(
        &self,
        _query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let mut kept: Vec<PostCandidate> = Vec::new();
        let mut removed: Vec<PostCandidate> = Vec::new();
        let mut best_per_convo: HashMap<u64, (usize, f64)> = HashMap::new();

        for candidate in candidates {
            let conversation_id = get_conversation_id(&candidate);
            let score = candidate.score.unwrap_or(0.0);

            if let Some((kept_idx, best_score)) = best_per_convo.get_mut(&conversation_id) {
                if score > *best_score {
                    let previous = std::mem::replace(&mut kept[*kept_idx], candidate);
                    removed.push(previous);
                    *best_score = score;
                } else {
                    removed.push(candidate);
                }
            } else {
                let idx = kept.len();
                best_per_convo.insert(conversation_id, (idx, score));
                kept.push(candidate);
            }
        }

        FilterResult { kept, removed }
    }
}

fn get_conversation_id(candidate: &PostCandidate) -> u64 {
    candidate
        .ancestors
        .iter()
        .copied()
        .min()
        .unwrap_or(candidate.tweet_id)
}
