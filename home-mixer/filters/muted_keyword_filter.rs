use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::sync::Arc;
use xai_candidate_pipeline::filter::{Filter, FilterResult};
use xai_post_text::{MatchTweetGroup, TokenSequence, TweetTokenizer, UserMutes};

pub struct MutedKeywordFilter {
    pub tokenizer: Arc<TweetTokenizer>,
}

impl MutedKeywordFilter {
    pub fn new() -> Self {
        let tokenizer = TweetTokenizer::new();
        Self {
            tokenizer: Arc::new(tokenizer),
        }
    }
}

impl Filter<ScoredPostsQuery, PostCandidate> for MutedKeywordFilter {
    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let muted_keywords = query.user_features.muted_keywords.clone();

        if muted_keywords.is_empty() {
            return FilterResult {
                kept: candidates,
                removed: vec![],
            };
        }

        let tokenizer = self.tokenizer.clone();
        tokio::task::block_in_place(|| {
            let tokenized = muted_keywords.iter().map(|k| tokenizer.tokenize(k));
            let token_sequences: Vec<TokenSequence> = tokenized.collect::<Vec<_>>();
            let user_mutes = UserMutes::new(token_sequences);
            let matcher = MatchTweetGroup::new(user_mutes);

            let mut kept = Vec::new();
            let mut removed = Vec::new();

            for candidate in candidates {
                let tweet_text_token_sequence = tokenizer.tokenize(&candidate.tweet_text);
                if matcher.matches(&tweet_text_token_sequence) {
                    removed.push(candidate);
                } else {
                    kept.push(candidate);
                }
            }

            FilterResult { kept, removed }
        })
    }
}
