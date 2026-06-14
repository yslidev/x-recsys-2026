use crate::clients::tweet_entity_service_client::TESClient;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::utils::{MokaCache, default_moka_cache};
use xai_candidate_pipeline::hydrator::{CacheStore, CachedHydrator};
use xai_stats_receiver::global_stats_receiver;

const FOUND_SCOPE: [(&str, &str); 1] = [("hydration", "found")];
const MISSING_SCOPE: [(&str, &str); 1] = [("hydration", "missing")];

pub struct CoreDataCandidateHydrator {
    pub tes_client: Arc<dyn TESClient + Send + Sync>,
    pub cache: MokaCache<u64, CoreDataCacheValue>,
}

impl CoreDataCandidateHydrator {
    pub async fn new(tes_client: Arc<dyn TESClient + Send + Sync>) -> Self {
        let cache = default_moka_cache();
        Self { tes_client, cache }
    }
}

#[async_trait]
impl CachedHydrator<ScoredPostsQuery, PostCandidate> for CoreDataCandidateHydrator {
    type CacheKey = u64;

    type CacheValue = CoreDataCacheValue;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.has_cached_posts
    }

    fn cache_store(&self) -> &dyn CacheStore<Self::CacheKey, Self::CacheValue> {
        &self.cache
    }
    fn cache_key(&self, candidate: &PostCandidate) -> Self::CacheKey {
        candidate.tweet_id
    }

    fn cache_value(&self, hydrated: &PostCandidate) -> Self::CacheValue {
        CoreDataCacheValue {
            author_id: hydrated.author_id,
            retweeted_user_id: hydrated.retweeted_user_id,
            retweeted_tweet_id: hydrated.retweeted_tweet_id,
            in_reply_to_tweet_id: hydrated.in_reply_to_tweet_id,
            tweet_text: hydrated.tweet_text.clone(),
        }
    }

    fn hydrate_from_cache(&self, value: Self::CacheValue) -> PostCandidate {
        PostCandidate {
            author_id: value.author_id,
            retweeted_user_id: value.retweeted_user_id,
            retweeted_tweet_id: value.retweeted_tweet_id,
            in_reply_to_tweet_id: value.in_reply_to_tweet_id,
            tweet_text: value.tweet_text,
            ..Default::default()
        }
    }

    async fn hydrate_from_client(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let client = &self.tes_client;

        let tweet_ids: Vec<u64> = candidates.iter().map(|c| c.tweet_id).collect();

        let post_features = client.get_tweet_core_datas(tweet_ids.clone()).await;

        let mut hydrated_candidates = Vec::with_capacity(candidates.len());
        let mut hydrated_count = 0usize;
        let mut missing_count = 0usize;
        for tweet_id in tweet_ids {
            let post_features = post_features.get(&tweet_id);
            match post_features {
                Some(Ok(Some(core_data))) => {
                    hydrated_count += 1;
                    let text = core_data.text.clone();
                    let hydrated = PostCandidate {
                        author_id: core_data.author_id,
                        retweeted_user_id: core_data.source_user_id,
                        retweeted_tweet_id: core_data.source_tweet_id,
                        in_reply_to_tweet_id: core_data.in_reply_to_tweet_id,
                        tweet_text: text,
                        ..Default::default()
                    };
                    hydrated_candidates.push(Ok(hydrated));
                }
                Some(Ok(None)) | None => {
                    missing_count += 1;
                    hydrated_candidates.push(Ok(PostCandidate::default()));
                }
                Some(Err(err)) => {
                    hydrated_candidates.push(Err(err.to_string()));
                }
            }
        }

        self.record_hydration_stats(hydrated_count, missing_count);

        hydrated_candidates
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.retweeted_user_id = hydrated.retweeted_user_id;
        candidate.retweeted_tweet_id = hydrated.retweeted_tweet_id;
        candidate.in_reply_to_tweet_id = hydrated.in_reply_to_tweet_id;
        candidate.tweet_text = hydrated.tweet_text;
    }
}

#[derive(Clone, Debug)]
pub struct CoreDataCacheValue {
    pub author_id: u64,
    pub retweeted_user_id: Option<u64>,
    pub retweeted_tweet_id: Option<u64>,
    pub in_reply_to_tweet_id: Option<u64>,
    pub tweet_text: String,
}

impl CoreDataCandidateHydrator {
    fn record_hydration_stats(&self, hydrated_count: usize, missing_count: usize) {
        if let Some(receiver) = global_stats_receiver() {
            let metric_name = format!("{}.hydrate", self.name());
            receiver.incr(metric_name.as_str(), &FOUND_SCOPE, hydrated_count as u64);
            receiver.incr(metric_name.as_str(), &MISSING_SCOPE, missing_count as u64);
        }
    }
}
