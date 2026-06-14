use crate::clients::tweet_entity_service_client::TESClient;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::utils::{MokaCache, default_moka_cache};
use xai_candidate_pipeline::hydrator::{CacheStore, CachedHydrator};

pub struct SubscriptionHydrator {
    pub tes_client: Arc<dyn TESClient + Send + Sync>,
    pub cache: MokaCache<u64, Option<u64>>,
}

impl SubscriptionHydrator {
    pub async fn new(tes_client: Arc<dyn TESClient + Send + Sync>) -> Self {
        let cache = default_moka_cache();
        Self { tes_client, cache }
    }
}

#[async_trait]
impl CachedHydrator<ScoredPostsQuery, PostCandidate> for SubscriptionHydrator {
    type CacheKey = u64;
    type CacheValue = Option<u64>;

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
        hydrated.subscription_author_id
    }

    fn hydrate_from_cache(&self, value: Self::CacheValue) -> PostCandidate {
        PostCandidate {
            subscription_author_id: value,
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

        let post_features = client.get_subscription_author_ids(tweet_ids.clone()).await;

        let mut hydrated_candidates = Vec::with_capacity(candidates.len());
        for tweet_id in tweet_ids {
            let post_features = post_features.get(&tweet_id);
            let hydrated = match post_features {
                Some(Ok(value)) => Ok(PostCandidate {
                    subscription_author_id: *value,
                    ..Default::default()
                }),
                None => Err(format!(
                    "Missing subscription author id for tweet_id={}",
                    tweet_id
                )),
                Some(Err(err)) => Err(err.to_string()),
            };
            hydrated_candidates.push(hydrated);
        }

        hydrated_candidates
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.subscription_author_id = hydrated.subscription_author_id;
    }
}
