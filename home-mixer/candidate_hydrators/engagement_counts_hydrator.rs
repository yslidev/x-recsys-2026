use crate::clients::tweet_entity_service_client::TESClient;
use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableContextFeatures;
use std::sync::Arc;
use std::time::Duration;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::utils::{
    MokaCache, TweetAgeExpiry, build_moka_cache_tweet_age,
};
use xai_candidate_pipeline::hydrator::{CacheStore, CachedHydrator};

#[derive(Clone, Debug)]
pub struct CachedCounts {
    fav_count: Option<i64>,
    reply_count: Option<i64>,
    repost_count: Option<i64>,
    quote_count: Option<i64>,
}

pub struct EngagementCountsHydrator {
    pub tes_client: Arc<dyn TESClient + Send + Sync>,
    cache: MokaCache<u64, CachedCounts>,
}

impl EngagementCountsHydrator {
    pub async fn new(tes_client: Arc<dyn TESClient + Send + Sync>) -> Self {
        let cache = build_moka_cache_tweet_age(
            1_000_000,
            TweetAgeExpiry {
                age_threshold: Duration::from_secs(30 * 60),
                new_tweet_ttl: Duration::from_secs(5 * 60),
                old_tweet_ttl: Duration::from_secs(10 * 60),
            },
        );
        Self { tes_client, cache }
    }
}

#[async_trait]
impl CachedHydrator<ScoredPostsQuery, PostCandidate> for EngagementCountsHydrator {
    type CacheKey = u64;
    type CacheValue = CachedCounts;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        (query.params.get(EnableContextFeatures) || query.is_shadow_traffic)
            && !query.has_cached_posts
    }

    fn cache_store(&self) -> &dyn CacheStore<Self::CacheKey, Self::CacheValue> {
        &self.cache
    }

    fn cache_key(&self, candidate: &PostCandidate) -> Self::CacheKey {
        candidate.get_original_tweet_id()
    }

    fn cache_value(&self, hydrated: &PostCandidate) -> Self::CacheValue {
        CachedCounts {
            fav_count: hydrated.fav_count,
            reply_count: hydrated.reply_count,
            repost_count: hydrated.repost_count,
            quote_count: hydrated.quote_count,
        }
    }

    fn hydrate_from_cache(&self, value: Self::CacheValue) -> PostCandidate {
        PostCandidate {
            fav_count: value.fav_count,
            reply_count: value.reply_count,
            repost_count: value.repost_count,
            quote_count: value.quote_count,
            ..Default::default()
        }
    }

    async fn hydrate_from_client(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let tweet_ids: Vec<u64> = candidates
            .iter()
            .map(|c| c.get_original_tweet_id())
            .collect();

        let counts_results = self.tes_client.get_api_counts(tweet_ids.clone()).await;

        tweet_ids
            .iter()
            .map(|tweet_id| {
                let counts = counts_results
                    .get(tweet_id)
                    .and_then(|r| r.as_ref().ok())
                    .and_then(|opt| opt.as_ref());
                Ok(PostCandidate {
                    fav_count: counts.and_then(|c| c.favorite_count),
                    reply_count: counts.and_then(|c| c.reply_count),
                    repost_count: counts.and_then(|c| c.retweet_count),
                    quote_count: counts.and_then(|c| c.quote_count),
                    ..Default::default()
                })
            })
            .collect()
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.fav_count = hydrated.fav_count;
        candidate.reply_count = hydrated.reply_count;
        candidate.repost_count = hydrated.repost_count;
        candidate.quote_count = hydrated.quote_count;
    }
}
