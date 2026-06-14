use crate::clients::tweet_entity_service_client::TESClient;
use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::utils::{MokaCache, default_moka_cache};
use xai_candidate_pipeline::hydrator::{CacheStore, CachedHydrator};

pub struct VideoDurationCandidateHydrator {
    pub tes_client: Arc<dyn TESClient + Send + Sync>,
    pub cache: MokaCache<u64, Option<i32>>,
}

impl VideoDurationCandidateHydrator {
    pub async fn new(tes_client: Arc<dyn TESClient + Send + Sync>) -> Self {
        let cache = default_moka_cache();
        Self { tes_client, cache }
    }
}

#[async_trait]
impl CachedHydrator<ScoredPostsQuery, PostCandidate> for VideoDurationCandidateHydrator {
    type CacheKey = u64;

    type CacheValue = Option<i32>;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.has_cached_posts
    }

    fn cache_store(&self) -> &dyn CacheStore<Self::CacheKey, Self::CacheValue> {
        &self.cache
    }
    fn cache_key(&self, candidate: &PostCandidate) -> Self::CacheKey {
        candidate.get_original_tweet_id()
    }

    fn cache_value(&self, hydrated: &PostCandidate) -> Self::CacheValue {
        hydrated.min_video_duration_ms
    }

    fn hydrate_from_cache(&self, value: Self::CacheValue) -> PostCandidate {
        PostCandidate {
            min_video_duration_ms: value,
            ..Default::default()
        }
    }

    async fn hydrate_from_client(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let client = &self.tes_client;

        let tweet_ids: Vec<u64> = candidates
            .iter()
            .map(|c| c.get_original_tweet_id())
            .collect();

        let durations = client.get_min_video_durations(tweet_ids.clone()).await;

        let mut hydrated_candidates = Vec::with_capacity(candidates.len());
        for tweet_id in tweet_ids {
            let hydrated = match durations.get(&tweet_id) {
                Some(Ok(min_video_duration_ms)) => Ok(PostCandidate {
                    min_video_duration_ms: min_video_duration_ms.map(|v| v as i32),
                    ..Default::default()
                }),
                None => Err(format!(
                    "Missing min video duration for tweet_id={}",
                    tweet_id
                )),
                Some(Err(err)) => Err(err.to_string()),
            };
            hydrated_candidates.push(hydrated);
        }

        hydrated_candidates
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.min_video_duration_ms = hydrated.min_video_duration_ms;
    }
}
