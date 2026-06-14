use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::MaxPostsToCache;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::redis_client::{self, RedisClient};
use xai_candidate_pipeline::component_library::utils::is_prod;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};

const REDIS_TTL_SECONDS: u64 = 180;
const ZSTD_COMPRESSION_LEVEL: i32 = 6;

pub struct RedisPostCandidateCacheSideEffect {
    redis_client: Arc<dyn RedisClient>,
}

impl RedisPostCandidateCacheSideEffect {
    pub fn new(redis_client: Arc<dyn RedisClient>) -> Self {
        Self { redis_client }
    }

    fn get_candidates_to_cache<'a>(
        selected: &'a [PostCandidate],
        non_selected: &'a [PostCandidate],
        max_posts_to_cache: usize,
    ) -> Vec<&'a PostCandidate> {
        let mut all_candidates: Vec<&PostCandidate> = selected
            .iter()
            .chain(non_selected.iter())
            .filter(|c| c.weighted_score.is_some_and(|s| s > 0.0))
            .collect();
        all_candidates.sort_by(|a, b| {
            b.weighted_score
                .unwrap()
                .partial_cmp(&a.weighted_score.unwrap())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_candidates.truncate(max_posts_to_cache);
        all_candidates
    }
}

#[async_trait]
impl SideEffect<ScoredPostsQuery, PostCandidate> for RedisPostCandidateCacheSideEffect {
    fn enable(&self, query: Arc<ScoredPostsQuery>) -> bool {
        is_prod() && !query.has_cached_posts
    }

    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, PostCandidate>>,
    ) -> Result<(), String> {
        let max_posts_to_cache = input.query.params.get(MaxPostsToCache);
        let user_id = input.query.user_id;

        let candidates_to_cache = Self::get_candidates_to_cache(
            &input.selected_candidates,
            &input.non_selected_candidates,
            max_posts_to_cache,
        );

        let cache_key = redis_client::cached_posts_key(
            user_id,
            &input.query.topic_ids,
            input.query.in_network_only,
            input.query.exclude_videos,
        );
        let json_payload =
            serde_json::to_vec(&candidates_to_cache).map_err(|err| err.to_string())?;
        let uncompressed_size = json_payload.len();
        let compressed_payload = tokio::task::spawn_blocking(move || {
            zstd::encode_all(json_payload.as_slice(), ZSTD_COMPRESSION_LEVEL)
        })
        .await
        .map_err(|err| err.to_string())?
        .map_err(|err| err.to_string())?;

        tracing::debug!(
            user_id = user_id,
            count = candidates_to_cache.len(),
            cache_key = cache_key.clone(),
            uncompressed_size = uncompressed_size,
            compressed_size = compressed_payload.len(),
            "RedisPostCandidateCacheSideEffect caching candidates"
        );

        self.redis_client
            .set_ex(cache_key, compressed_payload, REDIS_TTL_SECONDS)
            .await
            .map_err(|err| err.to_string())
    }
}
