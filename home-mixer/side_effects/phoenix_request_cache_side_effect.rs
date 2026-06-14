use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{EnablePhoenixRequestCacheSideEffect, PhoenixRequestCacheSideEffectTtlSeconds};
use crate::util::phoenix_request::{
    build_request_without_sequence_and_candidates, build_tweet_infos,
};
use futures::future::join_all;
use prost::Message;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::redis_client::RedisClient;
use xai_candidate_pipeline::component_library::utils::is_prod;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_recsys_proto::ProductSurface;

const KEY_PREFIX: &str = "phoenix_request_cache";

pub struct PhoenixRequestCacheSideEffect {
    phoenix_request_cache_redis_atla_client: Arc<dyn RedisClient>,
    phoenix_request_cache_redis_pdxa_client: Arc<dyn RedisClient>,
}

impl PhoenixRequestCacheSideEffect {
    pub fn new(
        phoenix_request_cache_redis_atla_client: Arc<dyn RedisClient>,
        phoenix_request_cache_redis_pdxa_client: Arc<dyn RedisClient>,
    ) -> Self {
        Self {
            phoenix_request_cache_redis_atla_client,
            phoenix_request_cache_redis_pdxa_client,
        }
    }
}

pub fn request_level_cache_key(prediction_request_id: u64) -> String {
    format!("{KEY_PREFIX}_{prediction_request_id}")
}

pub fn candidate_cache_key(prediction_request_id: u64, post_id: u64) -> String {
    format!("{KEY_PREFIX}_{prediction_request_id}_{post_id}")
}

#[async_trait]
impl SideEffect<ScoredPostsQuery, PostCandidate> for PhoenixRequestCacheSideEffect {
    fn enable(&self, query: Arc<ScoredPostsQuery>) -> bool {
        is_prod() && query.params.get(EnablePhoenixRequestCacheSideEffect)
    }

    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, PostCandidate>>,
    ) -> Result<(), String> {
        let query = &input.query;
        let ttl = query.params.get(PhoenixRequestCacheSideEffectTtlSeconds);

        if input.selected_candidates.is_empty() && input.non_selected_candidates.is_empty() {
            return Ok(());
        }

        let prediction_request_id = input
            .selected_candidates
            .iter()
            .find_map(|c| c.prediction_request_id);
        let Some(prediction_request_id) = prediction_request_id else {
            return Ok(());
        };

        let product_surface = if query.in_network_only {
            ProductSurface::HomeTimelineRankedFollowing
        } else {
            ProductSurface::HomeTimelineRanking
        };

        let mut entries: Vec<(String, Vec<u8>)> = Vec::new();

        if !query.has_cached_posts {
            let phoenix_request =
                build_request_without_sequence_and_candidates(query, product_surface);
            entries.push((
                request_level_cache_key(prediction_request_id),
                phoenix_request.encode_to_vec(),
            ));
        }

        for candidate in build_tweet_infos(query, &input.selected_candidates) {
            entries.push((
                candidate_cache_key(prediction_request_id, candidate.tweet_id),
                candidate.encode_to_vec(),
            ));
        }

        let mut futures: Vec<
            std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send + '_>>,
        > = Vec::new();

        for (key, value) in entries {
            let pdxa_key = key.clone();
            let pdxa_value = value.clone();
            futures.push(Box::pin(async move {
                self.phoenix_request_cache_redis_atla_client
                    .set_ex(key, value, ttl)
                    .await
                    .map_err(|e| e.to_string())
            }));
            futures.push(Box::pin(async move {
                self.phoenix_request_cache_redis_pdxa_client
                    .set_ex(pdxa_key, pdxa_value, ttl)
                    .await
                    .map_err(|e| format!("xdc: {e}"))
            }));
        }

        let results = join_all(futures).await;
        let total = results.len();
        let mut first_error: Option<String> = None;
        let mut error_count = 0usize;
        for result in results {
            if let Err(e) = result {
                error_count += 1;
                if first_error.is_none() {
                    first_error = Some(e);
                }
            }
        }

        if total > 0 && error_count * 10 > total {
            return Err(first_error.unwrap_or_default());
        }
        Ok(())
    }
}
