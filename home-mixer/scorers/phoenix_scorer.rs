use crate::models::candidate::CandidateHelpers;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    PhoenixInferenceClusterId, PhoenixRankerNewUserHistoryThreshold,
    PhoenixRankerNewUserInferenceClusterId, UseEgressSidecar,
};
use crate::util::phoenix_request::build_prediction_request;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::phoenix_prediction_client::{
    PhoenixCluster, PhoenixPredictionClient,
};

use xai_candidate_pipeline::component_library::utils::current_timestamp_millis;
use xai_candidate_pipeline::scorer::Scorer;
use xai_recsys_proto::ProductSurface;

pub struct PhoenixScorer {
    pub phoenix_client: Arc<dyn PhoenixPredictionClient + Send + Sync>,
    pub egress_client: Arc<dyn PhoenixPredictionClient + Send + Sync>,
}

impl PhoenixScorer {
    fn resolve_cluster(query: &ScoredPostsQuery) -> PhoenixCluster {
        let configured_cluster =
            PhoenixCluster::parse(&query.params.get(PhoenixInferenceClusterId));

        let threshold: u64 = query.params.get(PhoenixRankerNewUserHistoryThreshold);
        if threshold > 0 {
            let action_count = query
                .scoring_sequence
                .as_ref()
                .and_then(|s| s.metadata.as_ref())
                .map(|m| m.length)
                .unwrap_or(0);

            if action_count < threshold {
                return PhoenixCluster::parse(
                    &query.params.get(PhoenixRankerNewUserInferenceClusterId),
                );
            }
        }

        if let Some(decider) = &query.decider {
            match configured_cluster {
                PhoenixCluster::Experiment1Fou if decider.enabled("override_qf_use_lap7") => {
                    return PhoenixCluster::Experiment1Lap7;
                }
                PhoenixCluster::Experiment1Lap7 if decider.enabled("override_qf_use_fou") => {
                    return PhoenixCluster::Experiment1Fou;
                }
                _ => {}
            }
        }

        configured_cluster
    }
}

#[async_trait]
impl Scorer<ScoredPostsQuery, PostCandidate> for PhoenixScorer {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.has_cached_posts
    }

    async fn score(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let last_scored_at_ms = current_timestamp_millis();
        let product_surface = if query.in_network_only {
            ProductSurface::HomeTimelineRankedFollowing
        } else {
            ProductSurface::HomeTimelineRanking
        };

        if query.scoring_sequence.is_none() {
            return vec![Ok(PostCandidate::default()); candidates.len()];
        };

        let cluster = Self::resolve_cluster(query);
        let request = build_prediction_request(query, candidates, product_surface);

        let use_egress: bool = query.params.get(UseEgressSidecar);
        let client = if use_egress {
            &self.egress_client
        } else {
            &self.phoenix_client
        };

        let mut predictions = client.predict(cluster, request.clone()).await;

        if predictions.is_err() && use_egress {
            tracing::debug!("Egress predict failed, falling back");
            predictions = self.phoenix_client.predict(cluster, request).await;
        }

        let predictions = predictions.map_err(|e| format!("Phoenix prediction failed: {}", e));

        let predictions = match predictions {
            Ok(predictions) => predictions,
            Err(err) => return vec![Err(err); candidates.len()],
        };

        candidates
            .iter()
            .map(|c| PostCandidate {
                phoenix_scores: predictions.candidate_scores(&c.get_original_tweet_id()),
                prediction_request_id: Some(query.prediction_id),
                last_scored_at_ms,
                ..Default::default()
            })
            .map(Ok)
            .collect()
    }

    fn update(&self, candidate: &mut PostCandidate, scored: PostCandidate) {
        candidate.phoenix_scores = scored.phoenix_scores;
        candidate.prediction_request_id = scored.prediction_request_id;
        candidate.last_scored_at_ms = scored.last_scored_at_ms;
    }
}
