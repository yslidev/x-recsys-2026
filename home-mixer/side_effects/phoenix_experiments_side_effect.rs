use crate::clients::kafka_publisher_client::KafkaPublisherClient;
use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::params::UseEgressSidecar;
use crate::util::phoenix_request::build_prediction_request;
use futures::future::join_all;
use std::collections::BTreeSet;
use std::sync::Arc;
use strum::VariantArray;
use thrift::OrderedFloat;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::phoenix_prediction_client::{
    PhoenixCluster, PhoenixPredictionClient,
};
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_recsys_logging_thrift::{PredictionScore, ScoredCandidate, serialize_to_bytes_binary};
use xai_recsys_proto::{ProductSurface, language_code_string_to_enum};

pub struct PhoenixExperimentsSideEffect {
    phoenix_client: Arc<dyn PhoenixPredictionClient + Send + Sync>,
    egress_client: Arc<dyn PhoenixPredictionClient + Send + Sync>,
    kafka_client: Arc<dyn KafkaPublisherClient>,
}

impl PhoenixExperimentsSideEffect {
    pub fn new(
        phoenix_client: Arc<dyn PhoenixPredictionClient + Send + Sync>,
        egress_client: Arc<dyn PhoenixPredictionClient + Send + Sync>,
        kafka_client: Arc<dyn KafkaPublisherClient>,
    ) -> Self {
        Self {
            phoenix_client,
            egress_client,
            kafka_client,
        }
    }
}

#[async_trait]
impl SideEffect<ScoredPostsQuery, PostCandidate> for PhoenixExperimentsSideEffect {
    fn enable(&self, query: Arc<ScoredPostsQuery>) -> bool {
        query.is_shadow_traffic
    }

    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, PostCandidate>>,
    ) -> Result<(), String> {
        if input.query.scoring_sequence.is_none() {
            return Ok(());
        };

        let request_time_ms = input.query.request_time_ms;

        let product_surface = if input.query.in_network_only {
            ProductSurface::HomeTimelineRankedFollowing
        } else {
            ProductSurface::HomeTimelineRanking
        };

        let user_id = input.query.user_id;
        let use_egress: bool = input.query.params.get(UseEgressSidecar);

        let base_request =
            build_prediction_request(&input.query, &input.selected_candidates, product_surface);

        let futures = PhoenixCluster::VARIANTS
            .iter()
            .filter(|c| c.is_shadow_eligible())
            .map(|&cluster_id| {
                let client = if use_egress {
                    Arc::clone(&self.egress_client)
                } else {
                    Arc::clone(&self.phoenix_client)
                };
                let request = base_request.clone();
                async move {
                    let result = client.predict(cluster_id, request).await;
                    if let Err(ref err) = result {
                        tracing::error!(
                            "Phoenix experiment {:?} request failed: {}",
                            cluster_id,
                            err
                        );
                    }
                    (cluster_id, result)
                }
            });
        let experiment_results: Vec<_> = join_all(futures).await;

        for candidate in &input.selected_candidates {
            let mut all_scores: BTreeSet<Box<PredictionScore>> = BTreeSet::new();

            for (cluster_id, result) in &experiment_results {
                let Ok(predictions) = result else {
                    continue;
                };
                let cluster_name = format!("{:?}", cluster_id);
                let s = predictions.candidate_scores(&candidate.get_original_tweet_id());
                let score_fields: &[(&str, Option<f64>)] = &[
                    ("favorite", s.favorite_score),
                    ("reply", s.reply_score),
                    ("retweet", s.retweet_score),
                    ("photo_expand", s.photo_expand_score),
                    ("click", s.click_score),
                    ("profile_click", s.profile_click_score),
                    ("vqv", s.vqv_score),
                    ("share", s.share_score),
                    ("share_via_dm", s.share_via_dm_score),
                    ("share_via_copy_link", s.share_via_copy_link_score),
                    ("dwell", s.dwell_score),
                    ("quote", s.quote_score),
                    ("quoted_click", s.quoted_click_score),
                    ("quoted_vqv", s.quoted_vqv_score),
                    ("follow_author", s.follow_author_score),
                    ("not_interested", s.not_interested_score),
                    ("block_author", s.block_author_score),
                    ("mute_author", s.mute_author_score),
                    ("report", s.report_score),
                    ("not_dwelled", s.not_dwelled_score),
                    ("dwell_time", s.dwell_time),
                ];
                all_scores.extend(score_fields.iter().filter_map(|(name, score)| {
                    score.map(|v| {
                        Box::new(PredictionScore::new(
                            Some(format!("phoenix.{cluster_name}.{name}")),
                            Some(OrderedFloat::from(v)),
                        ))
                    })
                }));
            }

            let source_tweet_id = candidate.retweeted_tweet_id.unwrap_or(candidate.tweet_id);
            let scored = ScoredCandidate {
                tweet_id: candidate.tweet_id as i64,
                viewer_id: Some(user_id as i64),
                author_id: Some(candidate.author_id as i64),
                request_join_id: Some(input.query.request_id as i64),
                score: None,
                suggest_type: None,
                is_in_network: candidate.in_network,
                in_reply_to_tweet_id: candidate.in_reply_to_tweet_id.map(|id| id as i64),
                quoted_tweet_id: candidate.quoted_tweet_id.map(|id| id as i64),
                quoted_user_id: candidate.quoted_user_id.map(|id| id as i64),
                request_time_ms: Some(request_time_ms),
                source_tweet_id: Some(source_tweet_id as i64),
                prediction_scores: Some(all_scores),
                fav_count: candidate.fav_count,
                reply_count: candidate.reply_count,
                retweet_count: candidate.repost_count,
                quote_count: candidate.quote_count,
                has_media: candidate.has_media,
                language_code: candidate
                    .language_code
                    .as_deref()
                    .map(|lc| language_code_string_to_enum(lc) as i32),
                video_duration_ms: candidate.min_video_duration_ms,
            };

            match serialize_to_bytes_binary(&scored) {
                Ok(bytes) => {
                    if let Err(e) = self.kafka_client.send(&bytes).await {
                        tracing::error!("Failed to publish scored candidate to Kafka: {e}");
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to serialize scored candidate: {e}");
                }
            }
        }

        Ok(())
    }
}
