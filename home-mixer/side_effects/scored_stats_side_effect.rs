use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnablePhoenixRetrievalStatsExperimentBucket, PhoenixRetrievalInferenceClusterId,
    PhoenixRetrievalMOEInferenceClusterId,
};

use rand::random;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_feature_switches::ExperimentBucket;
use xai_home_mixer_proto::ServedType;
use xai_stats_receiver::{HistogramBuckets, StatsReceiverExt, global_stats_receiver};

const METRIC_PREFIX: &str = "ScoredStats";

const PRESENT_SCOPE: [(&str, &str); 1] = [("score_status", "present")];
const MISSING_SCOPE: [(&str, &str); 1] = [("score_status", "missing")];

const HEAVY_RANKER_TOP_K: &[usize] = &[1, 10, 35];

const DEFAULT_SAMPLING_RATE: f64 = 0.05;

pub struct ScoredStatsSideEffect;

#[async_trait]
impl SideEffect<ScoredPostsQuery, PostCandidate> for ScoredStatsSideEffect {
    fn enable(&self, _query: Arc<ScoredPostsQuery>) -> bool {
        true
    }

    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, PostCandidate>>,
    ) -> Result<(), String> {
        let Some(receiver) = global_stats_receiver() else {
            return Ok(());
        };

        let candidates = &input.selected_candidates;
        if candidates.is_empty() {
            return Ok(());
        }

        if !input.query.has_cached_posts {
            let retrieval_cluster: String =
                input.query.params.get(PhoenixRetrievalInferenceClusterId);
            let moe_cluster: String = input
                .query
                .params
                .get(PhoenixRetrievalMOEInferenceClusterId);
            let experiment_buckets = input
                .query
                .params
                .experiment_buckets(EnablePhoenixRetrievalStatsExperimentBucket);

            if !experiment_buckets.is_empty() || random::<f64>() < DEFAULT_SAMPLING_RATE {
                record_score_distributions(receiver.as_ref(), candidates);
                record_phoenix_retrieval_stats(
                    receiver.as_ref(),
                    &input.selected_candidates,
                    &input.non_selected_candidates,
                    &retrieval_cluster,
                    &experiment_buckets,
                );
                record_phoenix_retrieval_moe_stats(
                    receiver.as_ref(),
                    &input.selected_candidates,
                    &input.non_selected_candidates,
                    &moe_cluster,
                    &experiment_buckets,
                );
            }
        } else {
            if random::<f64>() < DEFAULT_SAMPLING_RATE {
                record_score_distributions(receiver.as_ref(), candidates);
            }
        }

        Ok(())
    }
}

fn record_head(
    receiver: &dyn StatsReceiverExt,
    name: &str,
    scores: impl Iterator<Item = Option<f64>>,
) {
    let distribution_key = format!("{METRIC_PREFIX}.scoreDistribution.{name}");
    let missing_key = format!("{METRIC_PREFIX}.scoreMissing.{name}");
    let mut present = 0u64;
    let mut missing = 0u64;
    for score in scores {
        match score {
            Some(value) => {
                present += 1;
                receiver.observe(&distribution_key, &[], value, HistogramBuckets::Bucket0To1);
            }
            None => {
                missing += 1;
            }
        }
    }
    receiver.incr(&missing_key, &PRESENT_SCOPE, present);
    receiver.incr(&missing_key, &MISSING_SCOPE, missing);
}

fn record_score_distributions(receiver: &dyn StatsReceiverExt, candidates: &[PostCandidate]) {
    record_head(
        receiver,
        "favorite",
        candidates.iter().map(|c| c.phoenix_scores.favorite_score),
    );
    record_head(
        receiver,
        "reply",
        candidates.iter().map(|c| c.phoenix_scores.reply_score),
    );
    record_head(
        receiver,
        "retweet",
        candidates.iter().map(|c| c.phoenix_scores.retweet_score),
    );
    record_head(
        receiver,
        "click",
        candidates.iter().map(|c| c.phoenix_scores.click_score),
    );
    record_head(
        receiver,
        "vqv",
        candidates.iter().map(|c| c.phoenix_scores.vqv_score),
    );
    record_head(
        receiver,
        "share",
        candidates.iter().map(|c| c.phoenix_scores.share_score),
    );
    record_head(
        receiver,
        "not_interested",
        candidates
            .iter()
            .map(|c| c.phoenix_scores.not_interested_score),
    );
    record_head(
        receiver,
        "weightedScore",
        candidates.iter().map(|c| c.weighted_score),
    );
    record_head(receiver, "finalScore", candidates.iter().map(|c| c.score));
}

fn record_phoenix_retrieval_stats(
    receiver: &dyn StatsReceiverExt,
    selected_candidates: &[PostCandidate],
    non_selected_candidates: &[PostCandidate],
    retrieval_cluster: &str,
    experiment_buckets: &[&ExperimentBucket],
) {
    record_retrieval_source_stats(
        receiver,
        selected_candidates,
        non_selected_candidates,
        retrieval_cluster,
        experiment_buckets,
        ServedType::ForYouPhoenixRetrieval,
        "PhoenixRetrievalTweets",
    );
}

fn record_phoenix_retrieval_moe_stats(
    receiver: &dyn StatsReceiverExt,
    selected_candidates: &[PostCandidate],
    non_selected_candidates: &[PostCandidate],
    retrieval_cluster: &str,
    experiment_buckets: &[&ExperimentBucket],
) {
    record_retrieval_source_stats(
        receiver,
        selected_candidates,
        non_selected_candidates,
        retrieval_cluster,
        experiment_buckets,
        ServedType::ForYouPhoenixRetrievalMoe,
        "PhoenixRetrievalMoeTweets",
    );
}

fn record_retrieval_source_stats(
    receiver: &dyn StatsReceiverExt,
    selected_candidates: &[PostCandidate],
    non_selected_candidates: &[PostCandidate],
    retrieval_cluster: &str,
    experiment_buckets: &[&ExperimentBucket],
    served_type: ServedType,
    metric_name: &str,
) {
    let mut all_candidates: Vec<&PostCandidate> = selected_candidates
        .iter()
        .chain(non_selected_candidates.iter())
        .collect();
    all_candidates.sort_by(|a, b| {
        let score_a = a.weighted_score.unwrap_or(f64::NEG_INFINITY);
        let score_b = b.weighted_score.unwrap_or(f64::NEG_INFINITY);
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let empty_bucket = ExperimentBucket::new("", "");
    let buckets: &[&ExperimentBucket] = if experiment_buckets.is_empty() {
        &[&empty_bucket]
    } else {
        experiment_buckets
    };

    let topk_key = format!("{METRIC_PREFIX}.{metric_name}.RankedTopK");
    for &k in HEAVY_RANKER_TOP_K {
        let count = all_candidates
            .iter()
            .take(k)
            .filter(|c| c.served_type == Some(served_type))
            .count();
        let k_str = match k {
            1 => "1",
            10 => "10",
            35 => "35",
            _ => "unknown",
        };

        for b in buckets {
            let scopes: [(&str, &str); 5] = [
                ("type", "sum"),
                ("retrieval_cluster", retrieval_cluster),
                ("k", k_str),
                ("ddg", &b.experiment),
                ("bucket", &b.bucket),
            ];
            receiver.incr(&topk_key, &scopes, count as u64);
            let req_scopes: [(&str, &str); 5] = [
                ("type", "requests"),
                ("retrieval_cluster", retrieval_cluster),
                ("k", k_str),
                ("ddg", &b.experiment),
                ("bucket", &b.bucket),
            ];
            receiver.incr(&topk_key, &req_scopes, 1);
        }
    }

    let served_key = format!("{METRIC_PREFIX}.{metric_name}.Served");
    let count = selected_candidates
        .iter()
        .filter(|c| c.served_type == Some(served_type))
        .count();

    for b in buckets {
        receiver.incr(
            &served_key,
            &[
                ("type", "sum"),
                ("retrieval_cluster", retrieval_cluster),
                ("ddg", &b.experiment),
                ("bucket", &b.bucket),
            ],
            count as u64,
        );
        receiver.incr(
            &served_key,
            &[
                ("type", "requests"),
                ("retrieval_cluster", retrieval_cluster),
                ("ddg", &b.experiment),
                ("bucket", &b.bucket),
            ],
            1,
        );
    }
}
