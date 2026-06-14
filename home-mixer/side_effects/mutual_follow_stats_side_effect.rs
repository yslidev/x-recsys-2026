use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnableMutualFollowJaccardHydration, EnablePhoenixMOESource, PhoenixRetrievalInferenceClusterId,
    PhoenixRetrievalMOEInferenceClusterId,
};
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_stats_receiver::{HistogramBuckets, global_stats_receiver};

const METRIC_PREFIX: &str = "MutualFollowJaccard";

pub struct MutualFollowStatsSideEffect;

#[async_trait]
impl SideEffect<ScoredPostsQuery, PostCandidate> for MutualFollowStatsSideEffect {
    fn enable(&self, query: Arc<ScoredPostsQuery>) -> bool {
        query.params.get(EnableMutualFollowJaccardHydration)
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

        let retrieval_cluster: String = input.query.params.get(PhoenixRetrievalInferenceClusterId);
        let moe_enabled: bool = input.query.params.get(EnablePhoenixMOESource);

        let mut scope: Vec<(&str, &str)> = vec![("retrieval_cluster", &retrieval_cluster)];
        let moe_cluster: String;
        if moe_enabled {
            moe_cluster = input
                .query
                .params
                .get(PhoenixRetrievalMOEInferenceClusterId);
            scope.push(("moe_cluster", &moe_cluster));
        }

        let avg_key = format!("{METRIC_PREFIX}.avgScore");

        let mut sum = 0.0f64;
        let mut present = 0u64;

        for candidate in candidates {
            if let Some(j) = candidate.mutual_follow_jaccard {
                present += 1;
                sum += j;
            }
        }

        if present > 0 {
            let avg = sum / present as f64;
            receiver.observe(&avg_key, &scope, avg, HistogramBuckets::Bucket0To1);
        }

        Ok(())
    }
}
