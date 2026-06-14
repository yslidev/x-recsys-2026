use crate::clients::user_action_aggregation_client::{
    UserActionAggregationClient, UserActionsCluster,
};
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    self as p, IncludeRealtimeActions, MaxSeqLengthScoring, PhoenixAggregationType,
    UasSourceDataType, UseXdsForUas, UserActionsClusterId,
};
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;
use xai_recsys_proto::{
    ResponseFormat, UserActionAggregationType, UserActionSequenceSourceDataType,
};

pub struct ScoringSequenceQueryHydrator {
    pub user_action_aggregation_client: Arc<dyn UserActionAggregationClient + Send + Sync>,
}

impl ScoringSequenceQueryHydrator {
    pub fn new(
        user_action_aggregation_client: Arc<dyn UserActionAggregationClient + Send + Sync>,
    ) -> Self {
        Self {
            user_action_aggregation_client,
        }
    }
}

#[async_trait]
impl QueryHydrator<ScoredPostsQuery> for ScoringSequenceQueryHydrator {
    async fn hydrate(&self, query: &ScoredPostsQuery) -> Result<ScoredPostsQuery, String> {
        let cluster = UserActionsCluster::parse(&query.params.get(UserActionsClusterId));
        let include_realtime: bool = query.params.get(IncludeRealtimeActions);
        let source_data_type =
            UserActionSequenceSourceDataType::from_str_name(&query.params.get(UasSourceDataType))
                .unwrap_or(UserActionSequenceSourceDataType::Arrow);
        let use_xds: bool = query.params.get(UseXdsForUas);
        let result = self
            .user_action_aggregation_client
            .fetch_aggregated_sequence(
                cluster,
                query.user_id,
                p::UAS_WINDOW_TIME_MS as u32,
                query.params.get(MaxSeqLengthScoring),
                UserActionAggregationType::from_str_name(&query.params.get(PhoenixAggregationType))
                    .unwrap_or(UserActionAggregationType::DenseWithNotInterestedIn),
                source_data_type,
                ResponseFormat::Arrow,
                if include_realtime { Some(true) } else { None },
                Some(query.prediction_id as i64),
                use_xds,
            )
            .await
            .map_err(|e| format!("Aggregation service call failed: {}", e))?;

        Ok(ScoredPostsQuery {
            scoring_sequence: Some(result.sequence),
            columnar_scoring_sequence: result.columnar_bytes,
            ..Default::default()
        })
    }

    fn update(&self, query: &mut ScoredPostsQuery, hydrated: ScoredPostsQuery) {
        query.scoring_sequence = hydrated.scoring_sequence;
        query.columnar_scoring_sequence = hydrated.columnar_scoring_sequence;
    }
}
