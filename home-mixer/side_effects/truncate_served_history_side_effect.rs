use crate::clients::served_history_client::{ServedHistoryClient, TimelineType, client_platform};
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableUrtMigrationComponents;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::utils::is_prod;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_home_mixer_proto::FeedItem;

const MAX_RESPONSES: usize = 50;

pub struct TruncateServedHistorySideEffect {
    client: Arc<dyn ServedHistoryClient>,
}

impl TruncateServedHistorySideEffect {
    pub fn new(client: Arc<dyn ServedHistoryClient>) -> Self {
        Self { client }
    }
}

#[async_trait]
impl SideEffect<ScoredPostsQuery, FeedItem> for TruncateServedHistorySideEffect {
    fn enable(&self, query: Arc<ScoredPostsQuery>) -> bool {
        is_prod()
            && query.params.get(EnableUrtMigrationComponents)
            && query.served_history.len() > MAX_RESPONSES
    }

    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, FeedItem>>,
    ) -> Result<(), String> {
        let query = &input.query;

        let to_delete: Vec<i64> = query
            .served_history
            .iter()
            .skip(MAX_RESPONSES)
            .filter_map(|sh| sh.served_time_ms)
            .collect();

        if to_delete.is_empty() {
            return Ok(());
        }

        let platform = client_platform::from_client_app_id(query.client_app_id);
        self.client
            .delete(query.user_id, TimelineType::Home, platform, &to_delete)
            .await
            .map_err(|e| e.to_string())
    }
}
