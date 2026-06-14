use crate::clients::past_request_timestamps_client::PastRequestTimestampsClient;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableUrtMigrationComponents;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::utils::client_utils::RequestContext;
use xai_candidate_pipeline::component_library::utils::is_prod;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_home_mixer_proto::FeedItem;
use xai_x_thrift::non_polling_timestamps::NonPollingTimestamps;

const MAX_NON_POLLING_TIMES: usize = 10;

pub struct UpdatePastRequestTimestampsSideEffect {
    client: Arc<dyn PastRequestTimestampsClient>,
}

impl UpdatePastRequestTimestampsSideEffect {
    pub fn new(client: Arc<dyn PastRequestTimestampsClient>) -> Self {
        Self { client }
    }
}

#[async_trait]
impl SideEffect<ScoredPostsQuery, FeedItem> for UpdatePastRequestTimestampsSideEffect {
    fn enable(&self, query: Arc<ScoredPostsQuery>) -> bool {
        let is_background_fetch =
            RequestContext::parse(&query.request_context) == RequestContext::BackgroundFetch;
        is_prod()
            && query.params.get(EnableUrtMigrationComponents)
            && !query.is_polling
            && !is_background_fetch
    }

    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, FeedItem>>,
    ) -> Result<(), String> {
        let query = &input.query;

        let now_ms = query.request_time_ms;

        let prior_timestamps = query
            .non_polling_timestamps
            .as_ref()
            .map(|npt| &npt.non_polling_timestamps_ms[..])
            .unwrap_or(&[]);
        let most_recent = query
            .non_polling_timestamps
            .as_ref()
            .and_then(|npt| npt.most_recent_home_latest_non_polling_timestamp_ms);

        let timestamps: Vec<i64> = std::iter::once(now_ms)
            .chain(prior_timestamps.iter().copied())
            .take(MAX_NON_POLLING_TIMES)
            .collect();

        let npt = NonPollingTimestamps::new(timestamps, most_recent);
        self.client.put(query.user_id, &npt).await
    }
}
