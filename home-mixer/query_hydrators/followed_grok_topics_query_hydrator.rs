use crate::clients::followed_grok_topics_store_client::FollowedGrokTopicsStoreClient;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnableContextFeatures, EnableNewUserTopicFiltering, EnableNewUserTopicRetrieval,
    NewUserTopicAgeThresholdSecs,
};
use std::sync::Arc;
use std::time::Duration;
use tonic::async_trait;

use xai_candidate_pipeline::component_library::utils::duration_since_creation_opt;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;
use xai_recsys_proto::grok_topics::{PARENT_TOPIC_IDS, ids_to_bool_array};

const MH_GET_TIMEOUT: Duration = Duration::from_millis(300);
pub struct FollowedGrokTopicsQueryHydrator {
    client: Arc<dyn FollowedGrokTopicsStoreClient>,
}

impl FollowedGrokTopicsQueryHydrator {
    pub fn new(client: Arc<dyn FollowedGrokTopicsStoreClient>) -> Self {
        Self { client }
    }
}

#[async_trait]
impl QueryHydrator<ScoredPostsQuery> for FollowedGrokTopicsQueryHydrator {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableContextFeatures)
            || query.is_shadow_traffic
            || query.params.get(EnableNewUserTopicRetrieval)
            || query.params.get(EnableNewUserTopicFiltering)
    }

    async fn hydrate(&self, query: &ScoredPostsQuery) -> Result<ScoredPostsQuery, String> {
        let ids = tokio::time::timeout(MH_GET_TIMEOUT, self.client.fetch(query.user_id))
            .await
            .map_err(|_| "FollowedGrokTopics MH get timed out".to_string())??;

        let topics = ids
            .as_ref()
            .map(|ids| ids_to_bool_array(ids, &PARENT_TOPIC_IDS));

        let new_user_topics_enabled = query.params.get(EnableNewUserTopicRetrieval)
            || query.params.get(EnableNewUserTopicFiltering);
        let new_user_topic_ids =
            if new_user_topics_enabled && !query.is_topic_request() && !query.in_network_only {
                let threshold = Duration::from_secs(query.params.get(NewUserTopicAgeThresholdSecs));
                let is_new_user = duration_since_creation_opt(query.user_id)
                    .map(|age| age < threshold)
                    .unwrap_or(false);

                if is_new_user {
                    ids.clone().unwrap_or_default()
                } else {
                    vec![]
                }
            } else {
                vec![]
            };

        Ok(ScoredPostsQuery {
            followed_grok_topics: topics,
            new_user_topic_ids,
            ..Default::default()
        })
    }

    fn update(&self, query: &mut ScoredPostsQuery, hydrated: ScoredPostsQuery) {
        query.followed_grok_topics = hydrated.followed_grok_topics;
        if !hydrated.new_user_topic_ids.is_empty() {
            query.new_user_topic_ids = hydrated.new_user_topic_ids;
        }
    }
}
