use crate::models::query::ScoredPostsQuery;
use crate::params::{EnableGrokTopicsHydration, MaxInferredGrokTopicCategories};
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::StratoClient;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;
use xai_recsys_proto::grok_topics::ids_to_multihot;

pub struct InferredGrokTopicsQueryHydrator {
    pub strato_client: Arc<dyn StratoClient + Send + Sync>,
}

#[async_trait]
impl QueryHydrator<ScoredPostsQuery> for InferredGrokTopicsQueryHydrator {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableGrokTopicsHydration)
    }

    async fn hydrate(&self, query: &ScoredPostsQuery) -> Result<ScoredPostsQuery, String> {
        let topics = self
            .strato_client
            .get_inferred_grok_topics(query.user_id)
            .await
            .map_err(|e| e.to_string())?
            .map(|ids| {
                let max_categories = query.params.get(MaxInferredGrokTopicCategories) as usize;
                ids_to_multihot(&ids, max_categories)
            });

        Ok(ScoredPostsQuery {
            inferred_grok_topics: topics,
            ..Default::default()
        })
    }

    fn update(&self, query: &mut ScoredPostsQuery, hydrated: ScoredPostsQuery) {
        query.inferred_grok_topics = hydrated.inferred_grok_topics;
    }
}
