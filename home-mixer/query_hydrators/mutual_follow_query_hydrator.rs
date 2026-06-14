use crate::models::query::ScoredPostsQuery;
use crate::params::EnableMutualFollowJaccardHydration;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::StratoClient;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;

pub struct MutualFollowQueryHydrator {
    pub strato_client: Arc<dyn StratoClient + Send + Sync>,
}

#[async_trait]
impl QueryHydrator<ScoredPostsQuery> for MutualFollowQueryHydrator {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableMutualFollowJaccardHydration)
    }

    async fn hydrate(&self, query: &ScoredPostsQuery) -> Result<ScoredPostsQuery, String> {
        let user_id = query.user_id as i64;
        let result = self
            .strato_client
            .get_minhash_with_count(user_id)
            .await
            .map_err(|e| e.to_string())?;

        let viewer_minhash = result.map(|(minhash, _count)| minhash);

        Ok(ScoredPostsQuery {
            viewer_minhash,
            ..Default::default()
        })
    }

    fn update(&self, query: &mut ScoredPostsQuery, hydrated: ScoredPostsQuery) {
        query.viewer_minhash = hydrated.viewer_minhash;
    }
}
