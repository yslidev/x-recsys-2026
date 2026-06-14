use crate::models::query::ScoredPostsQuery;
use crate::models::user_features::UserFeatures;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::SocialGraphClientOps;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;

pub struct MutedUserIdsQueryHydrator {
    pub socialgraph_client: Arc<dyn SocialGraphClientOps>,
}

#[async_trait]
impl QueryHydrator<ScoredPostsQuery> for MutedUserIdsQueryHydrator {
    async fn hydrate(&self, query: &ScoredPostsQuery) -> Result<ScoredPostsQuery, String> {
        let muted_user_ids = self
            .socialgraph_client
            .get_muted_user_ids(query.user_id)
            .await
            .map_err(|e| e.to_string())?;

        Ok(ScoredPostsQuery {
            user_features: UserFeatures {
                muted_user_ids,
                ..Default::default()
            },
            ..Default::default()
        })
    }

    fn update(&self, query: &mut ScoredPostsQuery, hydrated: ScoredPostsQuery) {
        query.user_features.muted_user_ids = hydrated.user_features.muted_user_ids;
    }
}
