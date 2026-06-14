use crate::clients::impressed_posts_client::ImpressedPostsClient;
use crate::models::query::ScoredPostsQuery;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;

pub struct ImpressedPostsQueryHydrator {
    pub client: Arc<dyn ImpressedPostsClient>,
}

#[async_trait]
impl QueryHydrator<ScoredPostsQuery> for ImpressedPostsQueryHydrator {
    async fn hydrate(&self, query: &ScoredPostsQuery) -> Result<ScoredPostsQuery, String> {
        let impressed_post_ids = self.client.get(query.user_id).await?;

        Ok(ScoredPostsQuery {
            impressed_post_ids,
            ..Default::default()
        })
    }

    fn update(&self, query: &mut ScoredPostsQuery, hydrated: ScoredPostsQuery) {
        query.impressed_post_ids = hydrated.impressed_post_ids;
    }
}
