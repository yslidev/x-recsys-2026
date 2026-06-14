use crate::clients::user_demographics_client::UserDemographicsClient;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableContextFeatures;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;

pub struct UserDemographicsQueryHydrator {
    pub client: Arc<dyn UserDemographicsClient>,
}

#[async_trait]
impl QueryHydrator<ScoredPostsQuery> for UserDemographicsQueryHydrator {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableContextFeatures) || query.is_shadow_traffic
    }

    async fn hydrate(&self, query: &ScoredPostsQuery) -> Result<ScoredPostsQuery, String> {
        let demographics = self.client.fetch(query.user_id).await?;

        Ok(ScoredPostsQuery {
            user_demographics: demographics,
            ..Default::default()
        })
    }

    fn update(&self, query: &mut ScoredPostsQuery, hydrated: ScoredPostsQuery) {
        query.user_demographics = hydrated.user_demographics;
    }
}
