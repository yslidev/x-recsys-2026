use crate::clients::past_request_timestamps_client::PastRequestTimestampsClient;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableUrtMigrationComponents;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;

pub struct PastRequestTimestampsQueryHydrator {
    client: Arc<dyn PastRequestTimestampsClient>,
}

impl PastRequestTimestampsQueryHydrator {
    pub fn new(client: Arc<dyn PastRequestTimestampsClient>) -> Self {
        Self { client }
    }
}

#[async_trait]
impl QueryHydrator<ScoredPostsQuery> for PastRequestTimestampsQueryHydrator {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableUrtMigrationComponents)
    }

    async fn hydrate(&self, query: &ScoredPostsQuery) -> Result<ScoredPostsQuery, String> {
        let non_polling_timestamps = self.client.get(query.user_id).await?;

        Ok(ScoredPostsQuery {
            non_polling_timestamps,
            ..Default::default()
        })
    }

    fn update(&self, query: &mut ScoredPostsQuery, hydrated: ScoredPostsQuery) {
        query.non_polling_timestamps = hydrated.non_polling_timestamps;
    }
}
