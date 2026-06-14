use crate::clients::followed_starter_packs_store_client::FollowedStarterPacksStoreClient;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableContextFeatures;
use std::sync::Arc;
use std::time::Duration;
use tonic::async_trait;

use xai_candidate_pipeline::query_hydrator::QueryHydrator;
use xai_recsys_proto::starter_packs::{PACK_IDS, ids_to_bool_array};

const MH_GET_TIMEOUT: Duration = Duration::from_millis(300);
pub struct FollowedStarterPacksQueryHydrator {
    client: Arc<dyn FollowedStarterPacksStoreClient>,
}

impl FollowedStarterPacksQueryHydrator {
    pub fn new(client: Arc<dyn FollowedStarterPacksStoreClient>) -> Self {
        Self { client }
    }
}

#[async_trait]
impl QueryHydrator<ScoredPostsQuery> for FollowedStarterPacksQueryHydrator {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableContextFeatures) || query.is_shadow_traffic
    }

    async fn hydrate(&self, query: &ScoredPostsQuery) -> Result<ScoredPostsQuery, String> {
        let ids = tokio::time::timeout(MH_GET_TIMEOUT, self.client.fetch(query.user_id))
            .await
            .map_err(|_| "FollowedStarterPacks MH get timed out".to_string())??;

        let packs = ids.map(|ids| ids_to_bool_array(&ids, &PACK_IDS));

        Ok(ScoredPostsQuery {
            followed_starter_packs: packs,
            ..Default::default()
        })
    }

    fn update(&self, query: &mut ScoredPostsQuery, hydrated: ScoredPostsQuery) {
        query.followed_starter_packs = hydrated.followed_starter_packs;
    }
}
