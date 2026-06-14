use crate::models::query::ScoredPostsQuery;
use crate::params::EnableIpFeature;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;
use xai_geo_ip::GeoIpLocationClient;

pub struct IpQueryHydrator {
    pub client: Arc<GeoIpLocationClient>,
}

#[async_trait]
impl QueryHydrator<ScoredPostsQuery> for IpQueryHydrator {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableIpFeature) && !query.ip_address.is_empty()
    }

    async fn hydrate(&self, query: &ScoredPostsQuery) -> Result<ScoredPostsQuery, String> {
        let result = self.client.fetch(&query.ip_address).await;

        Ok(ScoredPostsQuery {
            ip_location: result.location,
            ..Default::default()
        })
    }

    fn update(&self, query: &mut ScoredPostsQuery, hydrated: ScoredPostsQuery) {
        query.ip_location = hydrated.ip_location;
    }
}
