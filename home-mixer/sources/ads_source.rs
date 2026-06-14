use crate::clients::ad_index_client::AdIndexClient;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableAdsSource;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto::{FeedItem, feed_item};
use xai_recsys_proto::{AdIndexRequest, ClientContext, ProductSurface};

pub struct AdsSource {
    pub ad_index_client: Arc<dyn AdIndexClient + Send + Sync>,
}

#[async_trait]
impl Source<ScoredPostsQuery, FeedItem> for AdsSource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableAdsSource) && !query.is_preview
    }

    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<FeedItem>, String> {
        let request = build_ad_index_request(query);

        let response = self
            .ad_index_client
            .get_eligible_ads(request)
            .await
            .map_err(|e| format!("AdsSource: {e}"))?;

        let feed_items = response
            .ad_info
            .into_iter()
            .map(|ad| FeedItem {
                position: 0,
                item: Some(feed_item::Item::Ad(ad)),
            })
            .collect();
        Ok(feed_items)
    }
}

fn build_ad_index_request(query: &ScoredPostsQuery) -> AdIndexRequest {
    AdIndexRequest {
        user_id: query.user_id as i64,
        product_surface: ProductSurface::HomeTimelineRanking as i32,
        client_context: Some(ClientContext {
            user_id: query.user_id as i64,
            app_id: query.client_app_id as i64,
            country_code: query.country_code.clone(),
            language_code: query.language_code.clone(),
            ip_address: query.ip_address.clone(),
            user_agent: query.user_agent.clone(),
            user_roles: query.user_roles.clone(),
            device_id: query.device_id.clone(),
            mobile_device_id: query.mobile_device_id.clone(),
            mobile_device_ad_id: query.mobile_device_ad_id.clone(),
            ..Default::default()
        }),
        ..Default::default()
    }
}
