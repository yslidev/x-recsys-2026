use crate::clients::kafka_publisher_client::{
    CLIENT_SENT_IMPRESSIONS_TOPIC, KafkaCluster, KafkaPublisherClient, ProdKafkaPublisherClient,
};
use crate::models::query::ScoredPostsQuery;
use crate::params::EnablePublishSeenIdsToKafka;
use std::collections::BTreeSet;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::utils::is_prod;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_home_mixer_proto::FeedItem;
use xai_x_thrift::impression_store::{
    Impression, ImpressionList, PublishedImpressionList, SurfaceArea,
};
use xai_x_thrift::serialize_binary;

pub struct PublishSeenIdsToKafkaSideEffect {
    kafka_client: Arc<dyn KafkaPublisherClient>,
}

impl PublishSeenIdsToKafkaSideEffect {
    pub fn new(kafka_client: Arc<dyn KafkaPublisherClient>) -> Self {
        Self { kafka_client }
    }

    pub async fn prod() -> Self {
        Self::new(Arc::new(
            ProdKafkaPublisherClient::new(CLIENT_SENT_IMPRESSIONS_TOPIC, KafkaCluster::Bluebird)
                .await,
        ))
    }
}

#[async_trait]
impl SideEffect<ScoredPostsQuery, FeedItem> for PublishSeenIdsToKafkaSideEffect {
    fn enable(&self, query: Arc<ScoredPostsQuery>) -> bool {
        is_prod() && !query.seen_ids.is_empty() && query.params.get(EnablePublishSeenIdsToKafka)
    }

    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, FeedItem>>,
    ) -> Result<(), String> {
        let query = &input.query;
        if query.seen_ids.is_empty() {
            return Ok(());
        }

        let current_time = query.request_time_ms;

        let surface_areas = Some(BTreeSet::from([SurfaceArea::HOME_TIMELINE]));

        let impressions: Vec<Impression> = query
            .seen_ids
            .iter()
            .map(|&tweet_id| Impression {
                tweet_id: tweet_id as i64,
                impression_time: Some(current_time),
                linger_time_ms: None,
                surface_areas: surface_areas.clone(),
                media_details: None,
            })
            .collect();

        let published = PublishedImpressionList::new(
            query.user_id as i64,
            ImpressionList::new(impressions),
            current_time,
        );

        let bytes = serialize_binary(&published)
            .map_err(|e| format!("Thrift serialization failed: {e}"))?;

        self.kafka_client
            .send(&bytes)
            .await
            .map_err(|e| format!("Seen-IDs Kafka publish failed: {e}"))
    }
}
