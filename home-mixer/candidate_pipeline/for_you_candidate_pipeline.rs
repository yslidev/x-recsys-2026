use crate::clients::ad_index_client::{AdIndexClient, MockAdIndexClient, ProdAdIndexClient};
use crate::clients::kafka_publisher_client::{KafkaPublisherClient, MockKafkaPublisherClient};
use crate::clients::past_request_timestamps_client::{
    MockPastRequestTimestampsClient, PastRequestTimestampsClient, ProdPastRequestTimestampsClient,
};
use crate::clients::prompts_client::{MockPromptsClient, ProdPromptsClient, PromptsClient};
use crate::clients::served_history_client::{
    MockServedHistoryClient, ProdServedHistoryClient, ServedHistoryClient,
};
use crate::clients::tweet_entity_service_client::{MockTESClient, ProdTESClient, TESClient};
use crate::clients::who_to_follow_client::{
    MockWhoToFollowClient, ProdWhoToFollowClient, WhoToFollowClient,
};
use crate::models::query::ScoredPostsQuery;
use crate::params;
use crate::query_hydrators::past_request_timestamps_query_hydrator::PastRequestTimestampsQueryHydrator;
use crate::query_hydrators::served_history_query_hydrator::ServedHistoryQueryHydrator;
use crate::scored_posts_server::ScoredPostsServer;
use crate::selectors::BlenderSelector;
use crate::side_effects::ads_injection_logging_side_effect::AdsInjectionLoggingSideEffect;
use crate::side_effects::client_events_kafka_side_effect::ClientEventsKafkaSideEffect;
use crate::side_effects::for_you_response_stats_side_effect::ForYouResponseStatsSideEffect;
use crate::side_effects::publish_seen_ids_to_kafka_side_effect::PublishSeenIdsToKafkaSideEffect;
use crate::side_effects::served_candidates_kafka_side_effect::ServedCandidatesKafkaSideEffect;
use crate::side_effects::truncate_served_history_side_effect::TruncateServedHistorySideEffect;
use crate::side_effects::update_past_request_timestamps_side_effect::UpdatePastRequestTimestampsSideEffect;
use crate::side_effects::update_served_history_side_effect::UpdateServedHistorySideEffect;
use crate::sources::ads_source::AdsSource;
use crate::sources::prompts_source::PromptsSource;
use crate::sources::push_to_home_source::PushToHomeSource;
use crate::sources::scored_posts_source::ScoredPostsSource;
use crate::sources::who_to_follow_source::WhoToFollowSource;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::candidate_pipeline::CandidatePipeline;
use xai_candidate_pipeline::component_library::clients::{
    MockReplyMixerClient, ProdReplyMixerClient, ReplyMixerClient,
};
use xai_candidate_pipeline::filter::Filter;
use xai_candidate_pipeline::hydrator::Hydrator;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;
use xai_candidate_pipeline::scorer::Scorer;
use xai_candidate_pipeline::selector::Selector;
use xai_candidate_pipeline::side_effect::SideEffect;
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto::FeedItem;

pub struct ForYouCandidatePipeline {
    query_hydrators: Vec<Box<dyn QueryHydrator<ScoredPostsQuery>>>,
    sources: Vec<Box<dyn Source<ScoredPostsQuery, FeedItem>>>,
    selector: BlenderSelector,
    side_effects: Arc<Vec<Box<dyn SideEffect<ScoredPostsQuery, FeedItem>>>>,
}

impl ForYouCandidatePipeline {
    pub async fn new(scored_posts_server: Arc<ScoredPostsServer>, datacenter: &str) -> Self {
        let (
            ad_index_client,
            ads_injection_logging,
            publish_seen_ids,
            served_candidates,
            client_events,
            served_history_client,
            who_to_follow_client,
            prompts_client,
            past_request_timestamps_client,
            tes_client,
            reply_mixer_client,
        ) = tokio::join!(
            async {
                Arc::new(
                    ProdAdIndexClient::new(datacenter)
                        .await
                        .expect("Failed to create AdIndex client"),
                ) as Arc<dyn AdIndexClient + Send + Sync>
            },
            AdsInjectionLoggingSideEffect::prod(),
            PublishSeenIdsToKafkaSideEffect::prod(),
            ServedCandidatesKafkaSideEffect::prod(),
            ClientEventsKafkaSideEffect::prod(),
            async {
                Arc::new(
                    ProdServedHistoryClient::new(datacenter)
                        .await
                        .expect("Failed to create ServedHistoryClient"),
                ) as Arc<dyn ServedHistoryClient>
            },
            async {
                Arc::new(
                    ProdWhoToFollowClient::new(datacenter)
                        .await
                        .expect("Failed to create WhoToFollowClient"),
                ) as Arc<dyn WhoToFollowClient + Send + Sync>
            },
            async {
                Arc::new(
                    ProdPromptsClient::new(datacenter)
                        .await
                        .expect("Failed to create PromptsClient"),
                ) as Arc<dyn PromptsClient + Send + Sync>
            },
            async {
                Arc::new(
                    ProdPastRequestTimestampsClient::new(datacenter)
                        .await
                        .expect("Failed to create PastRequestTimestampsClient"),
                ) as Arc<dyn PastRequestTimestampsClient>
            },
            async {
                Arc::new(
                    ProdTESClient::new(None, datacenter)
                        .await
                        .expect("Failed to create TES client"),
                ) as Arc<dyn TESClient + Send + Sync>
            },
            async {
                Arc::new(
                    ProdReplyMixerClient::new(datacenter)
                        .await
                        .expect("Failed to create ReplyMixer client"),
                ) as Arc<dyn ReplyMixerClient>
            },
        );

        Self::build(
            scored_posts_server,
            ad_index_client,
            ads_injection_logging,
            publish_seen_ids,
            served_candidates,
            client_events,
            served_history_client,
            who_to_follow_client,
            prompts_client,
            past_request_timestamps_client,
            tes_client,
            reply_mixer_client,
        )
    }

    fn build(
        scored_posts_server: Arc<ScoredPostsServer>,
        ad_index_client: Arc<dyn AdIndexClient + Send + Sync>,
        ads_injection_logging: AdsInjectionLoggingSideEffect,
        publish_seen_ids: PublishSeenIdsToKafkaSideEffect,
        served_candidates: ServedCandidatesKafkaSideEffect,
        client_events: ClientEventsKafkaSideEffect,
        served_history_client: Arc<dyn ServedHistoryClient>,
        who_to_follow_client: Arc<dyn WhoToFollowClient + Send + Sync>,
        prompts_client: Arc<dyn PromptsClient + Send + Sync>,
        past_request_timestamps_client: Arc<dyn PastRequestTimestampsClient>,
        tes_client: Arc<dyn TESClient + Send + Sync>,
        reply_mixer_client: Arc<dyn ReplyMixerClient>,
    ) -> Self {
        let query_hydrators: Vec<Box<dyn QueryHydrator<ScoredPostsQuery>>> = vec![
            Box::new(ServedHistoryQueryHydrator::from_client(Arc::clone(
                &served_history_client,
            ))),
            Box::new(PastRequestTimestampsQueryHydrator::new(Arc::clone(
                &past_request_timestamps_client,
            ))),
        ];

        let sources: Vec<Box<dyn Source<ScoredPostsQuery, FeedItem>>> = vec![
            Box::new(ScoredPostsSource {
                scored_posts_server,
            }),
            Box::new(AdsSource { ad_index_client }),
            Box::new(WhoToFollowSource {
                who_to_follow_client,
            }),
            Box::new(PromptsSource { prompts_client }),
            Box::new(PushToHomeSource {
                tes_client,
                reply_mixer_client,
            }),
        ];

        let selector = BlenderSelector::new();

        let side_effects: Arc<Vec<Box<dyn SideEffect<ScoredPostsQuery, FeedItem>>>> =
            Arc::new(vec![
                Box::new(ads_injection_logging),
                Box::new(publish_seen_ids),
                Box::new(served_candidates),
                Box::new(client_events),
                Box::new(ForYouResponseStatsSideEffect),
                Box::new(UpdatePastRequestTimestampsSideEffect::new(
                    past_request_timestamps_client,
                )),
                Box::new(UpdateServedHistorySideEffect::new(Arc::clone(
                    &served_history_client,
                ))),
                Box::new(TruncateServedHistorySideEffect::new(served_history_client)),
            ]);

        Self {
            query_hydrators,
            sources,
            selector,
            side_effects,
        }
    }

    pub async fn mock(scored_posts_server: Arc<ScoredPostsServer>) -> Self {
        let ad_index_client: Arc<dyn AdIndexClient + Send + Sync> = Arc::new(MockAdIndexClient);
        let mock_kafka = Arc::new(MockKafkaPublisherClient) as Arc<dyn KafkaPublisherClient>;
        let ads_injection = AdsInjectionLoggingSideEffect::new(Arc::clone(&mock_kafka));
        let publish_seen_ids = PublishSeenIdsToKafkaSideEffect::new(Arc::clone(&mock_kafka));
        let served_candidates = ServedCandidatesKafkaSideEffect::new(Arc::clone(&mock_kafka));
        let client_events = ClientEventsKafkaSideEffect::new(mock_kafka);
        let served_history_client: Arc<dyn ServedHistoryClient> = Arc::new(MockServedHistoryClient);
        let who_to_follow_client: Arc<dyn WhoToFollowClient + Send + Sync> =
            Arc::new(MockWhoToFollowClient);
        let prompts_client: Arc<dyn PromptsClient + Send + Sync> = Arc::new(MockPromptsClient);
        let past_request_timestamps_client: Arc<dyn PastRequestTimestampsClient> =
            Arc::new(MockPastRequestTimestampsClient);
        let tes_client: Arc<dyn TESClient + Send + Sync> = Arc::new(MockTESClient::default());
        let reply_mixer_client: Arc<dyn ReplyMixerClient> = Arc::new(MockReplyMixerClient);
        Self::build(
            scored_posts_server,
            ad_index_client,
            ads_injection,
            publish_seen_ids,
            served_candidates,
            client_events,
            served_history_client,
            who_to_follow_client,
            prompts_client,
            past_request_timestamps_client,
            tes_client,
            reply_mixer_client,
        )
    }
}

#[async_trait]
impl CandidatePipeline<ScoredPostsQuery, FeedItem> for ForYouCandidatePipeline {
    fn query_hydrators(&self) -> &[Box<dyn QueryHydrator<ScoredPostsQuery>>] {
        &self.query_hydrators
    }

    fn sources(&self) -> &[Box<dyn Source<ScoredPostsQuery, FeedItem>>] {
        &self.sources
    }

    fn hydrators(&self) -> &[Box<dyn Hydrator<ScoredPostsQuery, FeedItem>>] {
        &[]
    }

    fn filters(&self) -> &[Box<dyn Filter<ScoredPostsQuery, FeedItem>>] {
        &[]
    }

    fn scorers(&self) -> &[Box<dyn Scorer<ScoredPostsQuery, FeedItem>>] {
        &[]
    }

    fn selector(&self) -> &dyn Selector<ScoredPostsQuery, FeedItem> {
        &self.selector
    }

    fn post_selection_hydrators(&self) -> &[Box<dyn Hydrator<ScoredPostsQuery, FeedItem>>] {
        &[]
    }

    fn post_selection_filters(&self) -> &[Box<dyn Filter<ScoredPostsQuery, FeedItem>>] {
        &[]
    }

    fn side_effects(&self) -> Arc<Vec<Box<dyn SideEffect<ScoredPostsQuery, FeedItem>>>> {
        Arc::clone(&self.side_effects)
    }

    fn result_size(&self) -> usize {
        params::FOR_YOU_MAX_RESULT_SIZE
    }
}
