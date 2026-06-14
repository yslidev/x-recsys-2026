use crate::candidate_hydrators::ads_brand_safety_hydrator::AdsBrandSafetyHydrator;
use crate::candidate_hydrators::ads_brand_safety_vf_hydrator::AdsBrandSafetyVfHydrator;
use crate::candidate_hydrators::blocked_by_hydrator::BlockedByHydrator;
use crate::candidate_hydrators::core_data_candidate_hydrator::CoreDataCandidateHydrator;
use crate::candidate_hydrators::filtered_topics_hydrator::FilteredTopicsHydrator;
use crate::candidate_hydrators::following_replied_users_hydrator::FollowingRepliedUsersHydrator;
use crate::candidate_hydrators::gizmoduck_hydrator::GizmoduckCandidateHydrator;
use crate::candidate_hydrators::has_media_hydrator::HasMediaHydrator;
use crate::candidate_hydrators::in_network_candidate_hydrator::InNetworkCandidateHydrator;
use crate::candidate_hydrators::language_code_hydrator::LanguageCodeHydrator;
use crate::candidate_hydrators::mutual_follow_jaccard_hydrator::MutualFollowJaccardHydrator;
use crate::candidate_hydrators::quote_hydrator::QuoteHydrator;
use crate::candidate_hydrators::subscription_hydrator::SubscriptionHydrator;
use crate::candidate_hydrators::tweet_type_metrics_hydrator::TweetTypeMetricsHydrator;
use crate::candidate_hydrators::vf_candidate_hydrator::VFCandidateHydrator;
use crate::candidate_hydrators::video_duration_candidate_hydrator::VideoDurationCandidateHydrator;
use crate::clients::followed_grok_topics_store_client::{
    FollowedGrokTopicsStoreClient, MockFollowedGrokTopicsStoreClient,
    ProdFollowedGrokTopicsStoreClient,
};
use crate::clients::followed_starter_packs_store_client::{
    FollowedStarterPacksStoreClient, MockFollowedStarterPacksStoreClient,
    ProdFollowedStarterPacksStoreClient,
};
use crate::clients::gender_prediction_client::{
    GenderPredictionGrpcClient, MockGenderPredictionGrpcClient, ProdGenderPredictionGrpcClient,
};
use crate::clients::gizmoduck_client::{GizmoduckClient, MockGizmoduckClient, ProdGizmoduckClient};
use crate::clients::impressed_posts_client::ImpressedPostsClient;
use crate::clients::kafka_publisher_client::{
    KafkaCluster, KafkaPublisherClient, MockKafkaPublisherClient, PHOENIX_SCORES_TOPIC,
    ProdKafkaPublisherClient, RERANKING_TOPIC,
};
use crate::clients::s2s::{S2S_CHAIN_PATH, S2S_CRT_PATH, S2S_KEY_PATH};
use crate::clients::tweet_entity_service_client::{MockTESClient, ProdTESClient, TESClient};
use crate::clients::user_action_aggregation_client::{
    MockUserActionAggregationClient, ProdUserActionAggregationClient, UserActionAggregationClient,
};
use crate::clients::user_demographics_client::{
    MockUserDemographicsClient, ProdUserDemographicsClient, UserDemographicsClient,
};
use crate::clients::user_inferred_gender_store_client::{
    MockUserInferredGenderStoreClient, ProdUserInferredGenderStoreClient,
    UserInferredGenderStoreClient,
};
use crate::clients::vm_ranker_client::{MockVMRankerClient, ProdVMRankerClient, VMRankerClient};
use crate::filters::age_filter::AgeFilter;
use crate::filters::ancillary_vf_filter::AncillaryVFFilter;
use crate::filters::author_socialgraph_filter::AuthorSocialgraphFilter;
use crate::filters::core_data_hydration_filter::CoreDataHydrationFilter;
use crate::filters::dedup_conversation_filter::DedupConversationFilter;
use crate::filters::drop_duplicates_filter::DropDuplicatesFilter;
use crate::filters::ineligible_subscription_filter::IneligibleSubscriptionFilter;
use crate::filters::muted_keyword_filter::MutedKeywordFilter;
use crate::filters::new_user_topic_ids_filter::NewUserTopicIdsFilter;
use crate::filters::previously_seen_posts_backup_filter::PreviouslySeenPostsBackupFilter;
use crate::filters::previously_seen_posts_filter::PreviouslySeenPostsFilter;
use crate::filters::previously_served_posts_filter::PreviouslyServedPostsFilter;
use crate::filters::retweet_deduplication_filter::RetweetDeduplicationFilter;
use crate::filters::self_tweet_filter::SelfTweetFilter;
use crate::filters::topic_ids_filter::TopicIdsFilter;
use crate::filters::vf_filter::VFFilter;
use crate::filters::video_filter::VideoFilter;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params;
use crate::query_hydrators::blocked_user_ids_query_hydrator::BlockedUserIdsQueryHydrator;
use crate::query_hydrators::cached_posts_query_hydrator::CachedPostsQueryHydrator;
use crate::query_hydrators::followed_grok_topics_query_hydrator::FollowedGrokTopicsQueryHydrator;
use crate::query_hydrators::followed_starter_packs_query_hydrator::FollowedStarterPacksQueryHydrator;
use crate::query_hydrators::followed_user_ids_query_hydrator::FollowedUserIdsQueryHydrator;
use crate::query_hydrators::ip_query_hydrator::IpQueryHydrator;
use crate::query_hydrators::impressed_posts_query_hydrator::ImpressedPostsQueryHydrator;
use crate::query_hydrators::impression_bloom_filter_query_hydrator::ImpressionBloomFilterQueryHydrator;
use crate::query_hydrators::inferred_grok_topics_query_hydrator::InferredGrokTopicsQueryHydrator;
use crate::query_hydrators::muted_user_ids_query_hydrator::MutedUserIdsQueryHydrator;
use crate::query_hydrators::mutual_follow_query_hydrator::MutualFollowQueryHydrator;
use crate::query_hydrators::retrieval_sequence_query_hydrator::RetrievalSequenceQueryHydrator;
use crate::query_hydrators::scoring_sequence_query_hydrator::ScoringSequenceQueryHydrator;
use crate::query_hydrators::subscribed_user_ids_query_hydrator::SubscribedUserIdsQueryHydrator;
use crate::query_hydrators::user_demographics_query_hydrator::UserDemographicsQueryHydrator;
use crate::query_hydrators::user_inferred_gender_query_hydrator::UserInferredGenderQueryHydrator;
use crate::scorers::phoenix_scorer::PhoenixScorer;
use crate::scorers::ranking_scorer::RankingScorer;
use crate::scorers::vm_ranker::VMRanker;
use crate::selectors::TopKScoreSelector;
use crate::side_effects::mutual_follow_stats_side_effect::MutualFollowStatsSideEffect;
use crate::side_effects::phoenix_experiments_side_effect::PhoenixExperimentsSideEffect;
use crate::side_effects::phoenix_request_cache_side_effect::PhoenixRequestCacheSideEffect;
use crate::side_effects::redis_post_candidate_cache_side_effect::RedisPostCandidateCacheSideEffect;
use crate::side_effects::reranking_kafka_side_effect::RerankingKafkaSideEffect;
use crate::side_effects::scored_stats_side_effect::ScoredStatsSideEffect;
use crate::sources::cached_posts_source::CachedPostsSource;
use crate::sources::phoenix_moe_source::PhoenixMOESource;
use crate::sources::phoenix_source::PhoenixSource;
use crate::sources::phoenix_topics_source::PhoenixTopicsSource;
use crate::sources::thunder_source::ThunderSource;
use crate::sources::tweet_mixer_source::TweetMixerSource;
use xai_candidate_pipeline::component_library::clients::{
    MockTweetMixerClient, ProdTweetMixerClient, TweetMixerClient,
};

use std::sync::Arc;
use std::time::Duration;
use tonic::async_trait;
use xai_candidate_pipeline::candidate_pipeline::CandidatePipeline;
use xai_candidate_pipeline::component_library::clients::ThunderClient;
use xai_candidate_pipeline::component_library::clients::egress_prediction_client::EgressPhoenixPredictionClient;
use xai_candidate_pipeline::component_library::clients::phoenix_prediction_client::{
    MockPredictClient, PhoenixPredictionClient, ProdPhoenixPredictionClient,
};
use xai_candidate_pipeline::component_library::clients::phoenix_retrieval_client::{
    MockRetrievalClient, PhoenixRetrievalClient, PhoenixRetrievalCluster,
    ProdPhoenixRetrievalClient,
};
use xai_candidate_pipeline::component_library::clients::redis_client::{
    MockRedisClient, RedisClient,
};
use xai_candidate_pipeline::component_library::clients::{
    ImpressionBloomFilterClient, MockImpressionBloomFilterClient, ProdImpressionBloomFilterClient,
};
use xai_candidate_pipeline::component_library::clients::{
    MockSocialGraphClient, SocialGraphClient, SocialGraphClientOps,
};
use xai_candidate_pipeline::component_library::clients::{
    MockStratoClient, ProdStratoClient, StratoClient,
};
use xai_candidate_pipeline::filter::Filter;
use xai_candidate_pipeline::hydrator::Hydrator;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;
use xai_candidate_pipeline::scorer::Scorer;
use xai_candidate_pipeline::selector::Selector;
use xai_candidate_pipeline::side_effect::SideEffect;
use xai_candidate_pipeline::source::Source;
use xai_geo_ip::GeoIpLocationClient;
use xai_redis_client::{XdsRedisClient, XdsRedisConfig};
use xai_visibility_filtering::vf_client::{
    MockVisibilityFilteringClient, ProdVisibilityFilteringClient, VisibilityFilteringClient,
};
use xai_visibility_filtering::vf_safety_labels_client::{MockVfClient, ProdVfClient, VfClient};
use xai_x_rpc::wily_lookup_service::ShardCoordinate;

pub struct PhoenixCandidatePipeline {
    query_hydrators: Vec<Box<dyn QueryHydrator<ScoredPostsQuery>>>,
    sources: Vec<Box<dyn Source<ScoredPostsQuery, PostCandidate>>>,
    hydrators: Vec<Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>>,
    filters: Vec<Box<dyn Filter<ScoredPostsQuery, PostCandidate>>>,
    scorers: Vec<Box<dyn Scorer<ScoredPostsQuery, PostCandidate>>>,
    selector: TopKScoreSelector,
    post_selection_hydrators: Vec<Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>>,
    post_selection_filters: Vec<Box<dyn Filter<ScoredPostsQuery, PostCandidate>>>,
    side_effects: Arc<Vec<Box<dyn SideEffect<ScoredPostsQuery, PostCandidate>>>>,
}

impl PhoenixCandidatePipeline {
    pub(crate) async fn build_with_clients(
        user_action_aggregation_client: Arc<dyn UserActionAggregationClient + Send + Sync>,
        phoenix_client: Arc<dyn PhoenixPredictionClient + Send + Sync>,
        egress_client: Arc<dyn PhoenixPredictionClient + Send + Sync>,
        phoenix_retrieval_client: Arc<dyn PhoenixRetrievalClient + Send + Sync>,
        thunder_client: Arc<ThunderClient>,
        strato_client: Arc<dyn StratoClient + Send + Sync>,
        tweet_mixer_client: Arc<dyn TweetMixerClient>,
        tes_client: Arc<dyn TESClient + Send + Sync>,
        gizmoduck_client: Arc<dyn GizmoduckClient + Send + Sync>,
        vf_client: Arc<dyn VisibilityFilteringClient + Send + Sync>,
        redis_client: Arc<dyn RedisClient + Send + Sync>,
        phoenix_kafka_client: Arc<dyn KafkaPublisherClient>,
        reranking_kafka_client: Arc<dyn KafkaPublisherClient>,
        socialgraph_client: Arc<dyn SocialGraphClientOps>,
        vm_ranker_client: Arc<dyn VMRankerClient>,
        safety_label_client: Arc<dyn xai_safety_label_store::SafetyLabelStoreClient>,
        vf_safety_labels_client: Arc<dyn VfClient>,
        phoenix_request_cache_redis_atla_client: Arc<dyn RedisClient + Send + Sync>,
        phoenix_request_cache_redis_pdxa_client: Arc<dyn RedisClient + Send + Sync>,
        impression_bloom_filter_client: Arc<dyn ImpressionBloomFilterClient>,
        ip_client: Arc<GeoIpLocationClient>,
        user_demographics_client: Arc<dyn UserDemographicsClient>,
        user_inferred_gender_store_client: Arc<dyn UserInferredGenderStoreClient>,
        user_inferred_gender_grpc_client: Arc<dyn GenderPredictionGrpcClient>,
        impressed_posts_client: Arc<dyn ImpressedPostsClient>,
        followed_grok_topics_client: Arc<dyn FollowedGrokTopicsStoreClient>,
        followed_starter_packs_client: Arc<dyn FollowedStarterPacksStoreClient>,
    ) -> PhoenixCandidatePipeline {
        let query_hydrators: Vec<Box<dyn QueryHydrator<ScoredPostsQuery>>> = vec![
            Box::new(ScoringSequenceQueryHydrator::new(
                user_action_aggregation_client.clone(),
            )),
            Box::new(RetrievalSequenceQueryHydrator::new(
                user_action_aggregation_client,
            )),
            Box::new(BlockedUserIdsQueryHydrator {
                socialgraph_client: socialgraph_client.clone(),
            }),
            Box::new(MutedUserIdsQueryHydrator {
                socialgraph_client: socialgraph_client.clone(),
            }),
            Box::new(FollowedUserIdsQueryHydrator {
                socialgraph_client: socialgraph_client.clone(),
            }),
            Box::new(SubscribedUserIdsQueryHydrator {
                socialgraph_client: socialgraph_client.clone(),
            }),
            Box::new(CachedPostsQueryHydrator {
                redis_client: redis_client.clone(),
            }),
            Box::new(MutualFollowQueryHydrator {
                strato_client: strato_client.clone(),
            }),
            Box::new(UserDemographicsQueryHydrator {
                client: user_demographics_client,
            }),
            Box::new(FollowedGrokTopicsQueryHydrator::new(
                followed_grok_topics_client,
            )),
            Box::new(FollowedStarterPacksQueryHydrator::new(
                followed_starter_packs_client,
            )),
            Box::new(InferredGrokTopicsQueryHydrator {
                strato_client: strato_client.clone(),
            }),
            Box::new(ImpressionBloomFilterQueryHydrator {
                client: impression_bloom_filter_client,
            }),
            Box::new(IpQueryHydrator {
                client: ip_client,
            }),
            Box::new(UserInferredGenderQueryHydrator::new(
                user_inferred_gender_store_client,
                user_inferred_gender_grpc_client,
            )),
        ];

        let _impressed_posts_hydrator = ImpressedPostsQueryHydrator {
            client: impressed_posts_client,
        };

        let phoenix_source = Box::new(PhoenixSource {
            phoenix_retrieval_client: phoenix_retrieval_client.clone(),
        });
        let phoenix_topics_source = Box::new(PhoenixTopicsSource {
            phoenix_retrieval_client: phoenix_retrieval_client.clone(),
        });
        let phoenix_moe_source = Box::new(PhoenixMOESource {
            phoenix_retrieval_client,
        });
        let thunder_source = Box::new(ThunderSource { thunder_client });
        let tweet_mixer_source = Box::new(TweetMixerSource { tweet_mixer_client });
        let cached_posts_source = Box::new(CachedPostsSource);
        let sources: Vec<Box<dyn Source<ScoredPostsQuery, PostCandidate>>> = vec![
            thunder_source,
            tweet_mixer_source,
            phoenix_source,
            phoenix_topics_source,
            phoenix_moe_source,
            cached_posts_source,
        ];

        let hydrators: Vec<Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>> = vec![
            Box::new(InNetworkCandidateHydrator),
            Box::new(CoreDataCandidateHydrator::new(tes_client.clone()).await),
            Box::new(QuoteHydrator::new(tes_client.clone(), socialgraph_client.clone()).await),
            Box::new(VideoDurationCandidateHydrator::new(tes_client.clone()).await),
            Box::new(HasMediaHydrator::new(tes_client.clone()).await),
            Box::new(SubscriptionHydrator::new(tes_client.clone()).await),
            Box::new(GizmoduckCandidateHydrator::new(gizmoduck_client).await),
            Box::new(BlockedByHydrator::new(socialgraph_client).await),
            Box::new(FilteredTopicsHydrator {
                strato_client: strato_client.clone(),
            }),
            Box::new(LanguageCodeHydrator::new(tes_client.clone()).await),
        ];

        let filters: Vec<Box<dyn Filter<ScoredPostsQuery, PostCandidate>>> = vec![
            Box::new(DropDuplicatesFilter),
            Box::new(CoreDataHydrationFilter),
            Box::new(AgeFilter::new(Duration::from_secs(params::MAX_POST_AGE))),
            Box::new(SelfTweetFilter),
            Box::new(RetweetDeduplicationFilter),
            Box::new(IneligibleSubscriptionFilter),
            Box::new(PreviouslySeenPostsFilter),
            Box::new(PreviouslySeenPostsBackupFilter),
            Box::new(PreviouslyServedPostsFilter),
            Box::new(MutedKeywordFilter::new()),
            Box::new(AuthorSocialgraphFilter),
            Box::new(VideoFilter),
            Box::new(TopicIdsFilter),
            Box::new(NewUserTopicIdsFilter),
        ];

        let phoenix_scorer = Box::new(PhoenixScorer {
            phoenix_client: phoenix_client.clone(),
            egress_client: Arc::clone(&egress_client),
        });
        let ranking_scorer = Box::new(RankingScorer);
        let vm_ranker = Box::new(VMRanker {
            client: vm_ranker_client,
        });
        let scorers: Vec<Box<dyn Scorer<ScoredPostsQuery, PostCandidate>>> =
            vec![phoenix_scorer, ranking_scorer, vm_ranker];

        let selector = TopKScoreSelector;

        let post_selection_hydrators: Vec<Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>> = vec![
            Box::new(VFCandidateHydrator::new(vf_client.clone()).await),
            Box::new(AdsBrandSafetyHydrator::new(safety_label_client)),
            Box::new(AdsBrandSafetyVfHydrator {
                client: vf_safety_labels_client,
            }),
            Box::new(TweetTypeMetricsHydrator::new()),
            Box::new(FollowingRepliedUsersHydrator),
            Box::new(MutualFollowJaccardHydrator {
                strato_client: strato_client.clone(),
            }),
        ];

        let post_selection_filters: Vec<Box<dyn Filter<ScoredPostsQuery, PostCandidate>>> = vec![
            Box::new(VFFilter),
            Box::new(AncillaryVFFilter),
            Box::new(DedupConversationFilter),
        ];

        let side_effects: Arc<Vec<Box<dyn SideEffect<ScoredPostsQuery, PostCandidate>>>> =
            Arc::new(vec![
                Box::new(PhoenixExperimentsSideEffect::new(
                    phoenix_client,
                    egress_client,
                    phoenix_kafka_client,
                )),
                Box::new(RerankingKafkaSideEffect::new(reranking_kafka_client)),
                Box::new(RedisPostCandidateCacheSideEffect::new(redis_client)),
                Box::new(ScoredStatsSideEffect),
                Box::new(MutualFollowStatsSideEffect),
                Box::new(PhoenixRequestCacheSideEffect::new(
                    phoenix_request_cache_redis_atla_client,
                    phoenix_request_cache_redis_pdxa_client,
                )),
            ]);

        PhoenixCandidatePipeline {
            query_hydrators,
            hydrators,
            filters,
            sources,
            scorers,
            selector,
            post_selection_hydrators,
            post_selection_filters,
            side_effects,
        }
    }

    pub async fn prod(
        shard_coordinate: Option<ShardCoordinate>,
        datacenter: &str,
    ) -> PhoenixCandidatePipeline {
        let local_cache_eds = String::new();
        let atla_phoenix_cache_eds = "";
        let pdxa_phoenix_cache_eds = "";

        let (
            flock_socialgraph_client,
            user_action_aggregation_client,
            phoenix_client,
            egress_client,
            phoenix_retrieval_client,
            thunder_client,
            strato_client,
            tweet_mixer_client,
            tes_client,
            gizmoduck_client,
            vf_client,
            redis_client,
            phoenix_request_cache_redis_atla_client,
            phoenix_request_cache_redis_pdxa_client,
            phoenix_kafka_client,
            reranking_kafka_client,
            vm_ranker_client,
            safety_label_client,
            vf_safety_labels_client,
            impression_bloom_filter_client,
            ip_client,
            user_demographics_client,
            user_inferred_gender_store_client,
            user_inferred_gender_grpc_client,
            impressed_posts_client,
            followed_grok_topics_client,
            followed_starter_packs_client,
        ) = tokio::join!(
            async {
                Arc::new(
                    SocialGraphClient::new(
                        datacenter,
                        &S2S_CHAIN_PATH,
                        &S2S_CRT_PATH,
                        &S2S_KEY_PATH,
                    )
                    .await
                    .expect("Failed to create flock SocialGraphClient"),
                ) as Arc<dyn SocialGraphClientOps>
            },
            async {
                Arc::new(
                    ProdUserActionAggregationClient::new()
                        .await
                        .expect("Failed to create User Action Aggregation client"),
                ) as Arc<dyn UserActionAggregationClient + Send + Sync>
            },
            async {
                Arc::new(
                    ProdPhoenixPredictionClient::new()
                        .await
                        .expect("Failed to create Phoenix prediction client"),
                ) as Arc<dyn PhoenixPredictionClient + Send + Sync>
            },
            async {
                Arc::new(
                    EgressPhoenixPredictionClient::connect()
                        .await
                        .expect("Failed to connect to egress sidecar"),
                ) as Arc<dyn PhoenixPredictionClient + Send + Sync>
            },
            async {
                Arc::new(
                    ProdPhoenixRetrievalClient::new(Some((
                        PhoenixRetrievalCluster::Experiment1Fou,
                        PhoenixRetrievalCluster::Experiment1Lap7,
                    )))
                    .await
                    .expect("Failed to create Phoenix retrieval client"),
                ) as Arc<dyn PhoenixRetrievalClient + Send + Sync>
            },
            async { Arc::new(ThunderClient::new().await) },
            async {
                Arc::new(
                    ProdStratoClient::new(shard_coordinate, datacenter)
                        .await
                        .expect("Failed to create Strato client"),
                ) as Arc<dyn StratoClient + Send + Sync>
            },
            async {
                Arc::new(
                    ProdTweetMixerClient::new(datacenter)
                        .await
                        .expect("Failed to create TweetMixer client"),
                ) as Arc<dyn TweetMixerClient>
            },
            async {
                Arc::new(
                    ProdTESClient::new(shard_coordinate, datacenter)
                        .await
                        .expect("Failed to create TES client"),
                ) as Arc<dyn TESClient + Send + Sync>
            },
            async {
                Arc::new(
                    ProdGizmoduckClient::new(
                        shard_coordinate,
                        datacenter,
                        Some("home-mixer.prod".to_string()),
                    )
                    .await
                    .expect("Failed to create Gizmoduck client"),
                ) as Arc<dyn GizmoduckClient + Send + Sync>
            },
            async {
                Arc::new(
                    ProdVisibilityFilteringClient::new(
                        S2S_CHAIN_PATH.clone(),
                        S2S_CRT_PATH.clone(),
                        S2S_KEY_PATH.clone(),
                        "home-mixer.prod".to_string(),
                        datacenter.to_string(),
                    )
                    .await
                    .expect("Failed to create VF client"),
                ) as Arc<dyn VisibilityFilteringClient + Send + Sync>
            },
            async {
                Arc::new(
                    XdsRedisClient::new(XdsRedisConfig {
                        eds_resource_name: local_cache_eds.clone(),
                    })
                    .await
                    .expect("Failed to create xDS Redis client for local cache"),
                ) as Arc<dyn RedisClient + Send + Sync>
            },
            async {
                Arc::new(
                    XdsRedisClient::new(XdsRedisConfig {
                        eds_resource_name: atla_phoenix_cache_eds.into(),
                    })
                    .await
                    .expect("Failed to create xDS Redis client for atla phoenix cache"),
                ) as Arc<dyn RedisClient + Send + Sync>
            },
            async {
                Arc::new(
                    XdsRedisClient::new(XdsRedisConfig {
                        eds_resource_name: pdxa_phoenix_cache_eds.into(),
                    })
                    .await
                    .expect("Failed to create xDS Redis client for pdxa phoenix cache"),
                ) as Arc<dyn RedisClient + Send + Sync>
            },
            async {
                Arc::new(
                    ProdKafkaPublisherClient::new(PHOENIX_SCORES_TOPIC, KafkaCluster::Aiml).await,
                ) as Arc<dyn KafkaPublisherClient>
            },
            async {
                Arc::new(
                    ProdKafkaPublisherClient::new(RERANKING_TOPIC, KafkaCluster::Phoenix).await,
                ) as Arc<dyn KafkaPublisherClient>
            },
            async {
                Arc::new(
                    ProdVMRankerClient::new()
                        .await
                        .expect("Failed to create VMRanker client"),
                ) as Arc<dyn VMRankerClient>
            },
            async {
                let s2s = xai_manhattan::s2s::S2sConfig {
                    client_cert_path: S2S_CRT_PATH.clone(),
                    client_key_path: S2S_KEY_PATH.clone(),
                    ca_cert_path: S2S_CHAIN_PATH.clone(),
                };
                Arc::new(
                    xai_safety_label_store::ProdSafetyLabelStoreClient::new(datacenter, s2s)
                        .await
                        .expect("Failed to create SafetyLabelStore client"),
                ) as Arc<dyn xai_safety_label_store::SafetyLabelStoreClient>
            },
            async {
                Arc::new(
                    ProdVfClient::new(datacenter)
                        .await
                        .expect("Failed to create VF SafetyLabels client")
                        .with_timeout_ms(500)
                        .with_max_batch_size(150),
                ) as Arc<dyn VfClient>
            },
            async {
                Arc::new(
                    ProdImpressionBloomFilterClient::new(datacenter)
                        .await
                        .expect("Failed to create ImpressionBloomFilter client"),
                ) as Arc<dyn ImpressionBloomFilterClient>
            },
            async {
                Arc::new(
                    GeoIpLocationClient::new(
                        &S2S_CHAIN_PATH,
                        &S2S_CRT_PATH,
                        &S2S_KEY_PATH,
                        datacenter,
                    )
                    .await
                    .expect("Failed to create GeoIpLocationClient"),
                )
            },
            async {
                let s2s = xai_manhattan::s2s::S2sConfig {
                    client_cert_path: S2S_CRT_PATH.clone(),
                    client_key_path: S2S_KEY_PATH.clone(),
                    ca_cert_path: S2S_CHAIN_PATH.clone(),
                };
                Arc::new(
                    ProdUserDemographicsClient::new(datacenter, s2s)
                        .await
                        .expect("Failed to create UserDemographics client"),
                ) as Arc<dyn UserDemographicsClient>
            },
            async {
                let s2s = xai_manhattan::s2s::S2sConfig {
                    client_cert_path: S2S_CRT_PATH.clone(),
                    client_key_path: S2S_KEY_PATH.clone(),
                    ca_cert_path: S2S_CHAIN_PATH.clone(),
                };
                Arc::new(
                    ProdUserInferredGenderStoreClient::new(datacenter, s2s)
                        .await
                        .expect("Failed to create UserInferredGenderStore client"),
                ) as Arc<dyn UserInferredGenderStoreClient>
            },
            async {
                Arc::new(
                    ProdGenderPredictionGrpcClient::new()
                        .await
                        .expect("Failed to create GenderPredictionGrpcClient"),
                ) as Arc<dyn GenderPredictionGrpcClient>
            },
            async {
                Arc::new(
                    crate::clients::impressed_posts_client::ProdImpressedPostsClient::new(
                        datacenter,
                    )
                    .await
                    .expect("Failed to create ImpressedPosts client"),
                ) as Arc<dyn ImpressedPostsClient>
            },
            async {
                let s2s = xai_manhattan::s2s::S2sConfig {
                    client_cert_path: S2S_CRT_PATH.clone(),
                    client_key_path: S2S_KEY_PATH.clone(),
                    ca_cert_path: S2S_CHAIN_PATH.clone(),
                };
                Arc::new(
                    ProdFollowedGrokTopicsStoreClient::new(datacenter, s2s)
                        .await
                        .expect("Failed to create FollowedGrokTopicsStore client"),
                ) as Arc<dyn FollowedGrokTopicsStoreClient>
            },
            async {
                let s2s = xai_manhattan::s2s::S2sConfig {
                    client_cert_path: S2S_CRT_PATH.clone(),
                    client_key_path: S2S_KEY_PATH.clone(),
                    ca_cert_path: S2S_CHAIN_PATH.clone(),
                };
                Arc::new(
                    ProdFollowedStarterPacksStoreClient::new(datacenter, s2s)
                        .await
                        .expect("Failed to create FollowedStarterPacksStore client"),
                ) as Arc<dyn FollowedStarterPacksStoreClient>
            },
        );

        PhoenixCandidatePipeline::build_with_clients(
            user_action_aggregation_client,
            phoenix_client,
            egress_client,
            phoenix_retrieval_client,
            thunder_client,
            strato_client,
            tweet_mixer_client,
            tes_client,
            gizmoduck_client,
            vf_client,
            redis_client,
            phoenix_kafka_client,
            reranking_kafka_client,
            flock_socialgraph_client,
            vm_ranker_client,
            safety_label_client,
            vf_safety_labels_client,
            phoenix_request_cache_redis_atla_client,
            phoenix_request_cache_redis_pdxa_client,
            impression_bloom_filter_client,
            ip_client,
            user_demographics_client,
            user_inferred_gender_store_client,
            user_inferred_gender_grpc_client,
            impressed_posts_client,
            followed_grok_topics_client,
            followed_starter_packs_client,
        )
        .await
    }

    pub async fn mock() -> PhoenixCandidatePipeline {
        let user_action_aggregation_client = Arc::new(MockUserActionAggregationClient);
        let phoenix_client = Arc::new(MockPredictClient);
        let phoenix_retrieval_client = Arc::new(MockRetrievalClient);
        let thunder_client = Arc::new(ThunderClient::mock());
        let strato_client = Arc::new(MockStratoClient::default());
        let tweet_mixer_client: Arc<dyn TweetMixerClient> = Arc::new(MockTweetMixerClient);
        let tes_client = Arc::new(MockTESClient::default());
        let gizmoduck_client = Arc::new(MockGizmoduckClient::default());
        let vf_client = Arc::new(MockVisibilityFilteringClient);
        let redis_client = Arc::new(MockRedisClient::default());
        let kafka_client: Arc<dyn KafkaPublisherClient> = Arc::new(MockKafkaPublisherClient);
        let reranking_kafka_client: Arc<dyn KafkaPublisherClient> =
            Arc::new(MockKafkaPublisherClient);
        let mock_socialgraph: Arc<dyn SocialGraphClientOps> = Arc::new(MockSocialGraphClient);
        let vm_ranker_client: Arc<dyn VMRankerClient> = Arc::new(MockVMRankerClient);
        let safety_label_client: Arc<dyn xai_safety_label_store::SafetyLabelStoreClient> =
            Arc::new(xai_safety_label_store::MockSafetyLabelStoreClient);
        let vf_safety_labels_client: Arc<dyn VfClient> = Arc::new(MockVfClient);
        let phoenix_request_cache_redis_atla_client = Arc::new(MockRedisClient::default());
        let phoenix_request_cache_redis_pdxa_client: Arc<dyn RedisClient + Send + Sync> =
            Arc::new(MockRedisClient::default());
        let impression_bloom_filter_client: Arc<dyn ImpressionBloomFilterClient> =
            Arc::new(MockImpressionBloomFilterClient::default());
        let ip_client = Arc::new(GeoIpLocationClient::mock());
        let user_demographics_client: Arc<dyn UserDemographicsClient> =
            Arc::new(MockUserDemographicsClient);
        let user_inferred_gender_store_client: Arc<dyn UserInferredGenderStoreClient> =
            Arc::new(MockUserInferredGenderStoreClient);
        let user_inferred_gender_grpc_client: Arc<dyn GenderPredictionGrpcClient> =
            Arc::new(MockGenderPredictionGrpcClient);
        let impressed_posts_client: Arc<dyn ImpressedPostsClient> =
            Arc::new(crate::clients::impressed_posts_client::MockImpressedPostsClient::default());
        let followed_grok_topics_client: Arc<dyn FollowedGrokTopicsStoreClient> =
            Arc::new(MockFollowedGrokTopicsStoreClient);
        let followed_starter_packs_client: Arc<dyn FollowedStarterPacksStoreClient> =
            Arc::new(MockFollowedStarterPacksStoreClient);
        PhoenixCandidatePipeline::build_with_clients(
            user_action_aggregation_client,
            phoenix_client.clone(),
            phoenix_client,
            phoenix_retrieval_client,
            thunder_client,
            strato_client,
            tweet_mixer_client,
            tes_client,
            gizmoduck_client,
            vf_client,
            redis_client,
            kafka_client,
            reranking_kafka_client,
            mock_socialgraph,
            vm_ranker_client,
            safety_label_client,
            vf_safety_labels_client,
            phoenix_request_cache_redis_atla_client,
            phoenix_request_cache_redis_pdxa_client,
            impression_bloom_filter_client,
            ip_client,
            user_demographics_client,
            user_inferred_gender_store_client,
            user_inferred_gender_grpc_client,
            impressed_posts_client,
            followed_grok_topics_client,
            followed_starter_packs_client,
        )
        .await
    }
}

#[async_trait]
impl CandidatePipeline<ScoredPostsQuery, PostCandidate> for PhoenixCandidatePipeline {
    fn query_hydrators(&self) -> &[Box<dyn QueryHydrator<ScoredPostsQuery>>] {
        &self.query_hydrators
    }

    fn sources(&self) -> &[Box<dyn Source<ScoredPostsQuery, PostCandidate>>] {
        &self.sources
    }
    fn hydrators(&self) -> &[Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>] {
        &self.hydrators
    }

    fn filters(&self) -> &[Box<dyn Filter<ScoredPostsQuery, PostCandidate>>] {
        &self.filters
    }

    fn scorers(&self) -> &[Box<dyn Scorer<ScoredPostsQuery, PostCandidate>>] {
        &self.scorers
    }

    fn selector(&self) -> &dyn Selector<ScoredPostsQuery, PostCandidate> {
        &self.selector
    }

    fn post_selection_hydrators(&self) -> &[Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>] {
        &self.post_selection_hydrators
    }

    fn post_selection_filters(&self) -> &[Box<dyn Filter<ScoredPostsQuery, PostCandidate>>] {
        &self.post_selection_filters
    }

    fn side_effects(&self) -> Arc<Vec<Box<dyn SideEffect<ScoredPostsQuery, PostCandidate>>>> {
        Arc::clone(&self.side_effects)
    }

    fn result_size(&self) -> usize {
        params::RESULT_SIZE
    }
}

