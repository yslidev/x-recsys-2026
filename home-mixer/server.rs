use crate::candidate_pipeline::for_you_candidate_pipeline::ForYouCandidatePipeline;
use crate::candidate_pipeline::phoenix_candidate_pipeline::PhoenixCandidatePipeline;
use crate::clients::gizmoduck_client::{
    GizmoduckClient, MockGizmoduckClient, ProdGizmoduckClient, ViewerData,
};
use crate::for_you_server::ForYouFeedServer;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params;
use crate::scored_posts_server::{ScoredPostsServer, build_debug_json};
use std::collections::HashMap;
use std::sync::Arc;
use tonic::codec::CompressionEncoding;
use tonic::{Request, Response, Status};
use tracing::{Instrument, info_span};
use xai_candidate_pipeline::candidate_pipeline::PipelineResult;
use xai_candidate_pipeline::component_library::utils::{
    days_since_creation, generate_request_id, is_sampled, non_zero, resolve_request_id,
};
use xai_decider::{Decider, DeciderStore};
use xai_feature_switches::{FeatureSwitches, RecipientBuilder};
use xai_home_mixer_proto as pb;
use xai_home_mixer_proto::{
    DebugScoredPostsResponse, ForYouFeedResponse, ForYouFeedUrtResponse, ScoredPost,
    ScoredPostsResponse,
};
use xai_pipeline_tracing::{B3RequestInfo, extract_b3_info};
use xai_recsys_proto::{network_type_string_to_enum, timezone_string_to_enum};
use xai_urt_thrift::cursor_utils;
use xai_urt_thrift::operation::CursorType;
use xai_x_rpc::wily_lookup_service::ShardCoordinate;

const VIEWER_ROLES_TIMEOUT_MS: u64 = 200;

pub struct RequestContext {
    pub b3_info: B3RequestInfo,
    pub query: ScoredPostsQuery,
    pub root_span: tracing::Span,
}

pub(crate) struct PipelineOutput {
    pub scored_posts: Vec<ScoredPost>,
    pub pipeline_result: PipelineResult<ScoredPostsQuery, PostCandidate>,
}

pub struct HomeMixerConfig {
    pub shard_coordinate: Option<ShardCoordinate>,
}

#[derive(Clone)]
pub struct QueryBuilder {
    feature_switches: Arc<FeatureSwitches>,
    decider: Decider,
    datacenter: String,
    gizmoduck_client: Arc<dyn GizmoduckClient + Send + Sync>,
}

impl QueryBuilder {
    pub async fn build(
        &self,
        mut b3_info: B3RequestInfo,
        proto_query: pb::ScoredPostsQuery,
        fs_overrides: std::collections::HashMap<String, String>,
        span_name: &'static str,
    ) -> Result<RequestContext, Status> {
        if proto_query.viewer_id == 0 {
            return Err(Status::invalid_argument("viewer_id must be specified"));
        }
        if params::TRACE_USER_IDS.contains(&proto_query.viewer_id) {
            b3_info.force_sample();
        }

        let viewer_data = self.fetch_viewer_data(proto_query.viewer_id).await;

        let in_network_only =
            proto_query.in_network_only || viewer_data.allow_for_you_recommendations == Some(false);

        let params = self.evaluate_feature_switches(
            &proto_query,
            &viewer_data.roles,
            viewer_data.has_phone_number,
            &fs_overrides,
        );

        let push_to_home_post_id = non_zero(proto_query.push_to_home_post_id);
        let device_status = proto_query.device_status.unwrap_or_default();
        let prediction_id = generate_request_id();
        let request_id = resolve_request_id(proto_query.request_id);
        let query = ScoredPostsQuery::new(
            proto_query.viewer_id,
            proto_query.client_app_id,
            proto_query.country_code,
            proto_query.language_code,
            proto_query.seen_ids,
            proto_query.served_ids,
            in_network_only,
            proto_query.is_bottom_request,
            params,
            self.decider.with_recipient(proto_query.viewer_id),
            viewer_data.roles,
            viewer_data.muted_keywords,
            viewer_data.follower_count,
            proto_query.topic_ids,
            proto_query.excluded_topic_ids,
            proto_query.exclude_videos,
            request_id,
            prediction_id,
            device_status.ip_address,
            device_status.user_agent,
            timezone_string_to_enum(device_status.time_zone.as_ref()),
            network_type_string_to_enum(device_status.device_network_type.as_ref()),
            device_status.client_version,
            device_status.device_id,
            device_status.mobile_device_id,
            device_status.mobile_device_ad_id,
            viewer_data.subscription_level,
            is_sampled(request_id, 0.5),
            proto_query.is_preview,
            viewer_data.age_in_years,
            push_to_home_post_id,
        );

        let root_span = b3_info.root_span(info_span!(
            "request",
            endpoint = span_name,
            trace = %b3_info.trace_id_str,
            user = %query.user_id,
            b3 = %b3_info.b3_sampled,
        ));

        Ok(RequestContext {
            b3_info,
            query,
            root_span,
        })
    }

    fn evaluate_feature_switches(
        &self,
        proto_query: &pb::ScoredPostsQuery,
        user_roles: &[String],
        has_phone_number: bool,
        fs_overrides: &std::collections::HashMap<String, String>,
    ) -> xai_feature_switches::Params {
        let recipient = RecipientBuilder::new()
            .user_id(proto_query.viewer_id)
            .country(&proto_query.country_code)
            .language(&proto_query.language_code)
            .client_app_id(proto_query.client_app_id as i64)
            .custom_string("datacenter", &self.datacenter)
            .custom_i64(
                "account_age_days",
                days_since_creation(proto_query.viewer_id),
            )
            .custom_bool("has_phone_number", has_phone_number);
        let recipient = if !user_roles.is_empty() {
            recipient.user_roles(user_roles.iter().cloned())
        } else {
            recipient
        };
        let mut results = self.feature_switches.match_recipient(&recipient.build());

        if !fs_overrides.is_empty() {
            for (key, value) in fs_overrides {
                results.override_fs(key.clone(), value);
            }
            tracing::info!(
                "Applied {} FS overrides: {:?}",
                fs_overrides.len(),
                fs_overrides.keys().collect::<Vec<_>>()
            );
        }

        results.into()
    }

    async fn fetch_viewer_data(&self, viewer_id: u64) -> ViewerData {
        match tokio::time::timeout(
            std::time::Duration::from_millis(VIEWER_ROLES_TIMEOUT_MS),
            self.gizmoduck_client.get_viewer_data(viewer_id),
        )
        .await
        {
            Ok(Ok(data)) => data,
            Ok(Err(_)) | Err(_) => ViewerData::default(),
        }
    }

    pub fn mock() -> Self {
        Self {
            feature_switches: Arc::new(FeatureSwitches::new(vec![]).unwrap()),
            decider: Decider::new(DeciderStore::new(HashMap::new())),
            datacenter: "mock".to_string(),
            gizmoduck_client: Arc::new(MockGizmoduckClient::default()),
        }
    }
}

pub struct HomeMixerServer {
    scored_posts: Arc<ScoredPostsServer>,
    for_you: Arc<ForYouFeedServer>,
}


#[tonic::async_trait]
impl pb::scored_posts_service_server::ScoredPostsService for ScoredPostsServer {
    #[xai_stats_macro::receive_stats(latency=Bucket500To2500)]
    async fn get_scored_posts(
        &self,
        request: Request<pb::ScoredPostsQuery>,
    ) -> Result<Response<ScoredPostsResponse>, Status> {
        let b3_info = extract_b3_info(request.metadata());
        let ctx = self
            .query_builder
            .build(
                b3_info,
                request.into_inner(),
                Default::default(),
                "scored_posts",
            )
            .await?;
        let RequestContext {
            b3_info,
            query,
            root_span,
        } = ctx;
        let output = self.run_pipeline(query).instrument(root_span).await?;

        let mut response = Response::new(ScoredPostsResponse {
            scored_posts: output.scored_posts,
        });
        b3_info.inject_trace_response_header(&mut response);
        Ok(response)
    }

    #[xai_stats_macro::receive_stats(latency=Bucket500To2500)]
    async fn get_debug_scored_posts(
        &self,
        request: Request<pb::DebugScoredPostsQuery>,
    ) -> Result<Response<DebugScoredPostsResponse>, Status> {
        let mut b3_info = extract_b3_info(request.metadata());
        b3_info.force_sample();

        let debug_query = request.into_inner();
        let fs_overrides = debug_query.feature_switch_overrides;
        let proto_query = debug_query.query.unwrap_or_default();

        let ctx = self
            .query_builder
            .build(b3_info, proto_query, fs_overrides, "debug_scored_posts")
            .await?;
        let RequestContext {
            b3_info,
            query,
            root_span,
        } = ctx;
        let output = self.run_pipeline(query).instrument(root_span).await?;

        let debug_json = build_debug_json(&output.pipeline_result);

        let mut response = Response::new(DebugScoredPostsResponse {
            scored_posts: output.scored_posts,
            debug_json,
        });
        b3_info.inject_trace_response_header(&mut response);
        Ok(response)
    }
}

#[tonic::async_trait]
impl pb::for_you_feed_service_server::ForYouFeedService for ForYouFeedServer {
    #[xai_stats_macro::receive_stats(latency=Bucket500To2500)]
    async fn get_for_you_feed(
        &self,
        request: Request<pb::ForYouFeedQuery>,
    ) -> Result<Response<ForYouFeedResponse>, Status> {
        let b3_info = extract_b3_info(request.metadata());
        let feed_query = request.into_inner();
        let proto_query = feed_query
            .query
            .ok_or_else(|| Status::invalid_argument("query must be specified"))?;
        let ctx = self
            .query_builder
            .build(b3_info, proto_query, Default::default(), "for_you_feed")
            .await?;
        let RequestContext {
            b3_info,
            query,
            root_span,
        } = ctx;
        let output = self.get_for_you_feed(query).instrument(root_span).await?;

        let mut response = Response::new(ForYouFeedResponse {
            items: output.items,
        });
        b3_info.inject_trace_response_header(&mut response);
        Ok(response)
    }

    #[xai_stats_macro::receive_stats(latency=Bucket500To2500)]
    async fn get_for_you_feed_urt(
        &self,
        request: Request<pb::ForYouFeedQuery>,
    ) -> Result<Response<ForYouFeedUrtResponse>, Status> {
        let b3_info = extract_b3_info(request.metadata());
        let feed_query = request.into_inner();
        let proto_query = feed_query
            .query
            .ok_or_else(|| Status::invalid_argument("query must be specified"))?;
        let cursor_str = proto_query.cursor.clone();
        let request_context = proto_query.request_context.clone();
        let is_polling = proto_query.is_polling;
        let ctx = self
            .query_builder
            .build(b3_info, proto_query, Default::default(), "for_you_feed_urt")
            .await?;
        let RequestContext {
            b3_info,
            mut query,
            root_span,
        } = ctx;

        query.request_context = request_context;
        query.is_polling = is_polling;
        if !cursor_str.is_empty() {
            match cursor_utils::decode_ordered_cursor(&cursor_str) {
                Ok(Some(c)) => {
                    query.is_bottom_request = c.cursor_type == Some(CursorType::BOTTOM);
                    query.is_top_request = c.cursor_type == Some(CursorType::TOP);
                    query.cursor = Some(c);
                }
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!(cursor_str, error = %e, "failed to decode URT cursor, ignoring");
                }
            }
        }
        let urt = self
            .get_for_you_feed_urt(query)
            .instrument(root_span)
            .await?;

        let mut response = Response::new(ForYouFeedUrtResponse { urt: urt.into() });
        b3_info.inject_trace_response_header(&mut response);
        Ok(response)
    }
}

#[tonic::async_trait]
impl xai_x_service_builder::XService for HomeMixerServer {
    type Config = HomeMixerConfig;

    async fn build(ctx: xai_x_service_builder::ServiceContext<HomeMixerConfig>) -> Self {
        let xai_x_service_builder::ServiceContext {
            feature_switches,
            decider,
            datacenter,
            config,
        } = ctx;

        let gizmoduck_client: Arc<dyn GizmoduckClient + Send + Sync> = Arc::new(
            ProdGizmoduckClient::new(
                config.shard_coordinate,
                &datacenter,
                Some("home-mixer.prod".to_string()),
            )
            .await
            .expect("Failed to create Gizmoduck client"),
        );

        let query_builder = QueryBuilder {
            feature_switches,
            decider,
            datacenter: datacenter.clone(),
            gizmoduck_client,
        };

        let phoenix_candidate_pipeline =
            Arc::new(PhoenixCandidatePipeline::prod(config.shard_coordinate, &datacenter).await);

        let scored_posts = Arc::new(ScoredPostsServer::new(
            query_builder.clone(),
            phoenix_candidate_pipeline,
        ));

        let for_you_pipeline =
            ForYouCandidatePipeline::new(Arc::clone(&scored_posts), &datacenter).await;

        let for_you = Arc::new(ForYouFeedServer::new(query_builder, for_you_pipeline));

        HomeMixerServer {
            scored_posts,
            for_you,
        }
    }

    fn register(self: Arc<Self>, routes: &mut tonic::service::RoutesBuilder) {
        routes.add_service(
            pb::scored_posts_service_server::ScoredPostsServiceServer::from_arc(Arc::clone(
                &self.scored_posts,
            ))
            .max_decoding_message_size(params::MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(params::MAX_GRPC_MESSAGE_SIZE)
            .accept_compressed(CompressionEncoding::Gzip)
            .accept_compressed(CompressionEncoding::Zstd)
            .send_compressed(CompressionEncoding::Gzip)
            .send_compressed(CompressionEncoding::Zstd),
        );
        routes.add_service(
            pb::for_you_feed_service_server::ForYouFeedServiceServer::from_arc(Arc::clone(
                &self.for_you,
            ))
            .max_decoding_message_size(params::MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(params::MAX_GRPC_MESSAGE_SIZE)
            .accept_compressed(CompressionEncoding::Gzip)
            .accept_compressed(CompressionEncoding::Zstd)
            .send_compressed(CompressionEncoding::Gzip)
            .send_compressed(CompressionEncoding::Zstd),
        );
    }
}
