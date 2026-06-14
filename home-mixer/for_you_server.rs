use crate::candidate_pipeline::for_you_candidate_pipeline::ForYouCandidatePipeline;
use crate::models::query::ScoredPostsQuery;
use crate::params;
use crate::server::QueryBuilder;
use crate::util::urt;
use tonic::Status;
use tracing::info;
use xai_candidate_pipeline::candidate_pipeline::CandidatePipeline;
use xai_home_mixer_proto::FeedItem;

pub(crate) struct ForYouFeedOutput {
    pub items: Vec<FeedItem>,
}

pub struct ForYouFeedServer {
    pub(crate) query_builder: QueryBuilder,
    pipeline: ForYouCandidatePipeline,
}

impl ForYouFeedServer {
    pub fn new(query_builder: QueryBuilder, pipeline: ForYouCandidatePipeline) -> Self {
        Self {
            query_builder,
            pipeline,
        }
    }

    pub(crate) async fn get_for_you_feed(
        &self,
        query: ScoredPostsQuery,
    ) -> Result<ForYouFeedOutput, Status> {
        if params::TEST_USER_IDS.contains(&query.user_id) {
            return Ok(ForYouFeedOutput { items: vec![] });
        }

        let result = self.pipeline.execute(query).await;

        Ok(ForYouFeedOutput {
            items: result.selected_candidates,
        })
    }

    pub(crate) async fn get_for_you_feed_urt(
        &self,
        query: ScoredPostsQuery,
    ) -> Result<Vec<u8>, Status> {
        log_request_info(&query);

        let cursor = query.cursor.clone();
        let request_context = query.request_context.clone();
        let client_app_id = query.client_app_id;
        let viewer_id = query.user_id;
        let language_code = query.language_code.clone();
        let country_code = query.country_code.clone();

        let output = self.get_for_you_feed(query).await?;

        let timeline_response = urt::make_urt_timeline(
            &output.items,
            cursor.as_ref(),
            &request_context,
            client_app_id,
            viewer_id,
            &language_code,
            if country_code.is_empty() {
                None
            } else {
                Some(&country_code)
            },
        );

        xai_urt_thrift::serialize_binary(&timeline_response)
            .map_err(|e| Status::internal(format!("failed to serialize URT: {e}")))
    }
}

fn log_request_info(query: &ScoredPostsQuery) {
    info!(
        request_id = query.request_id,
        user_id = query.user_id,
        app_id = query.client_app_id,
        request_context = %query.request_context,
        cursor = ?query.cursor,
        seen_ids = query.seen_ids.len(),
        "For You Feed URT request -"
    );
}
