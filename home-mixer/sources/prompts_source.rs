use crate::clients::prompts_client::PromptsClient;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnablePrompts;
use std::collections::BTreeSet;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto::{FeedItem, Prompt, feed_item};
use xai_prompts_thrift::injection_service::{
    ClientContext, DisplayContext, DisplayLocation, GetInjectionsRequest, PromptType,
    RequestTargetingContext,
};

pub struct PromptsSource {
    pub prompts_client: Arc<dyn PromptsClient + Send + Sync>,
}

#[async_trait]
impl Source<ScoredPostsQuery, FeedItem> for PromptsSource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnablePrompts)
    }

    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<FeedItem>, String> {
        let request = build_get_injections_request(query);

        let injections = self
            .prompts_client
            .get_injections(request)
            .await
            .map_err(|e| format!("PromptsSource: {e}"))?;

        injections
            .into_iter()
            .map(|injection| {
                let bytes = xai_prompts_thrift::serialize_binary(&injection)
                    .map_err(|e| format!("PromptsSource: serialization failed: {e}"))?;
                Ok(FeedItem {
                    position: 0,
                    item: Some(feed_item::Item::Prompt(Prompt {
                        injection: bytes.into(),
                    })),
                })
            })
            .collect()
    }
}

fn build_get_injections_request(query: &ScoredPostsQuery) -> GetInjectionsRequest {
    let client_context = ClientContext {
        user_id: Some(query.user_id as i64),
        client_application_id: Some(query.client_app_id as i64),
        device_id: if query.device_id.is_empty() {
            None
        } else {
            Some(query.device_id.clone())
        },
        country_code: Some(query.country_code.clone()),
        language_code: Some(query.language_code.clone()),
        user_agent: Some(query.user_agent.clone()),
        ip_address: Some(query.ip_address.clone()),
    };

    let display_context = DisplayContext {
        display_location: DisplayLocation::HOME_TIMELINE,
        timeline_id: None,
    };

    let request_targeting_context = RequestTargetingContext {
        ranking_disabler_with_latest_controls_avaliable: None,
        is_empty_state: None,
        is_first_request_after_signup: None,
        is_end_of_timeline: None,
    };

    let supported_prompt_types: BTreeSet<PromptType> = [
        PromptType::INLINE_PROMPT,
        PromptType::FULL_COVER,
        PromptType::HALF_COVER,
        PromptType::RELEVANCE_PROMPT,
    ]
    .into_iter()
    .collect();

    let user_roles = if query.user_roles.is_empty() {
        None
    } else {
        Some(query.user_roles.iter().cloned().collect())
    };

    GetInjectionsRequest {
        client_context,
        display_context,
        request_targeting_context: Some(request_targeting_context),
        user_roles,
        supported_prompt_types: Some(supported_prompt_types),
    }
}
