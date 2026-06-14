use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnableNewUserTopicRetrieval, PhoenixMaxResults, PhoenixRetrievalInferenceClusterId,
    PhoenixRetrievalNewUserHistoryThreshold, PhoenixRetrievalNewUserInferenceClusterId,
};
use crate::util::phoenix_request::{build_client_context, build_user_context};
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::phoenix_retrieval_client::{
    PhoenixRetrievalClient, PhoenixRetrievalCluster,
};
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto as pb;

pub struct PhoenixSource {
    pub phoenix_retrieval_client: Arc<dyn PhoenixRetrievalClient + Send + Sync>,
}

impl PhoenixSource {
    fn resolve_cluster(query: &ScoredPostsQuery) -> PhoenixRetrievalCluster {
        let configured_cluster =
            PhoenixRetrievalCluster::parse(&query.params.get(PhoenixRetrievalInferenceClusterId));

        let threshold: u64 = query.params.get(PhoenixRetrievalNewUserHistoryThreshold);
        if threshold > 0 {
            let action_count = query
                .retrieval_sequence
                .as_ref()
                .and_then(|s| s.metadata.as_ref())
                .map(|m| m.length)
                .unwrap_or(0);

            if action_count < threshold {
                return PhoenixRetrievalCluster::parse(
                    &query.params.get(PhoenixRetrievalNewUserInferenceClusterId),
                );
            }
        }

        if let Some(decider) = &query.decider {
            match configured_cluster {
                PhoenixRetrievalCluster::Experiment1Lap7
                    if decider.enabled("enable_phoenix_retrieval_lap7_to_fou") =>
                {
                    return PhoenixRetrievalCluster::Experiment1Fou;
                }
                PhoenixRetrievalCluster::Experiment1Fou
                    if decider.enabled("enable_phoenix_retrieval_fou_to_lap7") =>
                {
                    return PhoenixRetrievalCluster::Experiment1Lap7;
                }
                _ => {}
            }
        }

        configured_cluster
    }
}

#[async_trait]
impl Source<ScoredPostsQuery, PostCandidate> for PhoenixSource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        (!query.is_topic_request() || query.is_bulk_topic_request())
            && (!query.params.get(EnableNewUserTopicRetrieval) || !query.has_new_user_topic_ids())
            && !query.in_network_only
            && !query.has_cached_posts
    }

    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<PostCandidate>, String> {
        let user_id = query.user_id;

        let sequence = query
            .retrieval_sequence
            .as_ref()
            .ok_or_else(|| "PhoenixSource: missing retrieval_sequence".to_string())?;

        let cluster = Self::resolve_cluster(query);
        let client_context = build_client_context(query);
        let user_context = build_user_context(query);

        let response = self
            .phoenix_retrieval_client
            .retrieve(
                cluster,
                user_id,
                sequence.clone(),
                query.columnar_retrieval_sequence.clone(),
                query.params.get(PhoenixMaxResults),
                vec![],
                None,
                client_context,
                user_context,
            )
            .await
            .map_err(|e| format!("PhoenixSource: {}", e))?;

        let candidates: Vec<PostCandidate> = response
            .top_k_candidates
            .into_iter()
            .flat_map(|scored_candidates| scored_candidates.candidates)
            .filter_map(|scored_candidate| scored_candidate.candidate)
            .map(|tweet_info| PostCandidate {
                tweet_id: tweet_info.tweet_id,
                author_id: tweet_info.author_id,
                in_reply_to_tweet_id: Some(tweet_info.in_reply_to_tweet_id),
                retweeted_tweet_id: (tweet_info.retweeted_tweet_id != 0)
                    .then_some(tweet_info.retweeted_tweet_id),
                served_type: Some(pb::ServedType::ForYouPhoenixRetrieval),
                ..Default::default()
            })
            .collect();

        Ok(candidates)
    }
}
