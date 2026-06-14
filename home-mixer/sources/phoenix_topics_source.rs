use crate::filters::topic_ids_filter::{TopicFilteringOverrideMap, TopicIdExpansion};
use crate::models::candidate::PostCandidate;
use crate::models::candidate_features::TopicFilteringExperiment;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnableNewUserTopicRetrieval, PhoenixMaxResults, PhoenixRetrievalTopicInferenceClusterId,
    TopicFilteringId, TopicFilteringOverrides,
};
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::phoenix_retrieval_client::{
    PhoenixRetrievalClient, PhoenixRetrievalCluster,
};
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto as pb;
use xai_stats_receiver::global_stats_receiver;

const INCLUDED_TOPIC_METRIC: &str = "PhoenixTopicsSource.included_topic_id";

pub struct PhoenixTopicsSource {
    pub phoenix_retrieval_client: Arc<dyn PhoenixRetrievalClient + Send + Sync>,
}

#[async_trait]
impl Source<ScoredPostsQuery, PostCandidate> for PhoenixTopicsSource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        let has_topics = (query.is_topic_request() && !query.is_bulk_topic_request())
            || (query.params.get(EnableNewUserTopicRetrieval) && query.has_new_user_topic_ids());
        has_topics && !query.in_network_only && !query.has_cached_posts
    }

    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<PostCandidate>, String> {
        let user_id = query.user_id;

        let sequence = query
            .retrieval_sequence
            .as_ref()
            .ok_or_else(|| "PhoenixTopicsSource: missing retrieval_sequence".to_string())?;

        let cluster = PhoenixRetrievalCluster::parse(
            &query.params.get(PhoenixRetrievalTopicInferenceClusterId),
        );

        let effective_topic_ids = if query.is_topic_request() {
            &query.topic_ids
        } else {
            &query.new_user_topic_ids
        };

        let default_experiment =
            TopicFilteringExperiment::parse(&query.params.get(TopicFilteringId));
        let override_map =
            TopicFilteringOverrideMap::parse(&query.params.get(TopicFilteringOverrides));
        let topic_filter_mode = override_map
            .resolve(effective_topic_ids, default_experiment)
            .as_proto_mode();

        let max_results = query.params.get(PhoenixMaxResults);

        let topic_entity_ids: Vec<u64> = effective_topic_ids
            .iter()
            .map(|&tid| TopicIdExpansion::resolve_first(tid) as u64)
            .collect();

        if let Some(receiver) = global_stats_receiver() {
            for &tid in effective_topic_ids {
                let tid_str = tid.to_string();
                receiver.incr(INCLUDED_TOPIC_METRIC, &[("topic_id", &tid_str)], 1);
            }
        }

        let response = self
            .phoenix_retrieval_client
            .retrieve(
                cluster,
                user_id,
                sequence.clone(),
                query.columnar_retrieval_sequence.clone(),
                max_results,
                topic_entity_ids,
                Some(topic_filter_mode),
                None,
                None,
            )
            .await
            .map_err(|e| format!("PhoenixTopicsSource: {}", e))?;

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
