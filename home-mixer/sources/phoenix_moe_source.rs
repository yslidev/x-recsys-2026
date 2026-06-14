use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnablePhoenixMOESource, PhoenixMOEMaxResults, PhoenixRetrievalMOEInferenceClusterId,
};
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::phoenix_retrieval_client::{
    PhoenixRetrievalClient, PhoenixRetrievalCluster,
};
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto as pb;

pub struct PhoenixMOESource {
    pub phoenix_retrieval_client: Arc<dyn PhoenixRetrievalClient + Send + Sync>,
}

#[async_trait]
impl Source<ScoredPostsQuery, PostCandidate> for PhoenixMOESource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnablePhoenixMOESource)
            && (!query.is_topic_request() || query.is_bulk_topic_request())
            && !query.in_network_only
            && !query.has_cached_posts
    }

    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<PostCandidate>, String> {
        let user_id = query.user_id;

        let sequence = query
            .retrieval_sequence
            .as_ref()
            .ok_or_else(|| "PhoenixMOESource: missing retrieval_sequence".to_string())?;

        let cluster = PhoenixRetrievalCluster::parse(
            &query.params.get(PhoenixRetrievalMOEInferenceClusterId),
        );

        let response = self
            .phoenix_retrieval_client
            .retrieve(
                cluster,
                user_id,
                sequence.clone(),
                query.columnar_retrieval_sequence.clone(),
                query.params.get(PhoenixMOEMaxResults),
                vec![],
                None,
                None,
                None,
            )
            .await
            .map_err(|e| format!("PhoenixMOESource: {}", e))?;

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
                served_type: Some(pb::ServedType::ForYouPhoenixRetrievalMoe),
                ..Default::default()
            })
            .collect();

        Ok(candidates)
    }
}
