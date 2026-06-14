use crate::models::candidate::PostCandidate;
use crate::models::in_network_reply::InNetworkReply;
use crate::models::query::ScoredPostsQuery;
use crate::params::{ThunderAlgorithm, ThunderClusterId, ThunderMaxResults};
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::{ThunderClient, ThunderCluster};
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto as pb;
use xai_thunder_proto::GetInNetworkPostsRequest;
use xai_thunder_proto::in_network_posts_service_client::InNetworkPostsServiceClient;

pub struct ThunderSource {
    pub thunder_client: Arc<ThunderClient>,
}

#[async_trait]
impl Source<ScoredPostsQuery, PostCandidate> for ThunderSource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.has_cached_posts
    }

    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<PostCandidate>, String> {
        let configured = ThunderCluster::parse(&query.params.get(ThunderClusterId));
        let cluster = ThunderCluster::resolve(configured, query.decider.as_ref());
        let channel = self
            .thunder_client
            .get_random_channel(cluster)
            .ok_or_else(|| "ThunderSource: no available channel".to_string())?;

        let mut client = InNetworkPostsServiceClient::new(channel.clone());
        let following_list = &query.user_features.followed_user_ids;
        let excluded_ids = query.seen_ids.to_vec();

        let request = GetInNetworkPostsRequest {
            user_id: query.user_id,
            following_user_ids: following_list.iter().map(|&id| id as u64).collect(),
            max_results: query.params.get(ThunderMaxResults),
            exclude_tweet_ids: excluded_ids,
            algorithm: query.params.get(ThunderAlgorithm),
            debug: false,
            is_video_request: false,
        };

        let response = client
            .get_in_network_posts(request)
            .await
            .map_err(|e| format!("ThunderSource: {}", e))?;

        let posts = response.into_inner().posts;

        let replies: Vec<InNetworkReply> = posts
            .iter()
            .filter_map(|post| {
                post.in_reply_to_post_id.map(|reply_to_id| InNetworkReply {
                    author_id: post.author_id as u64,
                    in_reply_to_tweet_id: reply_to_id as u64,
                })
            })
            .collect();

        let _ = query.in_network_replies.set(replies);

        let candidates: Vec<PostCandidate> = posts
            .into_iter()
            .map(|post| {
                let in_reply_to_tweet_id = post
                    .in_reply_to_post_id
                    .and_then(|id| u64::try_from(id).ok());
                let conversation_id = post.conversation_id.and_then(|id| u64::try_from(id).ok());

                let mut ancestors = Vec::new();
                if let Some(reply_to) = in_reply_to_tweet_id {
                    ancestors.push(reply_to);
                    if let Some(root) = conversation_id.filter(|&root| root != reply_to) {
                        ancestors.push(root);
                    }
                }

                let served_type = if !query.in_network_only {
                    pb::ServedType::ForYouInNetwork
                } else {
                    pb::ServedType::RankedFollowing
                };

                let retweeted_tweet_id = post.source_post_id.and_then(|id| u64::try_from(id).ok());

                PostCandidate {
                    tweet_id: post.post_id as u64,
                    author_id: post.author_id as u64,
                    in_reply_to_tweet_id,
                    retweeted_tweet_id,
                    ancestors,
                    served_type: Some(served_type),
                    ..Default::default()
                }
            })
            .collect();

        Ok(candidates)
    }
}
