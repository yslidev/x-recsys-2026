use crate::models::candidate::PostCandidate;
use crate::models::in_network_reply::InNetworkReply;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnableFollowingRepliedUsersFacepile, FollowingRepliedUsersFacepileMaxPosts,
    FollowingRepliedUsersFacepileMinUsers,
};
use std::collections::HashMap;
use tonic::async_trait;
use xai_candidate_pipeline::hydrator::Hydrator;

const VIEWER_FOLLOWERS_THRESHOLD: i64 = 1000;

pub struct FollowingRepliedUsersHydrator;

impl FollowingRepliedUsersHydrator {
    fn build_reply_author_map(replies: &[InNetworkReply]) -> HashMap<u64, Vec<u64>> {
        let mut map: HashMap<u64, Vec<u64>> = HashMap::new();
        for reply in replies {
            map.entry(reply.in_reply_to_tweet_id)
                .or_default()
                .push(reply.author_id);
        }
        for authors in map.values_mut() {
            authors.sort_unstable();
            authors.dedup();
        }
        map
    }
}

#[async_trait]
impl Hydrator<ScoredPostsQuery, PostCandidate> for FollowingRepliedUsersHydrator {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        let has_enough_followers = query
            .user_features
            .follower_count
            .is_some_and(|c| c >= VIEWER_FOLLOWERS_THRESHOLD);

        has_enough_followers && query.params.get(EnableFollowingRepliedUsersFacepile)
    }

    async fn hydrate(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let min_users = query
            .params
            .get(FollowingRepliedUsersFacepileMinUsers)
            .max(0) as usize;
        let max_posts = query
            .params
            .get(FollowingRepliedUsersFacepileMaxPosts)
            .max(0) as usize;

        let empty = Vec::new();
        let replies = query.in_network_replies.get().unwrap_or(&empty);

        let reply_author_map = Self::build_reply_author_map(replies);

        let mut results = Vec::with_capacity(candidates.len());
        let mut selected_count: usize = 0;

        for candidate in candidates {
            let is_root_tweet = candidate.in_reply_to_tweet_id.is_none();

            let authors: Vec<u64> = reply_author_map
                .get(&candidate.tweet_id)
                .map(|ids| {
                    ids.iter()
                        .copied()
                        .filter(|&aid| aid != candidate.author_id)
                        .collect()
                })
                .unwrap_or_default();

            let eligible = is_root_tweet && authors.len() >= min_users;
            let user_ids = if eligible && selected_count < max_posts {
                selected_count += 1;
                authors
            } else {
                vec![]
            };

            results.push(Ok(PostCandidate {
                following_replied_user_ids: user_ids,
                ..Default::default()
            }));
        }

        results
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.following_replied_user_ids = hydrated.following_replied_user_ids;
    }
}
