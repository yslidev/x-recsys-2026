use crate::clients::vm_ranker_client::{VMRankerClient, VMRankerCluster};
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::*;
use crate::util::candidates_util;
use std::collections::HashMap;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::scorer::Scorer;
use xai_vm_ranker_proto::{DppParams, PhoenixScores, RankCandidate, RankRequest};

pub struct VMRanker {
    pub client: Arc<dyn VMRankerClient>,
}

#[async_trait]
impl Scorer<ScoredPostsQuery, PostCandidate> for VMRanker {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableVMRanker)
    }

    async fn score(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let cluster = VMRankerCluster::parse(&query.params.get(VMRankerClusterId));
        let request = build_request(query, candidates);

        let response = match self.client.rank(cluster, request).await {
            Ok(resp) => resp,
            Err(e) => {
                let msg = format!("VMRanker gRPC call failed: {e}");
                return vec![Err(msg); candidates.len()];
            }
        };

        let score_map: HashMap<u64, f64> = response
            .candidates
            .iter()
            .map(|sc| (sc.tweet_id, sc.score))
            .collect();

        candidates
            .iter()
            .map(|c| {
                let score = score_map.get(&c.tweet_id).copied().or(c.score);
                Ok(PostCandidate {
                    score,
                    ..Default::default()
                })
            })
            .collect()
    }

    fn update(&self, candidate: &mut PostCandidate, scored: PostCandidate) {
        candidate.score = scored.score;
    }
}

fn build_request(query: &ScoredPostsQuery, candidates: &[PostCandidate]) -> RankRequest {
    let min_video_duration_ms = query.params.get(MinVideoDurationMs);
    let vqv_weight_value = query.params.get(VqvWeight);
    let request_timestamp_ms = query.request_time_ms as u64;

    let proto_candidates: Vec<RankCandidate> = candidates
        .iter()
        .map(|c| {
            let phoenix_scores = Some(PhoenixScores {
                favorite_score: c.phoenix_scores.favorite_score,
                reply_score: c.phoenix_scores.reply_score,
                retweet_score: c.phoenix_scores.retweet_score,
                photo_expand_score: c.phoenix_scores.photo_expand_score,
                click_score: c.phoenix_scores.click_score,
                profile_click_score: c.phoenix_scores.profile_click_score,
                vqv_score: c.phoenix_scores.vqv_score,
                share_score: c.phoenix_scores.share_score,
                share_via_dm_score: c.phoenix_scores.share_via_dm_score,
                share_via_copy_link_score: c.phoenix_scores.share_via_copy_link_score,
                dwell_score: c.phoenix_scores.dwell_score,
                quote_score: c.phoenix_scores.quote_score,
                quoted_click_score: c.phoenix_scores.quoted_click_score,
                follow_author_score: c.phoenix_scores.follow_author_score,
                not_interested_score: c.phoenix_scores.not_interested_score,
                block_author_score: c.phoenix_scores.block_author_score,
                mute_author_score: c.phoenix_scores.mute_author_score,
                report_score: c.phoenix_scores.report_score,
                not_dwelled_score: c.phoenix_scores.not_dwelled_score,
                dwell_time: c.phoenix_scores.dwell_time,
                click_dwell_time: c.phoenix_scores.click_dwell_time,
            });

            let vqv_weight =
                candidates_util::vqv_weight(query, c, min_video_duration_ms, vqv_weight_value);

            RankCandidate {
                tweet_id: c.tweet_id,
                author_id: c.author_id,
                in_network: c.in_network.unwrap_or(false),
                is_retweet: c.retweeted_tweet_id.is_some(),
                is_reply: c.in_reply_to_tweet_id.is_some(),
                author_followers_count: c.author_followers_count.unwrap_or(0),
                vqv_ineligible: vqv_weight == 0.0,
                retweeted_tweet_id: c.retweeted_tweet_id.unwrap_or(0),
                score: c.score,
                phoenix_scores,
            }
        })
        .collect();

    let dpp_theta = query.params.get(VMRankerDppTheta);
    let dpp_max_selected_rank = query.params.get(VMRankerDppMaxSelectedRank);

    let dpp_params = if dpp_theta > 0.0 || dpp_max_selected_rank > 0 {
        Some(DppParams {
            theta: dpp_theta,
            max_selected_rank: dpp_max_selected_rank,
        })
    } else {
        None
    };

    RankRequest {
        viewer_id: query.user_id,
        request_timestamp_ms,
        candidates: proto_candidates,
        value_model_id: query.params.get(VMRankerValueModelId),
        viewer_following_count: query.user_features.followed_user_ids.len() as u32,
        dpp_params,
        new_user_age_threshold_secs: Some(query.params.get(NewUserAgeThresholdSecs)),
    }
}
