use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::util::tweet_type_metrics::*;
use std::collections::HashSet;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::utils::duration_since_creation_opt;
use xai_candidate_pipeline::hydrator::Hydrator;

const THIRTY_MINUTES_MS: u64 = 30 * 60 * 1000;
const ONE_HOUR_MS: u64 = 60 * 60 * 1000;
const SIX_HOURS_MS: u64 = 6 * 60 * 60 * 1000;
const TWELVE_HOURS_MS: u64 = 12 * 60 * 60 * 1000;
const TWENTY_FOUR_HOURS_MS: u64 = 24 * 60 * 60 * 1000;

pub struct TweetTypeMetricsHydrator;

impl TweetTypeMetricsHydrator {
    pub fn new() -> Self {
        Self
    }

    pub fn create_tweet_type_bitset(
        candidate: &PostCandidate,
        query: &ScoredPostsQuery,
    ) -> HashSet<usize> {
        let mut true_tweet_types = HashSet::new();

        true_tweet_types.insert(ANY_CANDIDATE);

        if candidate.retweeted_tweet_id.is_some() {
            true_tweet_types.insert(RETWEET);
        }

        if candidate.in_reply_to_tweet_id.is_some() {
            true_tweet_types.insert(REPLY);
        }

        if candidate.subscription_author_id.is_some() {
            true_tweet_types.insert(SUBSCRIPTION_POST);
        }

        if let Some(score) = candidate.score
            && score != 0.0
        {
            true_tweet_types.insert(FULL_SCORING_SUCCEEDED);
        }

        if !candidate.ancestors.is_empty() {
            true_tweet_types.insert(HAS_ANCESTORS);
        }

        if candidate.in_network.unwrap_or(true) {
            true_tweet_types.insert(IN_NETWORK);
        }

        if let Some(followers) = candidate.author_followers_count {
            let followers_u32 = followers as u32;
            if followers_u32 < 100 {
                true_tweet_types.insert(AUTHOR_FOLLOWERS_0_100);
            }
            if (100..1000).contains(&followers_u32) {
                true_tweet_types.insert(AUTHOR_FOLLOWERS_100_1K);
            }
            if (1000..10000).contains(&followers_u32) {
                true_tweet_types.insert(AUTHOR_FOLLOWERS_1K_10K);
            }
            if (10000..100000).contains(&followers_u32) {
                true_tweet_types.insert(AUTHOR_FOLLOWERS_10K_100K);
            }
            if (100000..1000000).contains(&followers_u32) {
                true_tweet_types.insert(AUTHOR_FOLLOWERS_100K_1M);
            }
            if followers_u32 >= 1000000 {
                true_tweet_types.insert(AUTHOR_FOLLOWERS_1M_PLUS);
            }
        }

        if candidate.min_video_duration_ms.is_some() {
            true_tweet_types.insert(VIDEO);
        }

        if let Some(duration_ms) = candidate.min_video_duration_ms {
            let duration_ms_u32 = duration_ms as u32;
            if duration_ms_u32 <= 10000 {
                true_tweet_types.insert(VIDEO_LTE_10_SEC);
            }
            if duration_ms_u32 > 10000 && duration_ms_u32 <= 60000 {
                true_tweet_types.insert(VIDEO_BT_10_60_SEC);
            }
            if duration_ms_u32 > 60000 {
                true_tweet_types.insert(VIDEO_GT_60_SEC);
            }
        }

        if let Some(age) = duration_since_creation_opt(candidate.tweet_id) {
            let age_ms = age.as_millis() as u64;

            if age_ms <= THIRTY_MINUTES_MS {
                true_tweet_types.insert(TWEET_AGE_LTE_30_MINUTES);
            }
            if age_ms <= ONE_HOUR_MS {
                true_tweet_types.insert(TWEET_AGE_LTE_1_HOUR);
            }
            if age_ms <= SIX_HOURS_MS {
                true_tweet_types.insert(TWEET_AGE_LTE_6_HOURS);
            }
            if age_ms <= TWELVE_HOURS_MS {
                true_tweet_types.insert(TWEET_AGE_LTE_12_HOURS);
            }
            if age_ms >= TWENTY_FOUR_HOURS_MS {
                true_tweet_types.insert(TWEET_AGE_GTE_24_HOURS);
            }
        }

        let served_size = query.served_ids.len();
        if served_size == 0 {
            true_tweet_types.insert(EMPTY_REQUEST);
        }
        if served_size < 3 {
            true_tweet_types.insert(NEAR_EMPTY);
        }
        if served_size < 20 {
            true_tweet_types.insert(SERVED_SIZE_LESS_THAN_20);
        }
        if served_size < 10 {
            true_tweet_types.insert(SERVED_SIZE_LESS_THAN_10);
        }
        if served_size < 5 {
            true_tweet_types.insert(SERVED_SIZE_LESS_THAN_5);
        }

        true_tweet_types
    }

    pub fn bitset_to_bytes(bits: &HashSet<usize>) -> Vec<u8> {
        if bits.is_empty() {
            return Vec::new();
        }

        let max_bit = bits.iter().max().copied().unwrap_or(0);
        let num_bytes = (max_bit / 8) + 1;
        let mut bytes = vec![0u8; num_bytes];

        for &bit_index in bits {
            let byte_index = bit_index / 8;
            let bit_offset = bit_index % 8;
            bytes[byte_index] |= 1u8 << bit_offset;
        }

        bytes
    }
}

#[async_trait]
impl Hydrator<ScoredPostsQuery, PostCandidate> for TweetTypeMetricsHydrator {
    async fn hydrate(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let mut hydrated_candidates = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let true_tweet_types = Self::create_tweet_type_bitset(candidate, query);

            let tweet_type_metrics = Some(Self::bitset_to_bytes(&true_tweet_types));

            let hydrated = PostCandidate {
                tweet_type_metrics,
                ..Default::default()
            };
            hydrated_candidates.push(Ok(hydrated));
        }

        hydrated_candidates
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.tweet_type_metrics = hydrated.tweet_type_metrics;
    }
}
