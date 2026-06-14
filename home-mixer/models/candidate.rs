use crate::models::brand_safety::BrandSafetyVerdict;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
pub use xai_candidate_pipeline::component_library::models::PhoenixScores;
use xai_home_mixer_proto as pb;
use xai_visibility_filtering::models as vf;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PostCandidate {
    pub tweet_id: u64,
    pub author_id: u64,
    pub tweet_text: String,
    pub in_reply_to_tweet_id: Option<u64>,
    pub retweeted_tweet_id: Option<u64>,
    pub retweeted_user_id: Option<u64>,
    pub quoted_tweet_id: Option<u64>,
    pub quoted_user_id: Option<u64>,
    pub phoenix_scores: PhoenixScores,
    pub prediction_request_id: Option<u64>,
    pub last_scored_at_ms: Option<u64>,
    pub weighted_score: Option<f64>,
    pub score: Option<f64>,
    #[serde(
        serialize_with = "serialize_served_type",
        deserialize_with = "deserialize_served_type"
    )]
    pub served_type: Option<pb::ServedType>,
    pub in_network: Option<bool>,
    pub ancestors: Vec<u64>,
    pub min_video_duration_ms: Option<i32>,
    pub quoted_video_duration_ms: Option<i32>,
    pub author_followers_count: Option<i32>,
    pub author_screen_name: Option<String>,
    pub retweeted_screen_name: Option<String>,
    pub visibility_reason: Option<vf::FilteredReason>,
    pub drop_ancillary_posts: Option<bool>,
    pub subscription_author_id: Option<u64>,
    pub tweet_type_metrics: Option<Vec<u8>>,
    pub author_blocks_viewer: Option<bool>,
    pub quoted_author_blocks_viewer: Option<bool>,
    pub filtered_topic_ids: Option<Vec<i64>>,
    pub unfiltered_topic_ids: Option<Vec<i64>>,
    #[serde(default)]
    pub following_replied_user_ids: Vec<u64>,
    pub has_media: Option<bool>,
    pub language_code: Option<String>,
    pub fav_count: Option<i64>,
    pub reply_count: Option<i64>,
    pub repost_count: Option<i64>,
    pub quote_count: Option<i64>,
    pub mutual_follow_jaccard: Option<f64>,
    pub brand_safety_verdict: Option<BrandSafetyVerdict>,
    #[serde(default)]
    pub safety_labels: Vec<SafetyLabelInfo>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SafetyLabelInfo {
    #[serde(with = "xai_safety_label_store::types::serde_label_type")]
    pub label_type: xai_x_thrift::tweet_safety_label::SafetyLabelType,
    pub description: Option<String>,
    pub source: Option<String>,
}

fn serialize_served_type<S>(
    served_type: &Option<pb::ServedType>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    served_type.map(|value| value as i32).serialize(serializer)
}

fn deserialize_served_type<'de, D>(deserializer: D) -> Result<Option<pb::ServedType>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<i32>::deserialize(deserializer)?;
    match value {
        None => Ok(None),
        Some(raw_value) => pb::ServedType::try_from(raw_value)
            .map(Some)
            .map_err(|_| serde::de::Error::custom("invalid ServedType value")),
    }
}

pub trait CandidateHelpers {
    fn get_screen_names(&self) -> HashMap<u64, String>;
    fn get_original_tweet_id(&self) -> u64;
    fn get_original_author_id(&self) -> u64;
    fn as_tweet_info(&self, is_followed_by_viewer: bool) -> xai_recsys_proto::TweetInfo;
}

impl CandidateHelpers for PostCandidate {
    fn get_screen_names(&self) -> HashMap<u64, String> {
        let mut screen_names = HashMap::<u64, String>::new();
        if let Some(author_screen_name) = self.author_screen_name.clone() {
            screen_names.insert(self.author_id, author_screen_name);
        }
        if let (Some(retweeted_screen_name), Some(retweeted_user_id)) =
            (self.retweeted_screen_name.clone(), self.retweeted_user_id)
        {
            screen_names.insert(retweeted_user_id, retweeted_screen_name);
        }
        screen_names
    }

    fn get_original_tweet_id(&self) -> u64 {
        self.retweeted_tweet_id.unwrap_or(self.tweet_id)
    }

    fn get_original_author_id(&self) -> u64 {
        self.retweeted_user_id.unwrap_or(self.author_id)
    }

    fn as_tweet_info(&self, is_followed_by_viewer: bool) -> xai_recsys_proto::TweetInfo {
        xai_recsys_proto::TweetInfo {
            tweet_id: self.get_original_tweet_id(),
            author_id: self.get_original_author_id(),
            retweeting_tweet_id: if self.retweeted_tweet_id.is_some() {
                self.tweet_id
            } else {
                0
            },
            retweeting_author_id: if self.retweeted_user_id.is_some() {
                self.author_id
            } else {
                0
            },
            quoted_tweet_id: self.quoted_tweet_id.unwrap_or(0),
            quoted_author_id: self.quoted_user_id.unwrap_or(0),
            in_reply_to_tweet_id: self.in_reply_to_tweet_id.unwrap_or(0),
            is_author_followed_by_user: is_followed_by_viewer,
            min_video_duration_ms: self.min_video_duration_ms.map(|ms| ms as u64).unwrap_or(0),
            fav_count: self.fav_count.unwrap_or(0) as u64,
            retweet_count: self.repost_count.unwrap_or(0) as u64,
            quote_count: self.quote_count.unwrap_or(0) as u64,
            reply_count: self.reply_count.unwrap_or(0) as u64,
            language_code: xai_recsys_proto::language_code_string_to_enum(
                self.language_code.as_deref().unwrap_or(""),
            ) as i32,
            tweet_bool_features: Some(xai_recsys_proto::TweetBoolFeatures {
                has_media: self.has_media.unwrap_or(false),
                is_retweet: self.retweeted_tweet_id.is_some(),
                is_quote: self.quoted_tweet_id.is_some(),
                is_reply: self.in_reply_to_tweet_id.is_some(),
                ..Default::default()
            }),
            ..Default::default()
        }
    }
}
