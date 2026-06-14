use serde::Serialize;
use std::sync::OnceLock;

#[derive(Clone, Debug, Serialize)]
pub struct InNetworkReply {
    pub author_id: u64,
    pub in_reply_to_tweet_id: u64,
}

pub type InNetworkReplies = OnceLock<Vec<InNetworkReply>>;

pub fn serialize_in_network_replies<S>(
    replies: &InNetworkReplies,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match replies.get() {
        Some(v) => v.serialize(serializer),
        None => serializer.serialize_none(),
    }
}
