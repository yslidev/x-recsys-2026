use serde::{Deserialize, Serialize};
use thrift::protocol::{TInputProtocol, TOutputProtocol, TType};
use xai_strato::MValCodec;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct UserFeatures {
    pub muted_keywords: Vec<String>,
    pub blocked_user_ids: Vec<i64>,
    pub muted_user_ids: Vec<i64>,
    pub followed_user_ids: Vec<i64>,
    pub subscribed_user_ids: Vec<i64>,
    pub follower_count: Option<i64>,
}

const MUTED_KEYWORDS_FIELD: i16 = 23093;
const BLOCKED_USER_IDS_FIELD: i16 = -28831;
const MUTED_USER_IDS_FIELD: i16 = -7422;
const FOLLOWED_USER_IDS_FIELD: i16 = -8003;
const SUBSCRIBED_USER_IDS_FIELD: i16 = -21197;
const FOLLOWER_COUNT_FIELD: i16 = -23663;

impl MValCodec for UserFeatures {
    fn thrift_type() -> TType {
        TType::Struct
    }

    fn from_thrift(proto: &mut dyn TInputProtocol) -> Self {
        proto.read_struct_begin().unwrap();
        let mut muted_keywords: Vec<String> = Vec::new();
        let mut blocked_user_ids: Vec<i64> = Vec::new();
        let mut muted_user_ids: Vec<i64> = Vec::new();
        let mut followed_user_ids: Vec<i64> = Vec::new();
        let mut subscribed_user_ids: Vec<i64> = Vec::new();
        let mut follower_count: Option<i64> = None;
        loop {
            let field = proto.read_field_begin().unwrap();
            if field.field_type == TType::Stop {
                break;
            }
            match field.id {
                Some(MUTED_KEYWORDS_FIELD) => {
                    muted_keywords = Vec::<String>::from_thrift(proto);
                }
                Some(BLOCKED_USER_IDS_FIELD) => {
                    blocked_user_ids = Vec::<i64>::from_thrift(proto);
                }
                Some(MUTED_USER_IDS_FIELD) => {
                    muted_user_ids = Vec::<i64>::from_thrift(proto);
                }
                Some(FOLLOWED_USER_IDS_FIELD) => {
                    followed_user_ids = Vec::<i64>::from_thrift(proto);
                }
                Some(SUBSCRIBED_USER_IDS_FIELD) => {
                    subscribed_user_ids = Vec::<i64>::from_thrift(proto);
                }
                Some(FOLLOWER_COUNT_FIELD) => {
                    follower_count = Option::<i64>::from_thrift(proto);
                }
                _ => {
                    proto.skip(field.field_type).unwrap();
                }
            }
            proto.read_field_end().unwrap();
        }
        proto.read_struct_end().unwrap();
        UserFeatures {
            muted_keywords,
            blocked_user_ids,
            muted_user_ids,
            followed_user_ids,
            subscribed_user_ids,
            follower_count,
        }
    }

    fn to_thrift(&self, _proto: &mut dyn TOutputProtocol) {
        panic!("Not implemented: to_thrift for UserFeatures")
    }
}
