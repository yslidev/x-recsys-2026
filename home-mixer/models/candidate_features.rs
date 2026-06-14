use serde::{Deserialize, Serialize};
use thrift::protocol::{TInputProtocol, TOutputProtocol, TType};
use xai_strato::MValCodec;

pub use xai_core_entities::entities::{
    ExclusiveTweetControl, GizmoduckMuteSettings, GizmoduckMutedKeyword, GizmoduckUser,
    GizmoduckUserCounts, GizmoduckUserProfile, GizmoduckUserResult, MediaEntities, MediaEntity,
    MediaInfo, MuteSurface, PureCoreData, Reply, Share, VideoInfo,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct BlockedByUserIds {
    pub blocked_by_user_ids: Vec<i64>,
}

const BLOCKED_BY_USER_IDS_FIELD: i16 = -29174;
impl MValCodec for BlockedByUserIds {
    fn thrift_type() -> TType {
        TType::Struct
    }

    fn from_thrift(proto: &mut dyn TInputProtocol) -> Self {
        proto.read_struct_begin().unwrap();
        let mut blocked_by_user_ids: Vec<i64> = Vec::new();
        loop {
            let field = proto.read_field_begin().unwrap();
            if field.field_type == TType::Stop {
                break;
            }
            match field.id {
                Some(BLOCKED_BY_USER_IDS_FIELD) => {
                    blocked_by_user_ids = Vec::<i64>::from_thrift(proto);
                }
                _ => {
                    proto.skip(field.field_type).unwrap();
                }
            }
            proto.read_field_end().unwrap();
        }
        proto.read_struct_end().unwrap();
        BlockedByUserIds {
            blocked_by_user_ids,
        }
    }

    fn to_thrift(&self, _proto: &mut dyn TOutputProtocol) {
        panic!("Not implemented: to_thrift for BlockedByUserIds")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TopicFilteringExperiment {
    Unfiltered,
    CuratedV0,
    CuratedV0V1,
    PostBased90Pct,
    PostBased75Pct,
    PostBased50Pct,
}

impl TopicFilteringExperiment {
    pub fn parse(s: &str) -> Self {
        match s {
            "Unfiltered" => Self::Unfiltered,
            "CuratedV0" => Self::CuratedV0,
            "CuratedV0V1" => Self::CuratedV0V1,
            "PostBased90Pct" => Self::PostBased90Pct,
            "PostBased75Pct" => Self::PostBased75Pct,
            "PostBased50Pct" => Self::PostBased50Pct,
            _ => Self::Unfiltered,
        }
    }

    pub fn as_proto_mode(&self) -> i32 {
        match self {
            Self::Unfiltered => 0,
            Self::CuratedV0 => 1,
            Self::CuratedV0V1 => 2,
            Self::PostBased90Pct => 3,
            Self::PostBased75Pct => 4,
            Self::PostBased50Pct => 5,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct FilteredTopicsByExperiment {
    pub unfiltered_topic_ids: Option<Vec<i64>>,
    pub curated_v0_topic_ids: Option<Vec<i64>>,
    pub curated_v0_v1_topic_ids: Option<Vec<i64>>,
    pub post_based_90pct_topic_ids: Option<Vec<i64>>,
    pub post_based_75pct_topic_ids: Option<Vec<i64>>,
    pub post_based_50pct_topic_ids: Option<Vec<i64>>,
}

impl FilteredTopicsByExperiment {
    pub fn topic_ids_for_experiment(
        &self,
        experiment: TopicFilteringExperiment,
    ) -> Option<&Vec<i64>> {
        match experiment {
            TopicFilteringExperiment::Unfiltered => self.unfiltered_topic_ids.as_ref(),
            TopicFilteringExperiment::CuratedV0 => self.curated_v0_topic_ids.as_ref(),
            TopicFilteringExperiment::CuratedV0V1 => self.curated_v0_v1_topic_ids.as_ref(),
            TopicFilteringExperiment::PostBased90Pct => self.post_based_90pct_topic_ids.as_ref(),
            TopicFilteringExperiment::PostBased75Pct => self.post_based_75pct_topic_ids.as_ref(),
            TopicFilteringExperiment::PostBased50Pct => self.post_based_50pct_topic_ids.as_ref(),
        }
    }
}

impl MValCodec for FilteredTopicsByExperiment {
    fn thrift_type() -> TType {
        TType::Struct
    }

    fn from_thrift(proto: &mut dyn TInputProtocol) -> Self {
        proto.read_struct_begin().unwrap();
        let mut result = FilteredTopicsByExperiment::default();
        loop {
            let field = proto.read_field_begin().unwrap();
            if field.field_type == TType::Stop {
                break;
            }
            match field.id {
                Some(1) => result.unfiltered_topic_ids = Some(Vec::<i64>::from_thrift(proto)),
                Some(2) => result.curated_v0_topic_ids = Some(Vec::<i64>::from_thrift(proto)),
                Some(3) => result.curated_v0_v1_topic_ids = Some(Vec::<i64>::from_thrift(proto)),
                Some(4) => result.post_based_90pct_topic_ids = Some(Vec::<i64>::from_thrift(proto)),
                Some(5) => result.post_based_75pct_topic_ids = Some(Vec::<i64>::from_thrift(proto)),
                Some(6) => result.post_based_50pct_topic_ids = Some(Vec::<i64>::from_thrift(proto)),
                _ => {
                    proto.skip(field.field_type).unwrap();
                }
            }
            proto.read_field_end().unwrap();
        }
        proto.read_struct_end().unwrap();
        result
    }

    fn to_thrift(&self, _proto: &mut dyn TOutputProtocol) {
        panic!("Not implemented: to_thrift for FilteredTopicsByExperiment")
    }
}
