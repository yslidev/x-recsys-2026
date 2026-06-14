mod partition_organic_blender;
mod safe_gap_blender;
pub(crate) mod util;

pub use partition_organic_blender::PartitionOrganicAdsBlender;
pub use safe_gap_blender::SafeGapAdsBlender;

use util::{record_ad_risk_stats, record_post_verdict_stats};
use xai_home_mixer_proto::{FeedItem, ScoredPost};
use xai_recsys_proto::AdIndexInfo;

pub trait AdsBlender: Send + Sync {
    fn blend_inner(&self, scored_posts: Vec<ScoredPost>, ads: Vec<AdIndexInfo>) -> Vec<FeedItem>;

    fn blend(&self, scored_posts: Vec<ScoredPost>, ads: Vec<AdIndexInfo>) -> Vec<FeedItem> {
        record_post_verdict_stats(&scored_posts);
        record_ad_risk_stats(&ads);
        self.blend_inner(scored_posts, ads)
    }
}
