use super::AdsBlender;
use super::util::*;
use crate::params::RESULT_SIZE;
use xai_home_mixer_proto::{FeedItem, ScoredPost, feed_item};
use xai_recsys_proto::AdIndexInfo;
use xai_stats_receiver::global_stats_receiver;

const ENFORCEMENT_METRIC: &str = "PartitionOrganic.enforcement";

pub struct PartitionOrganicAdsBlender;

impl AdsBlender for PartitionOrganicAdsBlender {
    fn blend_inner(&self, scored_posts: Vec<ScoredPost>, ads: Vec<AdIndexInfo>) -> Vec<FeedItem> {
        blend_impl(scored_posts, ads, MIN_POSTS_FOR_ADS)
    }
}

pub(crate) fn blend_impl(
    scored_posts: Vec<ScoredPost>,
    ads: Vec<AdIndexInfo>,
    min_posts: usize,
) -> Vec<FeedItem> {
    let n = scored_posts.len();

    if ads.is_empty() || n < min_posts {
        return posts_to_feed_items(scored_posts);
    }

    let spacing = compute_spacing(&ads);

    let safe_count = scored_posts.iter().filter(|p| !has_avoid(p)).count();
    let max_from_safe = safe_count / 2;
    let expected_from_spacing = if spacing.requested > 0 {
        n.saturating_sub(1) / spacing.requested
    } else {
        0
    };
    let actual_ads = ads.len().min(expected_from_spacing).min(max_from_safe);

    if actual_ads == 0 {
        return posts_to_feed_items(scored_posts);
    }

    let mut safe: Vec<ScoredPost> = Vec::new();
    let mut unsafe_posts: Vec<ScoredPost> = Vec::new();
    for post in scored_posts {
        if has_avoid(&post) {
            unsafe_posts.push(post);
        } else {
            safe.push(post);
        }
    }

    let num_safe = safe.len();
    let group_size = num_safe / actual_ads;

    let mut safe_opts: Vec<Option<ScoredPost>> = safe.into_iter().map(Some).collect();
    let mut triples: Vec<(AdIndexInfo, ScoredPost, ScoredPost)> = Vec::new();

    let mut bsr_drop: u64 = 0;
    let mut bsr_ok: u64 = 0;
    let mut handle_drop: u64 = 0;
    let mut keyword_drop: u64 = 0;

    let mut group_idx = 0;

    for ad in ads {
        if group_idx >= actual_ads {
            break;
        }
        let group_start = group_idx * group_size;
        let above_ref = safe_opts[group_start].as_ref();
        let below_ref = safe_opts[group_start + 1].as_ref();

        if should_drop_bsr_low(&ad, above_ref, below_ref) {
            bsr_drop += 1;
            continue;
        }
        if is_bsr_low_ad(&ad) {
            bsr_ok += 1;
        }

        if should_drop_handle(&ad, above_ref, below_ref) {
            handle_drop += 1;
            continue;
        }

        if should_drop_keyword(&ad, above_ref, below_ref) {
            keyword_drop += 1;
            continue;
        }

        let above = safe_opts[group_start].take().unwrap();
        let below = safe_opts[group_start + 1].take().unwrap();
        triples.push((ad, above, below));
        group_idx += 1;
    }

    let placed_ads = triples.len();
    emit_enforcement_metrics(bsr_drop, bsr_ok, handle_drop, keyword_drop);

    if placed_ads == 0 {
        let mut all_posts: Vec<ScoredPost> = safe_opts.into_iter().flatten().collect();
        all_posts.extend(unsafe_posts);
        all_posts.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        return posts_to_feed_items(all_posts);
    }

    let mut filler: Vec<ScoredPost> =
        Vec::with_capacity(num_safe - 2 * placed_ads + unsafe_posts.len());
    filler.extend(safe_opts.into_iter().flatten());
    filler.extend(unsafe_posts);
    filler.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let inter_ad_gaps = placed_ads;
    let filler_per_gap = filler.len() / inter_ad_gaps;
    let remainder = filler.len() % inter_ad_gaps;
    let mut filler_iter = filler.into_iter();

    let mut items: Vec<FeedItem> = Vec::with_capacity(n + placed_ads);

    for (i, (ad, above, below)) in triples.into_iter().enumerate() {
        items.push(FeedItem {
            position: 0,
            item: Some(feed_item::Item::Post(above)),
        });
        items.push(FeedItem {
            position: 0,
            item: Some(feed_item::Item::Ad(ad)),
        });
        items.push(FeedItem {
            position: 0,
            item: Some(feed_item::Item::Post(below)),
        });

        let count = filler_per_gap + if i >= inter_ad_gaps - remainder { 1 } else { 0 };
        for _ in 0..count {
            if let Some(post) = filler_iter.next() {
                items.push(FeedItem {
                    position: 0,
                    item: Some(feed_item::Item::Post(post)),
                });
            }
        }
    }

    items.truncate(RESULT_SIZE);
    if matches!(items.last(), Some(item) if matches!(item.item, Some(feed_item::Item::Ad(_)))) {
        items.pop();
    }
    for (i, item) in items.iter_mut().enumerate() {
        item.position = i as i32;
    }

    items
}

fn emit_enforcement_metrics(bsr_drop: u64, bsr_ok: u64, handle_drop: u64, keyword_drop: u64) {
    let Some(receiver) = global_stats_receiver() else {
        return;
    };
    if bsr_drop > 0 {
        receiver.incr(ENFORCEMENT_METRIC, &[("action", "drop")], bsr_drop);
    }
    if bsr_ok > 0 {
        receiver.incr(ENFORCEMENT_METRIC, &[("action", "ok")], bsr_ok);
    }
    if handle_drop > 0 {
        receiver.incr(
            ENFORCEMENT_METRIC,
            &[("action", "handle_drop")],
            handle_drop,
        );
    }
    if keyword_drop > 0 {
        receiver.incr(
            ENFORCEMENT_METRIC,
            &[("action", "keyword_drop")],
            keyword_drop,
        );
    }
}
