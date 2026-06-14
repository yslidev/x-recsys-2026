use super::AdsBlender;
use super::util::*;
use xai_home_mixer_proto::{FeedItem, ScoredPost};
use xai_recsys_proto::AdIndexInfo;

pub struct SafeGapAdsBlender;

impl AdsBlender for SafeGapAdsBlender {
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

    let safe_gaps = find_safe_gaps(&scored_posts);
    let spacing = compute_spacing(&ads);
    let first_ideal = ads[0].insert_position.max(0) as usize;
    let placements = assign_ads_to_gaps(&safe_gaps, ads.len(), &spacing, first_ideal);

    interleave_and_finalize(scored_posts, ads, &placements)
}

pub(crate) fn assign_ads_to_gaps(
    safe_gaps: &[usize],
    num_ads: usize,
    spacing: &AdSpacing,
    first_ideal: usize,
) -> Vec<usize> {
    let mut placements: Vec<usize> = Vec::new();
    let mut search_from: usize = 0;
    let mut prev_ideal = first_ideal;

    for _ in 0..num_ads {
        if search_from >= safe_gaps.len() {
            break;
        }

        let (ideal, min) = match placements.last() {
            None => (first_ideal, 1),
            Some(&last_actual) => {
                let ideal = prev_ideal + spacing.requested;
                let min = (prev_ideal + spacing.min).max(last_actual + DEFAULT_SPACING.min);
                (ideal, min)
            }
        };

        let gap = find_best_gap(&safe_gaps[search_from..], ideal, min);

        match gap {
            Some((offset, g)) => {
                placements.push(g);
                search_from += offset + 1;
                prev_ideal = ideal;
            }
            None => break,
        }
    }

    placements
}

pub(crate) fn find_best_gap(gaps: &[usize], ideal: usize, min: usize) -> Option<(usize, usize)> {
    let min_offset = gaps.partition_point(|&g| g < min);
    if min_offset >= gaps.len() {
        return None;
    }
    let candidates = &gaps[min_offset..];
    let ideal_pos = candidates.partition_point(|&g| g < ideal);

    let chosen = if ideal_pos >= candidates.len() {
        candidates.len() - 1
    } else if ideal_pos == 0 {
        0
    } else {
        let below = candidates[ideal_pos - 1];
        let above = candidates[ideal_pos];
        if ideal - below <= above - ideal {
            ideal_pos - 1
        } else {
            ideal_pos
        }
    };

    Some((min_offset + chosen, candidates[chosen]))
}
