use crate::models::query::ScoredPostsQuery;
use crate::params::AdsBlenderType;
use crate::util::country_codes::bucket_country;
use std::sync::Arc;
use tonic::async_trait;
use tracing::info;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_core_entities::entities::SubscriptionLevel;
use xai_home_mixer_proto::{FeedItem, feed_item};
use xai_stats_receiver::global_stats_receiver;

const RESPONSE_METRIC: &str = "ForYouFeed.response";
const TYPE_TOTAL: (&str, &str) = ("type", "total");
const TYPE_POSTS: (&str, &str) = ("type", "posts");
const TYPE_ADS: (&str, &str) = ("type", "ads");
const TYPE_EMPTY_ADS: (&str, &str) = ("type", "empty_ads");
const TYPE_EMPTY_POSTS: (&str, &str) = ("type", "empty_posts");
const TYPE_SUFFICIENT_ADS: (&str, &str) = ("type", "sufficient_ads");
const TYPE_SUFFICIENT_POSTS: (&str, &str) = ("type", "sufficient_posts");
const STAGE_RESPONSE: (&str, &str) = ("stage", "response");
const STAGE_FETCHED: (&str, &str) = ("stage", "fetched");

pub struct ForYouResponseStatsSideEffect;

#[async_trait]
impl SideEffect<ScoredPostsQuery, FeedItem> for ForYouResponseStatsSideEffect {
    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, FeedItem>>,
    ) -> Result<(), String> {
        let fetched_posts =
            count_posts(&input.selected_candidates) + count_posts(&input.non_selected_candidates);
        let fetched_ads =
            count_ads(&input.selected_candidates) + count_ads(&input.non_selected_candidates);

        let query = &input.query;
        let blender_type = query.params.get(AdsBlenderType);

        stat_response(
            &input.selected_candidates,
            fetched_posts,
            fetched_ads,
            &query.country_code,
            query.subscription_level,
            &blender_type,
        );

        Ok(())
    }
}

fn count_posts(items: &[FeedItem]) -> usize {
    items
        .iter()
        .filter(|i| matches!(i.item, Some(feed_item::Item::Post(_))))
        .count()
}

fn count_ads(items: &[FeedItem]) -> usize {
    items
        .iter()
        .filter(|i| matches!(i.item, Some(feed_item::Item::Ad(_))))
        .count()
}

fn stat_response(
    items: &[FeedItem],
    fetched_posts: usize,
    fetched_ads: usize,
    country_code: &str,
    subscription_level: Option<SubscriptionLevel>,
    blender_type: &str,
) {
    let post_count = items
        .iter()
        .filter(|i| matches!(i.item, Some(feed_item::Item::Post(_))))
        .count();
    let ad_count = items
        .iter()
        .filter(|i| matches!(i.item, Some(feed_item::Item::Ad(_))))
        .count();

    let sub_status = subscription_level.map(|s| s.as_str()).unwrap_or("none");

    info!(
        "ForYouFeed response - {post_count} posts, {ad_count} ads \
         (fetched {fetched_posts} posts, {fetched_ads} ads), \
         country {country_code}, subscription {sub_status}, blender {blender_type}.",
    );

    if let Some(receiver) = global_stats_receiver() {
        let sub = ("subscription", sub_status);
        let blender = ("blender", blender_type);
        receiver.incr(RESPONSE_METRIC, &[TYPE_TOTAL, sub], 1);
        receiver.incr(
            RESPONSE_METRIC,
            &[TYPE_TOTAL, sub, ("country", bucket_country(country_code))],
            1,
        );
        receiver.incr(
            RESPONSE_METRIC,
            &[TYPE_POSTS, sub, blender],
            post_count as u64,
        );
        receiver.incr(RESPONSE_METRIC, &[TYPE_ADS, sub, blender], ad_count as u64);

        if ad_count == 0 {
            receiver.incr(
                RESPONSE_METRIC,
                &[TYPE_EMPTY_ADS, STAGE_RESPONSE, sub, blender],
                1,
            );
        }
        if post_count == 0 {
            receiver.incr(
                RESPONSE_METRIC,
                &[TYPE_EMPTY_POSTS, STAGE_RESPONSE, sub, blender],
                1,
            );
        }
        if ad_count >= 5 {
            receiver.incr(
                RESPONSE_METRIC,
                &[TYPE_SUFFICIENT_ADS, STAGE_RESPONSE, sub, blender],
                1,
            );
        }
        if post_count >= 20 {
            receiver.incr(
                RESPONSE_METRIC,
                &[TYPE_SUFFICIENT_POSTS, STAGE_RESPONSE, sub, blender],
                1,
            );
        }

        if fetched_ads == 0 {
            receiver.incr(RESPONSE_METRIC, &[TYPE_EMPTY_ADS, STAGE_FETCHED, sub], 1);
        }
        if fetched_posts == 0 {
            receiver.incr(RESPONSE_METRIC, &[TYPE_EMPTY_POSTS, STAGE_FETCHED, sub], 1);
        }
        if fetched_ads >= 5 {
            receiver.incr(
                RESPONSE_METRIC,
                &[TYPE_SUFFICIENT_ADS, STAGE_FETCHED, sub],
                1,
            );
        }
        if fetched_posts >= 20 {
            receiver.incr(
                RESPONSE_METRIC,
                &[TYPE_SUFFICIENT_POSTS, STAGE_FETCHED, sub],
                1,
            );
        }
    }
}
