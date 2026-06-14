use crate::clients::tweet_entity_service_client::TESClient;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableQuotedVqvDurationCheck;
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::SocialGraphClientOps;
use xai_candidate_pipeline::component_library::utils::{MokaCache, default_moka_cache};
use xai_candidate_pipeline::hydrator::{CacheStore, Hydrator};

pub struct QuoteHydrator {
    pub tes_client: Arc<dyn TESClient + Send + Sync>,
    pub socialgraph_client: Arc<dyn SocialGraphClientOps>,
    pub cache: MokaCache<u64, QuoteCacheValue>,
}

impl QuoteHydrator {
    pub async fn new(
        tes_client: Arc<dyn TESClient + Send + Sync>,
        socialgraph_client: Arc<dyn SocialGraphClientOps>,
    ) -> Self {
        let cache = default_moka_cache();
        Self {
            tes_client,
            socialgraph_client,
            cache,
        }
    }

    async fn get_quoted_video_durations(
        &self,
        quoted_tweet_ids: Vec<u64>,
    ) -> HashMap<u64, Option<i32>> {
        if quoted_tweet_ids.is_empty() {
            return HashMap::new();
        }
        let result = tokio::time::timeout(
            std::time::Duration::from_millis(200),
            self.tes_client.get_min_video_durations(quoted_tweet_ids),
        )
        .await;
        match result {
            Ok(durations) => durations
                .into_iter()
                .filter_map(|(id, result)| result.ok().map(|d| (id, d.map(|v| v as i32))))
                .collect(),
            Err(_) => HashMap::new(),
        }
    }

    async fn get_blocked_by(&self, viewer_id: u64, quoted_user_ids: Vec<u64>) -> HashSet<u64> {
        if quoted_user_ids.is_empty() {
            return HashSet::new();
        }
        self.socialgraph_client
            .check_blocked_by(viewer_id, &quoted_user_ids)
            .await
            .unwrap_or_default()
    }
}

#[derive(Clone, Debug)]
pub struct QuoteCacheValue {
    pub quoted_tweet_id: Option<u64>,
    pub quoted_user_id: Option<u64>,
}

#[async_trait]
impl Hydrator<ScoredPostsQuery, PostCandidate> for QuoteHydrator {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.has_cached_posts
    }

    async fn hydrate(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let tweet_ids: Vec<u64> = candidates.iter().map(|c| c.tweet_id).collect();

        let mut cache_misses: Vec<u64> = Vec::new();
        let mut resolved: Vec<(u64, Option<u64>, Option<u64>)> =
            Vec::with_capacity(tweet_ids.len());

        for &tweet_id in &tweet_ids {
            if let Some(cached) = self.cache.get(&tweet_id).await {
                resolved.push((tweet_id, cached.quoted_tweet_id, cached.quoted_user_id));
            } else {
                cache_misses.push(tweet_id);
                resolved.push((tweet_id, None, None));
            }
        }

        if !cache_misses.is_empty() {
            let quoted_tweets = self
                .tes_client
                .get_quoted_tweets(cache_misses.clone())
                .await;

            for entry in resolved.iter_mut() {
                let tweet_id = entry.0;
                if !cache_misses.contains(&tweet_id) {
                    continue;
                }
                let (qt_tweet_id, qt_user_id) = match quoted_tweets.get(&tweet_id) {
                    Some(Ok(Some(qt))) => (Some(qt.tweet_id), Some(qt.user_id)),
                    _ => (None, None),
                };
                entry.1 = qt_tweet_id;
                entry.2 = qt_user_id;

                self.cache
                    .insert(
                        tweet_id,
                        QuoteCacheValue {
                            quoted_tweet_id: qt_tweet_id,
                            quoted_user_id: qt_user_id,
                        },
                    )
                    .await;
            }
        }

        let quoted_user_ids: Vec<u64> = resolved
            .iter()
            .filter_map(|(_, _, uid)| *uid)
            .collect::<HashSet<u64>>()
            .into_iter()
            .collect();

        let fetch_quoted_duration = query.params.get(EnableQuotedVqvDurationCheck);
        let quoted_tweet_ids: Vec<u64> = if fetch_quoted_duration {
            resolved
                .iter()
                .filter_map(|(_, qt_id, _)| *qt_id)
                .collect::<HashSet<u64>>()
                .into_iter()
                .collect()
        } else {
            Vec::new()
        };

        let (blocked_by, quoted_durations) = tokio::join!(
            self.get_blocked_by(query.user_id, quoted_user_ids),
            self.get_quoted_video_durations(quoted_tweet_ids),
        );

        resolved
            .iter()
            .map(|(_, qt_tweet_id, qt_user_id)| {
                let quoted_author_blocks_viewer = qt_user_id
                    .map(|uid| blocked_by.contains(&uid))
                    .unwrap_or(false);
                let quoted_video_duration_ms = qt_tweet_id
                    .and_then(|id| quoted_durations.get(&id).copied())
                    .flatten();
                Ok(PostCandidate {
                    quoted_tweet_id: *qt_tweet_id,
                    quoted_user_id: *qt_user_id,
                    quoted_author_blocks_viewer: Some(quoted_author_blocks_viewer),
                    quoted_video_duration_ms,
                    ..Default::default()
                })
            })
            .collect()
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.quoted_tweet_id = hydrated.quoted_tweet_id;
        candidate.quoted_user_id = hydrated.quoted_user_id;
        candidate.quoted_author_blocks_viewer = hydrated.quoted_author_blocks_viewer;
        candidate.quoted_video_duration_ms = hydrated.quoted_video_duration_ms;
    }
}
