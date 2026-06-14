use crate::clients::gizmoduck_client::GizmoduckClient;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::utils::{MokaCache, default_moka_cache};
use xai_candidate_pipeline::hydrator::{CacheStore, CachedHydrator};

pub struct GizmoduckCandidateHydrator {
    pub gizmoduck_client: Arc<dyn GizmoduckClient + Send + Sync>,
    pub cache: MokaCache<GizmoduckCacheKey, GizmoduckCacheValue>,
}

impl GizmoduckCandidateHydrator {
    pub async fn new(gizmoduck_client: Arc<dyn GizmoduckClient + Send + Sync>) -> Self {
        let cache = default_moka_cache();
        Self {
            gizmoduck_client,
            cache,
        }
    }
}

#[async_trait]
impl CachedHydrator<ScoredPostsQuery, PostCandidate> for GizmoduckCandidateHydrator {
    type CacheKey = GizmoduckCacheKey;
    type CacheValue = GizmoduckCacheValue;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.has_cached_posts
    }

    fn cache_store(&self) -> &dyn CacheStore<Self::CacheKey, Self::CacheValue> {
        &self.cache
    }
    fn cache_key(&self, candidate: &PostCandidate) -> Self::CacheKey {
        GizmoduckCacheKey {
            author_id: candidate.author_id,
            retweeted_user_id: candidate.retweeted_user_id,
        }
    }

    fn cache_value(&self, hydrated: &PostCandidate) -> Self::CacheValue {
        GizmoduckCacheValue {
            author_followers_count: hydrated.author_followers_count,
            author_screen_name: hydrated.author_screen_name.clone(),
            retweeted_screen_name: hydrated.retweeted_screen_name.clone(),
        }
    }

    fn hydrate_from_cache(&self, value: Self::CacheValue) -> PostCandidate {
        PostCandidate {
            author_followers_count: value.author_followers_count,
            author_screen_name: value.author_screen_name,
            retweeted_screen_name: value.retweeted_screen_name,
            ..Default::default()
        }
    }

    async fn hydrate_from_client(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let client = &self.gizmoduck_client;

        let author_ids: Vec<_> = candidates.iter().map(|c| c.author_id).collect();
        let author_ids: Vec<_> = author_ids.iter().map(|&x| x as i64).collect();
        let retweet_user_ids: Vec<_> = candidates.iter().map(|c| c.retweeted_user_id).collect();
        let retweet_user_ids: Vec<_> = retweet_user_ids.iter().flatten().collect();
        let retweet_user_ids: Vec<_> = retweet_user_ids.iter().map(|&&x| x as i64).collect();

        let mut user_ids_to_fetch = Vec::with_capacity(author_ids.len() + retweet_user_ids.len());
        user_ids_to_fetch.extend(author_ids);
        user_ids_to_fetch.extend(retweet_user_ids);
        user_ids_to_fetch.dedup();

        let users = client.get_users(user_ids_to_fetch).await;

        let mut hydrated_candidates = Vec::with_capacity(candidates.len());

        for candidate in candidates {
            let user = users.get(&(candidate.author_id as i64));
            let user = match user {
                Some(Ok(Some(user))) => Ok(Some(user)),
                Some(Ok(None)) | None => Ok(None),
                Some(Err(err)) => Err(err.to_string()),
            };

            let retweet_user = candidate
                .retweeted_user_id
                .and_then(|retweeted_user_id| users.get(&(retweeted_user_id as i64)));
            let retweet_user = match retweet_user {
                Some(Ok(Some(user))) => Ok(Some(user)),
                Some(Ok(None)) | None => Ok(None),
                Some(Err(err)) => Err(err.to_string()),
            };

            let hydrated = match (user, retweet_user) {
                (Ok(user), Ok(retweet_user)) => {
                    let user_counts = user.and_then(|user| user.user.as_ref().map(|u| &u.counts));
                    let user_profile = user.and_then(|user| user.user.as_ref().map(|u| &u.profile));

                    let author_followers_count: Option<i32> =
                        user_counts.map(|x| x.followers_count).map(|x| x as i32);
                    let author_screen_name: Option<String> =
                        user_profile.map(|x| x.screen_name.clone());

                    let retweet_profile =
                        retweet_user.and_then(|user| user.user.as_ref().map(|u| &u.profile));
                    let retweeted_screen_name: Option<String> =
                        retweet_profile.map(|x| x.screen_name.clone());

                    Ok(PostCandidate {
                        author_followers_count,
                        author_screen_name,
                        retweeted_screen_name,
                        ..Default::default()
                    })
                }
                (Err(err), _) | (_, Err(err)) => Err(err),
            };
            hydrated_candidates.push(hydrated);
        }

        hydrated_candidates
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.author_followers_count = hydrated.author_followers_count;
        candidate.author_screen_name = hydrated.author_screen_name;
        candidate.retweeted_screen_name = hydrated.retweeted_screen_name;
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct GizmoduckCacheKey {
    pub author_id: u64,
    pub retweeted_user_id: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct GizmoduckCacheValue {
    pub author_followers_count: Option<i32>,
    pub author_screen_name: Option<String>,
    pub retweeted_screen_name: Option<String>,
}
