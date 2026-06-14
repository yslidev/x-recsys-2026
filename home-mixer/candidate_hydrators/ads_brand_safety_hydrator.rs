use crate::models::brand_safety::{
    BrandSafetyVerdict, botmaker_rule_category, botmaker_rule_id_from, compute_verdict,
    truncate_description, worst_verdict,
};
use crate::models::candidate::{PostCandidate, SafetyLabelInfo};
use crate::models::query::ScoredPostsQuery;
use crate::params::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::utils::{
    MokaCache, TweetAgeExpiry, build_moka_cache_tweet_age,
};
use xai_candidate_pipeline::hydrator::{CacheStore, CachedHydrator};
use xai_safety_label_store::SafetyLabelStoreClient;

const CACHE_SIZE: u64 = 1_000_000;

#[derive(Clone, Default)]
pub struct CachedBrandSafety {
    verdict: BrandSafetyVerdict,
    safety_labels: Vec<SafetyLabelInfo>,
}
pub struct AdsBrandSafetyHydrator {
    pub client: Arc<dyn SafetyLabelStoreClient>,
    pub cache: MokaCache<u64, CachedBrandSafety>,
}

impl AdsBrandSafetyHydrator {
    pub fn new(client: Arc<dyn SafetyLabelStoreClient>) -> Self {
        let cache = build_moka_cache_tweet_age(
            CACHE_SIZE,
            TweetAgeExpiry {
                age_threshold: Duration::from_secs(5 * 60),
                new_tweet_ttl: Duration::from_secs(60),
                old_tweet_ttl: Duration::from_secs(60 * 60),
            },
        );
        Self { client, cache }
    }
}

#[async_trait]
impl CachedHydrator<ScoredPostsQuery, PostCandidate> for AdsBrandSafetyHydrator {
    type CacheKey = u64;
    type CacheValue = CachedBrandSafety;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableAdsBrandSafetyHydrator)
            && !query
                .decider
                .as_ref()
                .is_some_and(|d| d.enabled("vf_brand_safety_dark_traffic"))
    }

    fn cache_store(&self) -> &dyn CacheStore<Self::CacheKey, Self::CacheValue> {
        &self.cache
    }

    fn cache_key(&self, candidate: &PostCandidate) -> Self::CacheKey {
        candidate.tweet_id
    }

    fn cache_value(&self, hydrated: &PostCandidate) -> Self::CacheValue {
        CachedBrandSafety {
            verdict: hydrated.brand_safety_verdict.unwrap_or_default(),
            safety_labels: hydrated.safety_labels.clone(),
        }
    }

    fn hydrate_from_cache(&self, value: Self::CacheValue) -> PostCandidate {
        PostCandidate {
            brand_safety_verdict: Some(value.verdict),
            safety_labels: value.safety_labels,
            ..Default::default()
        }
    }

    async fn hydrate_from_client(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let mut all_ids: HashSet<u64> = HashSet::new();
        for c in candidates {
            all_ids.insert(c.retweeted_tweet_id.unwrap_or(c.tweet_id));
            if let Some(qt_id) = c.quoted_tweet_id {
                all_ids.insert(qt_id);
            }
        }

        let all_ids_vec: Vec<u64> = all_ids.into_iter().collect();
        let all_ids_i64: Vec<i64> = all_ids_vec.iter().map(|&id| id as i64).collect();

        let mut label_map: HashMap<u64, xai_safety_label_store::types::SafetyLabelMap> =
            HashMap::new();
        let mut error_map: HashMap<u64, String> = HashMap::new();

        match self.client.batch_get_all_labels(&all_ids_i64).await {
            Ok(per_key_results) => {
                for (&id, result) in all_ids_vec.iter().zip(per_key_results) {
                    match result {
                        Ok(labels) => {
                            label_map.insert(id, labels);
                        }
                        Err(e) => {
                            error_map.insert(id, e.to_string());
                        }
                    }
                }
            }
            Err(e) => {
                let err_str = e.to_string();
                for &id in &all_ids_vec {
                    error_map.insert(id, err_str.clone());
                }
            }
        }

        candidates
            .iter()
            .map(|c| {
                let primary_id = c.retweeted_tweet_id.unwrap_or(c.tweet_id);

                if let Some(err) = error_map.get(&primary_id) {
                    return Err(format!("safety label lookup error for tweet {primary_id}: {err}"));
                }

                let empty = HashMap::new();
                let primary_labels = label_map.get(&primary_id).unwrap_or(&empty);
                let mut verdict = compute_verdict(primary_labels, primary_id);
                let mut safety_labels: Vec<SafetyLabelInfo> = primary_labels
                    .iter()
                    .map(|(k, v)| SafetyLabelInfo {
                        label_type: *k,
                        description: v.source.as_deref().map(truncate_description),
                        source: botmaker_rule_id_from(v)
                            .map(|id| botmaker_rule_category(id).to_string()),
                    })
                    .collect();

                if let Some(qt_id) = c.quoted_tweet_id {
                    if error_map.contains_key(&qt_id) {
                        verdict = worst_verdict(&verdict, &BrandSafetyVerdict::MediumRisk);
                    } else {
                        let qt_labels = label_map.get(&qt_id).unwrap_or(&empty);
                        verdict = worst_verdict(&verdict, &compute_verdict(qt_labels, qt_id));
                        safety_labels.extend(qt_labels.iter().map(|(k, v)| {
                            SafetyLabelInfo {
                                label_type: *k,
                                description: v.source.as_deref().map(truncate_description),
                                source: botmaker_rule_id_from(v)
                                    .map(|id| botmaker_rule_category(id).to_string()),
                            }
                        }));
                    }
                }

                safety_labels.sort_unstable_by_key(|l| i32::from(l.label_type));
                safety_labels.dedup_by(|a, b| a.label_type == b.label_type);

                Ok(PostCandidate {
                    brand_safety_verdict: Some(verdict),
                    safety_labels,
                    ..Default::default()
                })
            })
            .collect()
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.brand_safety_verdict = hydrated.brand_safety_verdict;
        candidate.safety_labels = hydrated.safety_labels;
    }
}
