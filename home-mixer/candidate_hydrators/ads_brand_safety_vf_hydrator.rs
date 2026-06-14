use crate::models::brand_safety::{
    BrandSafetyVerdict, botmaker_rule_category, botmaker_rule_id_from, compute_verdict,
    truncate_description, worst_verdict,
};
use crate::models::candidate::{PostCandidate, SafetyLabelInfo};
use crate::models::query::ScoredPostsQuery;
use crate::params::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::hydrator::Hydrator;
use xai_visibility_filtering::vf_safety_labels_client::VfClient;

pub struct AdsBrandSafetyVfHydrator {
    pub client: Arc<dyn VfClient>,
}

#[async_trait]
impl Hydrator<ScoredPostsQuery, PostCandidate> for AdsBrandSafetyVfHydrator {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableAdsBrandSafetyHydrator)
            && query
                .decider
                .as_ref()
                .is_some_and(|d| d.enabled("vf_brand_safety_dark_traffic"))
    }

    async fn hydrate(
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

        let tweet_ids: Vec<u64> = all_ids.into_iter().collect();
        let batch = match self.client.get_safety_labels(tweet_ids).await {
            Ok(batch) => batch,
            Err(e) => {
                let err = format!("VF get_safety_labels failed: {e}");
                return candidates.iter().map(|_| Err(err.clone())).collect();
            }
        };

        let failed_ids: HashSet<u64> = batch.failures.keys().copied().collect();
        let label_map = batch.labels;

        candidates
            .iter()
            .map(|c| {
                let primary_id = c.retweeted_tweet_id.unwrap_or(c.tweet_id);

                if failed_ids.contains(&primary_id) {
                    return Err(format!("VF lookup failed for tweet {primary_id}"));
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
                    if failed_ids.contains(&qt_id) {
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
