use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableMutualFollowJaccardHydration;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::StratoClient;
use xai_candidate_pipeline::hydrator::Hydrator;

const MIN_HASHES: usize = 256;

pub struct MutualFollowJaccardHydrator {
    pub strato_client: Arc<dyn StratoClient + Send + Sync>,
}

fn jaccard_from_minhash(a: &[i64], b: &[i64]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let matching = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
    matching as f64 / len as f64
}

#[async_trait]
impl Hydrator<ScoredPostsQuery, PostCandidate> for MutualFollowJaccardHydrator {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableMutualFollowJaccardHydration) && query.viewer_minhash.is_some()
    }

    async fn hydrate(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let viewer_minhash = match &query.viewer_minhash {
            Some(mh) if mh.len() >= MIN_HASHES => mh,
            _ => {
                return candidates
                    .iter()
                    .map(|_| {
                        Ok(PostCandidate {
                            mutual_follow_jaccard: None,
                            ..Default::default()
                        })
                    })
                    .collect();
            }
        };

        let unique_author_ids: Vec<i64> = candidates
            .iter()
            .map(|c| c.author_id as i64)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let results = self
            .strato_client
            .batch_get_minhash_with_count(&unique_author_ids)
            .await;

        let mut author_result: HashMap<i64, Result<Option<Vec<i64>>, String>> = HashMap::new();
        for (uid, result) in unique_author_ids.iter().zip(results) {
            match result {
                Ok(Some((minhash, _count))) if minhash.len() >= MIN_HASHES => {
                    author_result.insert(*uid, Ok(Some(minhash)));
                }
                Ok(Some((minhash, _))) => {
                    author_result.insert(
                        *uid,
                        Err(format!(
                            "Invalid minhash length {} (need >= {}) for author_id={}",
                            minhash.len(),
                            MIN_HASHES,
                            uid,
                        )),
                    );
                }
                Ok(None) => {
                    author_result.insert(*uid, Ok(None));
                }
                Err(e) => {
                    author_result.insert(*uid, Err(e.to_string()));
                }
            }
        }

        candidates
            .iter()
            .map(|c| {
                let author_id = c.author_id as i64;
                match author_result.get(&author_id) {
                    Some(Ok(Some(author_mh))) => Ok(PostCandidate {
                        mutual_follow_jaccard: Some(jaccard_from_minhash(
                            viewer_minhash,
                            author_mh,
                        )),
                        ..Default::default()
                    }),
                    Some(Ok(None)) => Ok(PostCandidate {
                        mutual_follow_jaccard: None,
                        ..Default::default()
                    }),
                    Some(Err(err)) => Err(err.clone()),
                    None => Err(format!(
                        "Missing minhash fetch result for author_id={}",
                        author_id,
                    )),
                }
            })
            .collect()
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.mutual_follow_jaccard = hydrated.mutual_follow_jaccard;
    }
}
