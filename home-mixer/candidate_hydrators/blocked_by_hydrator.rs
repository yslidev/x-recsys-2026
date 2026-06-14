use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::SocialGraphClientOps;
use xai_candidate_pipeline::hydrator::Hydrator;

pub struct BlockedByHydrator {
    pub socialgraph_client: Arc<dyn SocialGraphClientOps>,
}

impl BlockedByHydrator {
    pub async fn new(socialgraph_client: Arc<dyn SocialGraphClientOps>) -> Self {
        Self { socialgraph_client }
    }
}

#[async_trait]
impl Hydrator<ScoredPostsQuery, PostCandidate> for BlockedByHydrator {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.has_cached_posts
    }

    async fn hydrate(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let author_ids: Vec<u64> = candidates.iter().map(|x| x.author_id).collect();

        let blocked_by_user_ids = match self
            .socialgraph_client
            .check_blocked_by(query.user_id, &author_ids)
            .await
        {
            Ok(ids) => ids,
            Err(e) => {
                let err_msg = e.to_string();
                return candidates.iter().map(|_| Err(err_msg.clone())).collect();
            }
        };
        candidates
            .iter()
            .map(|candidate| {
                let author_blocks_viewer = blocked_by_user_ids.contains(&candidate.author_id);
                Ok(PostCandidate {
                    author_blocks_viewer: Some(author_blocks_viewer),
                    ..Default::default()
                })
            })
            .collect()
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.author_blocks_viewer = hydrated.author_blocks_viewer;
    }
}
