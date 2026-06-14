use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use tonic::async_trait;
use xai_candidate_pipeline::source::Source;

pub struct CachedPostsSource;

#[async_trait]
impl Source<ScoredPostsQuery, PostCandidate> for CachedPostsSource {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.has_cached_posts
    }

    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<PostCandidate>, String> {
        Ok(query.cached_posts.clone())
    }
}
