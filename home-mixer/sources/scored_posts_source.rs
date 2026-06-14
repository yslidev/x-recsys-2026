use crate::models::query::ScoredPostsQuery;
use crate::scored_posts_server::ScoredPostsServer;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto::{FeedItem, feed_item};

pub struct ScoredPostsSource {
    pub scored_posts_server: Arc<ScoredPostsServer>,
}

#[async_trait]
impl Source<ScoredPostsQuery, FeedItem> for ScoredPostsSource {
    async fn source(&self, query: &ScoredPostsQuery) -> Result<Vec<FeedItem>, String> {
        let output = self
            .scored_posts_server
            .run_pipeline(query.clone())
            .await
            .map_err(|e| format!("ScoredPostsSource: {e}"))?;

        let feed_items = output
            .scored_posts
            .into_iter()
            .map(|post| FeedItem {
                position: 0,
                item: Some(feed_item::Item::Post(post)),
            })
            .collect();

        Ok(feed_items)
    }
}
