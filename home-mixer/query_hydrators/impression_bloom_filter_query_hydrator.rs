use crate::models::query::ImpressionBloomFilterEntry;
use crate::models::query::ScoredPostsQuery;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::ImpressionBloomFilterClient;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;
use xai_x_thrift::impression_bloom_filter::SurfaceArea;

pub struct ImpressionBloomFilterQueryHydrator {
    pub client: Arc<dyn ImpressionBloomFilterClient>,
}

#[async_trait]
impl QueryHydrator<ScoredPostsQuery> for ImpressionBloomFilterQueryHydrator {
    async fn hydrate(&self, query: &ScoredPostsQuery) -> Result<ScoredPostsQuery, String> {
        let user_id = query.user_id as i64;

        let bloom_filter_thrift = self
            .client
            .get(user_id, SurfaceArea::HOME_TIMELINE)
            .await
            .map_err(|e| e.to_string())?;

        let entries: Vec<ImpressionBloomFilterEntry> = bloom_filter_thrift
            .and_then(|s| s.entries)
            .unwrap_or_default()
            .iter()
            .map(|e| thrift_entry_to_proto(e))
            .collect();

        Ok(ScoredPostsQuery {
            bloom_filter_entries: entries,
            ..Default::default()
        })
    }

    fn update(&self, query: &mut ScoredPostsQuery, hydrated: ScoredPostsQuery) {
        query.bloom_filter_entries = hydrated.bloom_filter_entries;
    }
}

fn thrift_entry_to_proto(
    entry: &xai_x_thrift::impression_bloom_filter::ImpressionBloomFilterEntry,
) -> ImpressionBloomFilterEntry {
    ImpressionBloomFilterEntry {
        bloom_filter: entry.bloom_filter.iter().map(|&v| v as u64).collect(),
        size_cap: entry.size_cap,
        false_positive_rate: entry.false_positive_rate.into(),
    }
}
