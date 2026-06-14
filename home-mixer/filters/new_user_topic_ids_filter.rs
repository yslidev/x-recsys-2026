use crate::filters::topic_ids_filter::TopicIdExpansion;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableNewUserTopicFiltering;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct NewUserTopicIdsFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for NewUserTopicIdsFilter {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableNewUserTopicFiltering)
            && query.has_new_user_topic_ids()
            && !query.is_topic_request()
    }

    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let expanded =
            TopicIdExpansion::expand(&query.new_user_topic_ids.iter().copied().collect());

        let (kept, removed) = candidates.into_iter().partition(|c| {
            c.in_network == Some(true)
                || matches!(&c.filtered_topic_ids, Some(t) if !t.is_empty() && t.iter().any(|tid| expanded.contains(tid)))
        });

        FilterResult { kept, removed }
    }
}
