use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::util::candidates_util::get_related_post_ids;
use xai_candidate_pipeline::component_library::utils::BloomFilter;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

/// Filter out previously seen posts using a Bloom Filter and
/// the seen IDs sent in the request directly from the client
pub struct PreviouslySeenPostsFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for PreviouslySeenPostsFilter {
    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let bloom_filters = query
            .bloom_filter_entries
            .iter()
            .map(|e| BloomFilter::from_parts(e.size_cap, e.false_positive_rate, &e.bloom_filter))
            .collect::<Vec<_>>();

        let (removed, kept): (Vec<_>, Vec<_>) = candidates.into_iter().partition(|c| {
            get_related_post_ids(c).iter().any(|&post_id| {
                query.seen_ids.contains(&post_id)
                    || bloom_filters
                        .iter()
                        .any(|filter| filter.may_contain(post_id))
            })
        });

        FilterResult { kept, removed }
    }
}
