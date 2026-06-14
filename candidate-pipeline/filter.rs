use crate::candidate_pipeline::{PipelineCandidate, PipelineQuery};
use crate::util;
use std::any::{Any, type_name_of_val};
use tracing::{Span, field::Empty};
use xai_stats_receiver::global_stats_receiver;

const KEPT_SCOPE: [(&str, &str); 1] = [("requests", "kept")];
const REMOVED_SCOPE: [(&str, &str); 1] = [("requests", "removed")];

pub struct FilterResult<C> {
    pub kept: Vec<C>,
    pub removed: Vec<C>,
}

/// Filters run sequentially and partition candidates into kept and removed sets
pub trait Filter<Q, C>: Any + Send + Sync
where
    Q: PipelineQuery,
    C: PipelineCandidate,
{
    /// Decide if this filter should run for the given query
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    #[xai_stats_macro::receive_stats(latency=Bucket0To50)]
    #[tracing::instrument(skip_all, name = "filter", fields(
        name = self.name(),
        input_count = candidates.len(),
        kept_count = Empty,
        removed_count = Empty,
        filter_rate = Empty,
    ))]
    fn run(&self, query: &Q, candidates: Vec<C>) -> FilterResult<C> {
        let result = self.filter(query, candidates);
        let total = result.kept.len() + result.removed.len();
        let rate = if total > 0 {
            result.removed.len() as f64 / total as f64
        } else {
            0.0
        };
        let span = Span::current();
        span.record("kept_count", result.kept.len());
        span.record("removed_count", result.removed.len());
        span.record("filter_rate", format!("{:.3}", rate).as_str());
        self.stat(&result);
        result
    }

    /// Filter candidates by evaluating each against some criteria.
    /// Returns a FilterResult containing kept candidates (which continue to the next stage)
    /// and removed candidates (which are excluded from further processing).
    fn filter(&self, query: &Q, candidates: Vec<C>) -> FilterResult<C>;

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }

    fn stat(&self, result: &FilterResult<C>) {
        if let Some(receiver) = global_stats_receiver() {
            let metric_name = format!("{}.run", self.name());
            receiver.incr(metric_name.as_str(), &KEPT_SCOPE, result.kept.len() as u64);
            receiver.incr(
                metric_name.as_str(),
                &REMOVED_SCOPE,
                result.removed.len() as u64,
            );
        }
    }
}
