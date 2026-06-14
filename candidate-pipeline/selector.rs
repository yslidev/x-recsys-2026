use crate::candidate_pipeline::{PipelineCandidate, PipelineQuery};
use crate::util;
use std::any::type_name_of_val;
use tracing::{Span, field::Empty};

pub struct SelectResult<C> {
    pub selected: Vec<C>,
    pub non_selected: Vec<C>,
}

impl<C> SelectResult<C> {
    pub fn len(&self) -> usize {
        self.selected.len()
    }

    pub fn is_empty(&self) -> bool {
        self.selected.is_empty() && self.non_selected.is_empty()
    }
}

pub trait Selector<Q, C>: Send + Sync
where
    Q: PipelineQuery,
    C: PipelineCandidate,
{
    /// Decide if this selector should run for the given query
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    #[xai_stats_macro::receive_stats(latency=Bucket0To50, size=Bucket0To50)]
    #[tracing::instrument(skip_all, name = "selector", fields(
        name = self.name(),
        input_count = candidates.len(),
        selected_count = Empty,
        non_selected_count = Empty,
    ))]
    fn run(&self, query: &Q, candidates: Vec<C>) -> SelectResult<C> {
        let result = self.select(query, candidates);
        let span = Span::current();
        span.record("selected_count", result.selected.len());
        span.record("non_selected_count", result.non_selected.len());
        result
    }

    // Returns (selected, non_selected).
    fn select(&self, _query: &Q, candidates: Vec<C>) -> SelectResult<C> {
        let mut sorted = self.sort(candidates);
        if let Some(limit) = self.size() {
            let non_selected = sorted.split_off(limit.min(sorted.len()));
            SelectResult {
                selected: sorted,
                non_selected,
            }
        } else {
            SelectResult {
                selected: sorted,
                non_selected: vec![],
            }
        }
    }

    /// Extract the score from a candidate to use for sorting.
    fn score(&self, candidate: &C) -> f64;

    /// Sort candidates by their scores in descending order.
    fn sort(&self, candidates: Vec<C>) -> Vec<C> {
        let mut sorted = candidates;
        sorted.sort_by(|a, b| {
            self.score(b)
                .partial_cmp(&self.score(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    /// Optionally provide a size to select. Defaults to no truncation if not overridden.
    fn size(&self) -> Option<usize> {
        None
    }

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}
