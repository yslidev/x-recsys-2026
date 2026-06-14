use crate::filter::Filter;
use crate::hydrator::Hydrator;
use crate::query_hydrator::QueryHydrator;
use crate::scorer::Scorer;
use crate::selector::SelectResult;
use crate::selector::Selector;
use crate::side_effect::{SideEffect, SideEffectInput};
use crate::source::Source;
use crate::util;
use futures::future::join_all;
use std::any::type_name_of_val;
use std::sync::Arc;
use std::time::Instant;
use tonic::async_trait;
use tracing::{Span, field::Empty, info};
use xai_stats_receiver::{HistogramBuckets, global_stats_receiver};

const FINAL_RESULT_SIZE_SCOPE: [(&str, &str); 1] = [("requests", "result_size")];
const FINAL_RESULT_EMPTY_SCOPE: [(&str, &str); 1] = [("requests", "result_empty")];

#[derive(Copy, Clone, Debug)]
pub enum PipelineStage {
    QueryHydrator,
    DependentQueryHydrator,
    Source,
    Hydrator,
    PostSelectionHydrator,
    Filter,
    PostSelectionFilter,
    Scorer,
    Selector,
    SideEffect,
}

pub struct PipelineComponents {
    pub stage: PipelineStage,
    pub components: Vec<String>,
}

pub struct PipelineResult<Q, C> {
    pub retrieved_candidates: Vec<C>,
    pub filtered_candidates: Vec<C>,
    pub selected_candidates: Vec<C>,
    pub query: Arc<Q>,
}

impl<Q: Default, C> PipelineResult<Q, C> {
    /// Create an empty result with a default query. Useful for short-circuiting
    /// requests (e.g. test users) without running the pipeline.
    pub fn empty() -> Self {
        Self {
            retrieved_candidates: vec![],
            filtered_candidates: vec![],
            selected_candidates: vec![],
            query: Arc::new(Q::default()),
        }
    }
}
pub trait PipelineQuery: Clone + Send + Sync + 'static {
    fn params(&self) -> &xai_feature_switches::Params;
    fn decider(&self) -> Option<&xai_decider::Decider>;
}

pub trait PipelineCandidate: Clone + Send + Sync + 'static {}
impl<T> PipelineCandidate for T where T: Clone + Send + Sync + 'static {}

#[async_trait]
pub trait CandidatePipeline<Q, C>: Send + Sync
where
    Q: PipelineQuery,
    C: PipelineCandidate,
{
    fn query_hydrators(&self) -> &[Box<dyn QueryHydrator<Q>>];
    fn dependent_query_hydrators(&self) -> &[Box<dyn QueryHydrator<Q>>] {
        &[]
    }
    fn sources(&self) -> &[Box<dyn Source<Q, C>>];
    fn hydrators(&self) -> &[Box<dyn Hydrator<Q, C>>];
    fn filters(&self) -> &[Box<dyn Filter<Q, C>>];
    fn scorers(&self) -> &[Box<dyn Scorer<Q, C>>];
    fn selector(&self) -> &dyn Selector<Q, C>;
    fn post_selection_hydrators(&self) -> &[Box<dyn Hydrator<Q, C>>];
    fn post_selection_filters(&self) -> &[Box<dyn Filter<Q, C>>];
    fn side_effects(&self) -> Arc<Vec<Box<dyn SideEffect<Q, C>>>>;
    fn result_size(&self) -> usize;
    fn finalize(&self, _query: &Q, _candidates: &mut Vec<C>) {}

    #[xai_stats_macro::receive_stats(latency=Bucket500To2500)]
    async fn execute(&self, query: Q) -> PipelineResult<Q, C> {
        let hydrated_query = self.hydrate_query(query).await;
        let hydrated_query = self.hydrate_dependent_query(hydrated_query).await;

        let candidates = self.fetch_candidates(&hydrated_query).await;

        let hydrated_candidates = self.hydrate(&hydrated_query, candidates).await;

        let (kept_candidates, mut filtered_candidates) =
            self.filter(&hydrated_query, hydrated_candidates.clone());

        let scored_candidates = self.score(&hydrated_query, kept_candidates).await;

        let SelectResult {
            selected: selected_candidates,
            non_selected: mut non_selected_candidates,
        } = self.select(&hydrated_query, scored_candidates);

        let post_selection_hydrated_candidates = self
            .hydrate_post_selection(&hydrated_query, selected_candidates)
            .await;

        let (mut final_candidates, post_selection_filtered_candidates) =
            self.filter_post_selection(&hydrated_query, post_selection_hydrated_candidates);
        filtered_candidates.extend(post_selection_filtered_candidates);

        let truncated_candidates =
            final_candidates.split_off(self.result_size().min(final_candidates.len()));
        non_selected_candidates.extend(truncated_candidates);

        self.finalize(&hydrated_query, &mut final_candidates);

        self.stat_result_size(&final_candidates);

        let arc_hydrated_query = Arc::new(hydrated_query);
        let input = Arc::new(SideEffectInput {
            query: arc_hydrated_query.clone(),
            selected_candidates: final_candidates.clone(),
            non_selected_candidates, // candidates are moved so we don't need to clone them
        });
        self.run_side_effects(input);

        PipelineResult {
            retrieved_candidates: hydrated_candidates,
            filtered_candidates,
            selected_candidates: final_candidates,
            query: arc_hydrated_query,
        }
    }

    /// Return all configured components grouped by stage.
    fn components(&self) -> Vec<PipelineComponents> {
        fn stage<T: ?Sized>(
            stage: PipelineStage,
            items: &[Box<T>],
            name: impl Fn(&T) -> &str,
        ) -> PipelineComponents {
            PipelineComponents {
                stage,
                components: items
                    .iter()
                    .map(|item| name(item.as_ref()).to_string())
                    .collect(),
            }
        }

        vec![
            stage(PipelineStage::QueryHydrator, self.query_hydrators(), |h| {
                h.name()
            }),
            stage(
                PipelineStage::DependentQueryHydrator,
                self.dependent_query_hydrators(),
                |h| h.name(),
            ),
            stage(PipelineStage::Source, self.sources(), |s| s.name()),
            stage(PipelineStage::Hydrator, self.hydrators(), |h| h.name()),
            stage(PipelineStage::Filter, self.filters(), |f| f.name()),
            stage(PipelineStage::Scorer, self.scorers(), |s| s.name()),
            PipelineComponents {
                stage: PipelineStage::Selector,
                components: vec![self.selector().name().to_string()],
            },
            stage(
                PipelineStage::PostSelectionHydrator,
                self.post_selection_hydrators(),
                |h| h.name(),
            ),
            stage(
                PipelineStage::PostSelectionFilter,
                self.post_selection_filters(),
                |f| f.name(),
            ),
            stage(
                PipelineStage::SideEffect,
                self.side_effects().as_ref(),
                |s| s.name(),
            ),
        ]
    }

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }

    // -------------------------- Pipeline Execution --------------------------

    /// Run all query hydrators in parallel and merge results into the query.
    #[tracing::instrument(skip_all, name = "query_hydrators", fields(
        total_count = Empty,
        enabled_count = Empty,
        disabled = Empty,
    ))]
    async fn hydrate_query(&self, query: Q) -> Q {
        let start = Instant::now();
        let all = self.query_hydrators();
        Self::record_enabled_components(all.iter(), |h| h.enable(&query), |h| h.name());
        let hydrators: Vec<_> = all.iter().filter(|h| h.enable(&query)).collect();
        let hydrate_futures = hydrators.iter().map(|h| h.run(&query));
        let results = join_all(hydrate_futures).await;

        let mut hydrated_query = query;
        for (hydrator, result) in hydrators.iter().zip(results) {
            if let Ok(hydrated) = result {
                hydrator.update(&mut hydrated_query, hydrated);
            }
        }
        self.log_stage(start);
        hydrated_query
    }

    /// Run dependent query hydrators in parallel and merge results into the query.
    ///
    /// This stage runs **after** [`hydrate_query`], so the incoming query
    /// already has all initial features populated.
    #[tracing::instrument(skip_all, name = "dependent_query_hydrators", fields(
        total_count = Empty,
        enabled_count = Empty,
        disabled = Empty,
    ))]
    async fn hydrate_dependent_query(&self, query: Q) -> Q {
        let all = self.dependent_query_hydrators();
        if all.is_empty() {
            return query;
        }
        let start = Instant::now();
        Self::record_enabled_components(all.iter(), |h| h.enable(&query), |h| h.name());
        let hydrators: Vec<_> = all.iter().filter(|h| h.enable(&query)).collect();
        let hydrate_futures = hydrators.iter().map(|h| h.run(&query));
        let results = join_all(hydrate_futures).await;

        let mut hydrated_query = query;
        for (hydrator, result) in hydrators.iter().zip(results) {
            if let Ok(hydrated) = result {
                hydrator.update(&mut hydrated_query, hydrated);
            }
        }
        self.log_stage(start);
        hydrated_query
    }

    /// Run all candidate sources in parallel and collect results.
    #[tracing::instrument(skip_all, name = "sources", fields(
        total_count = Empty,
        enabled_count = Empty,
        disabled = Empty,
        candidate_count = Empty,
    ))]
    async fn fetch_candidates(&self, query: &Q) -> Vec<C> {
        let start = Instant::now();
        let all = self.sources();
        Self::record_enabled_components(all.iter(), |s| s.enable(query), |s| s.name());
        let sources: Vec<_> = all.iter().filter(|s| s.enable(query)).collect();
        let source_futures = sources.iter().map(|s| s.run(query));
        let results = join_all(source_futures).await;

        let mut collected = Vec::new();
        for mut candidates in results.into_iter().flatten() {
            collected.append(&mut candidates);
        }
        Span::current().record("candidate_count", collected.len());
        self.log_stage_size(start, collected.len());
        collected
    }

    /// Run all candidate hydrators in parallel and merge results into candidates.
    #[tracing::instrument(skip_all, name = "hydrators", fields(
        total_count = Empty,
        enabled_count = Empty,
        disabled = Empty,
    ))]
    async fn hydrate(&self, query: &Q, candidates: Vec<C>) -> Vec<C> {
        self.run_hydrators(query, candidates, self.hydrators(), PipelineStage::Hydrator)
            .await
    }

    /// Run post-selection candidate hydrators in parallel and merge results into candidates.
    #[tracing::instrument(skip_all, name = "post_selection_hydrators", fields(
        total_count = Empty,
        enabled_count = Empty,
        disabled = Empty,
    ))]
    async fn hydrate_post_selection(&self, query: &Q, candidates: Vec<C>) -> Vec<C> {
        self.run_hydrators(
            query,
            candidates,
            self.post_selection_hydrators(),
            PipelineStage::PostSelectionHydrator,
        )
        .await
    }

    /// Shared helper to hydrate with a provided hydrator list.
    async fn run_hydrators(
        &self,
        query: &Q,
        mut candidates: Vec<C>,
        hydrators: &[Box<dyn Hydrator<Q, C>>],
        _stage: PipelineStage,
    ) -> Vec<C> {
        let start = Instant::now();
        Self::record_enabled_components(hydrators.iter(), |h| h.enable(query), |h| h.name());
        let hydrators: Vec<_> = hydrators.iter().filter(|h| h.enable(query)).collect();
        let hydrate_futures = hydrators.iter().map(|h| h.run(query, &candidates));
        let results = join_all(hydrate_futures).await;
        for (hydrator, result) in hydrators.iter().zip(results) {
            hydrator.update_all(&mut candidates, result);
        }
        self.log_stage_size(start, candidates.len());
        candidates
    }

    /// Run all filters sequentially. Each filter partitions candidates into kept and removed.
    #[tracing::instrument(skip_all, name = "filters", fields(
        total_count = Empty,
        enabled_count = Empty,
        disabled = Empty,
        input_count = candidates.len(),
        kept_count = Empty,
        removed_count = Empty,
        filter_rate = Empty,
    ))]
    fn filter(&self, query: &Q, candidates: Vec<C>) -> (Vec<C>, Vec<C>) {
        self.run_filters(query, candidates, self.filters(), PipelineStage::Filter)
    }

    /// Run post-scoring filters sequentially on already-scored candidates.
    #[tracing::instrument(skip_all, name = "post_selection_filters", fields(
        total_count = Empty,
        enabled_count = Empty,
        disabled = Empty,
        input_count = candidates.len(),
        kept_count = Empty,
        removed_count = Empty,
        filter_rate = Empty,
    ))]
    fn filter_post_selection(&self, query: &Q, candidates: Vec<C>) -> (Vec<C>, Vec<C>) {
        self.run_filters(
            query,
            candidates,
            self.post_selection_filters(),
            PipelineStage::PostSelectionFilter,
        )
    }

    // Shared helper to run filters sequentially from a provided filter list.
    fn run_filters(
        &self,
        query: &Q,
        mut candidates: Vec<C>,
        filters: &[Box<dyn Filter<Q, C>>],
        _stage: PipelineStage,
    ) -> (Vec<C>, Vec<C>) {
        Self::record_enabled_components(filters.iter(), |f| f.enable(query), |f| f.name());
        let mut all_removed = Vec::new();
        let mut removed_per_filter: Vec<(String, usize)> = Vec::new();
        for filter in filters.iter().filter(|f| f.enable(query)) {
            let result = filter.run(query, candidates);
            if !result.removed.is_empty() {
                removed_per_filter.push((filter.name().to_string(), result.removed.len()));
            }
            candidates = result.kept;
            all_removed.extend(result.removed);
        }
        let kept = candidates.len();
        let removed = all_removed.len();
        let total = kept + removed;
        let rate = if total > 0 {
            removed as f64 / total as f64
        } else {
            0.0
        };
        Span::current().record("kept_count", kept);
        Span::current().record("removed_count", removed);
        Span::current().record("filter_rate", format!("{:.3}", rate).as_str());
        self.log_filters(kept, removed, &removed_per_filter);
        (candidates, all_removed)
    }

    /// Run all scorers sequentially and apply their results to candidates.
    #[tracing::instrument(skip_all, name = "scorers", fields(
        total_count = Empty,
        enabled_count = Empty,
        disabled = Empty,
    ))]
    async fn score(&self, query: &Q, mut candidates: Vec<C>) -> Vec<C> {
        let start = Instant::now();
        let all = self.scorers();
        Self::record_enabled_components(all.iter(), |s| s.enable(query), |s| s.name());
        for scorer in all.iter().filter(|s| s.enable(query)) {
            let scored = scorer.run(query, &candidates).await;
            scorer.update_all(&mut candidates, scored);
        }
        self.log_stage_size(start, candidates.len());
        candidates
    }

    /// Select (sort/truncate) candidates using the configured selector
    fn select(&self, query: &Q, candidates: Vec<C>) -> SelectResult<C> {
        if self.selector().enable(query) {
            self.selector().run(query, candidates)
        } else {
            SelectResult {
                selected: candidates,
                non_selected: vec![],
            }
        }
    }

    // Run all side effects in parallel
    fn run_side_effects(&self, input: Arc<SideEffectInput<Q, C>>) {
        let side_effects = self.side_effects();
        tokio::spawn(async move {
            let futures = side_effects
                .iter()
                .filter(|se| se.enable(input.query.clone()))
                .map(|se| se.run(input.clone()));
            let _ = join_all(futures).await;
        });
    }

    // -------------------------- Helpers --------------------------

    /// Iterate components, applying `is_enabled` to each, and record
    /// `total_count`, `enabled_count`, and (if any are disabled) the
    /// comma-joined names of disabled components on the current tracing span.
    fn record_enabled_components<'a, T: 'a>(
        items: impl Iterator<Item = &'a T>,
        is_enabled: impl Fn(&T) -> bool,
        get_name: impl Fn(&T) -> &str,
    ) {
        let mut total = 0usize;
        let mut disabled: Vec<&str> = Vec::new();
        for item in items {
            total += 1;
            if !is_enabled(item) {
                disabled.push(get_name(item));
            }
        }
        let span = Span::current();
        span.record("total_count", total);
        span.record("enabled_count", total - disabled.len());
        if !disabled.is_empty() {
            span.record("disabled", disabled.join(",").as_str());
        }
    }

    // -------------------------- Logging and Stats --------------------------

    fn log_stage(&self, start: Instant) {
        info!("latency_ms={}", start.elapsed().as_millis());
    }

    fn log_stage_size(&self, start: Instant, size: usize) {
        info!("latency_ms={} size={}", start.elapsed().as_millis(), size);
    }

    fn log_filters(&self, kept: usize, removed: usize, removed_per_filter: &[(String, usize)]) {
        let removed_summary = removed_per_filter
            .iter()
            .map(|(name, removed)| format!("{}={}", name, removed))
            .collect::<Vec<_>>()
            .join(",");
        info!(
            "kept {}, removed {} removed_per_filter [{}]",
            kept, removed, removed_summary,
        );
    }

    fn stat_result_size(&self, final_candidates: &[C]) {
        if let Some(receiver) = global_stats_receiver() {
            let response_size = final_candidates.len();
            let metric_name = format!("{}.execute", self.name());
            receiver.observe(
                metric_name.as_str(),
                &FINAL_RESULT_SIZE_SCOPE,
                response_size as f64,
                HistogramBuckets::Bucket0To50,
            );
            if response_size == 0 {
                receiver.incr(metric_name.as_str(), &FINAL_RESULT_EMPTY_SCOPE, 1u64);
            }
        }
    }
}
