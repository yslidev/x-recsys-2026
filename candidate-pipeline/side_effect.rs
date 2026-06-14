use crate::candidate_pipeline::{PipelineCandidate, PipelineQuery};
use crate::util;
use std::any::type_name_of_val;
use std::sync::Arc;
use tonic::async_trait;

// A side-effect is an action run that doesn't affect the pipeline result from being returned

#[derive(Clone)]
pub struct SideEffectInput<Q, C> {
    pub query: Arc<Q>,
    pub selected_candidates: Vec<C>,
    pub non_selected_candidates: Vec<C>,
}

#[async_trait]
pub trait SideEffect<Q, C>: Send + Sync
where
    Q: PipelineQuery,
    C: PipelineCandidate,
{
    /// Decide if this side-effect should be run
    fn enable(&self, _query: Arc<Q>) -> bool {
        true
    }

    #[xai_stats_macro::receive_stats]
    async fn run(&self, input: Arc<SideEffectInput<Q, C>>) -> Result<(), String> {
        self.side_effect(input).await
    }

    async fn side_effect(&self, input: Arc<SideEffectInput<Q, C>>) -> Result<(), String>;

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}
