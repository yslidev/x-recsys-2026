use std::any::{Any, type_name_of_val};
use tonic::async_trait;

use crate::candidate_pipeline::PipelineQuery;
use crate::util;
use tracing::error;

#[async_trait]
pub trait QueryHydrator<Q>: Any + Send + Sync
where
    Q: PipelineQuery,
{
    /// Decide if this query hydrator should run for the given query
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    #[xai_stats_macro::receive_stats]
    #[tracing::instrument(skip_all, name = "query_hydrator", fields(name = self.name()))]
    async fn run(&self, query: &Q) -> Result<Q, String> {
        match self.hydrate(query).await {
            Ok(hydrated) => Ok(hydrated),
            Err(err) => {
                error!("Failed: {}", err);
                Err(err)
            }
        }
    }

    /// Hydrate the query by performing async operations.
    /// Returns a new query with this hydrator's fields populated.
    async fn hydrate(&self, query: &Q) -> Result<Q, String>;

    /// Update the query with the hydrated fields.
    /// Only the fields this hydrator is responsible for should be copied.
    fn update(&self, query: &mut Q, hydrated: Q);

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}
