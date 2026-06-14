use std::any::{Any, type_name_of_val};
use tonic::async_trait;

use crate::candidate_pipeline::{PipelineCandidate, PipelineQuery};
use crate::util;
use tracing::{error, info};

#[async_trait]
pub trait Source<Q, C>: Any + Send + Sync
where
    Q: PipelineQuery,
    C: PipelineCandidate,
{
    /// Decide if this source should run for the given query
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    #[xai_stats_macro::receive_stats(size=Bucket500To1000)]
    #[tracing::instrument(skip_all, name = "source", fields(name = self.name()))]
    async fn run(&self, query: &Q) -> Result<Vec<C>, String> {
        match self.source(query).await {
            Ok(candidates) => {
                info!("Fetched {} candidates", candidates.len());
                Ok(candidates)
            }
            Err(err) => {
                error!("Failed: {}", err);
                Err(err)
            }
        }
    }

    async fn source(&self, query: &Q) -> Result<Vec<C>, String>;

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}
