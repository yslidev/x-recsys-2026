use crate::clients::gender_prediction_client::GenderPredictionGrpcClient;
use crate::clients::user_inferred_gender_store_client::UserInferredGenderStoreClient;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableInferredGenderHydration;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::utils::days_since_creation;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;
use xai_recsys_proto::gender_prediction::InferredGenderLabel;

pub struct UserInferredGenderQueryHydrator {
    mh_client: Arc<dyn UserInferredGenderStoreClient>,
    grpc_client: Arc<dyn GenderPredictionGrpcClient>,
}

impl UserInferredGenderQueryHydrator {
    pub fn new(
        mh_client: Arc<dyn UserInferredGenderStoreClient>,
        grpc_client: Arc<dyn GenderPredictionGrpcClient>,
    ) -> Self {
        Self {
            mh_client,
            grpc_client,
        }
    }
}

#[async_trait]
impl QueryHydrator<ScoredPostsQuery> for UserInferredGenderQueryHydrator {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.params.get(EnableInferredGenderHydration) || query.is_shadow_traffic
    }

    async fn hydrate(&self, query: &ScoredPostsQuery) -> Result<ScoredPostsQuery, String> {
        let result = match self.mh_client.fetch(query.user_id).await? {
            Some(r) => Some(r),
            None if is_new_user(query.user_id) => self.grpc_client.predict(query.user_id).await?,
            None => None,
        };

        Ok(ScoredPostsQuery {
            user_inferred_gender: result
                .as_ref()
                .and_then(|r| InferredGenderLabel::try_from(r.gender_label).ok()),
            user_inferred_gender_score: result.and_then(|r| r.prediction_score),
            ..Default::default()
        })
    }

    fn update(&self, query: &mut ScoredPostsQuery, hydrated: ScoredPostsQuery) {
        query.user_inferred_gender = hydrated.user_inferred_gender;
        query.user_inferred_gender_score = hydrated.user_inferred_gender_score;
    }
}

fn is_new_user(user_id: u64) -> bool {
    days_since_creation(user_id) == 0
}
