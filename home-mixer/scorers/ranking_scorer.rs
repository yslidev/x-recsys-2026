use crate::models::candidate::{PhoenixScores, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::params::*;
use crate::util::score_normalizer::normalize_score;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::time::Duration;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::utils::duration_since_creation_opt;
use xai_candidate_pipeline::scorer::Scorer;

struct ScoringWeights {
    favorite: f64,
    reply: f64,
    retweet: f64,
    photo_expand: f64,
    click: f64,
    profile_click: f64,
    vqv: f64,
    share: f64,
    share_via_dm: f64,
    share_via_copy_link: f64,
    dwell: f64,
    quote: f64,
    quoted_click: f64,
    quoted_vqv: f64,
    cont_dwell_time: f64,
    cont_click_dwell_time: f64,
    follow_author: f64,
    not_interested: f64,
    block_author: f64,
    mute_author: f64,
    report: f64,
    not_dwelled: f64,
    negative_sum: f64,
    total_sum: f64,
    min_video_duration_ms: i32,
    enable_quoted_vqv_duration_check: bool,
}

impl ScoringWeights {
    fn from_params(params: &xai_feature_switches::Params) -> Self {
        let favorite = params.get(FavoriteWeight);
        let reply = params.get(ReplyWeight);
        let retweet = params.get(RetweetWeight);
        let photo_expand = params.get(PhotoExpandWeight);
        let click = params.get(ClickWeight);
        let profile_click = params.get(ProfileClickWeight);
        let vqv = params.get(VqvWeight);
        let share = params.get(ShareWeight);
        let share_via_dm = params.get(ShareViaDmWeight);
        let share_via_copy_link = params.get(ShareViaCopyLinkWeight);
        let dwell = params.get(DwellWeight);
        let quote = params.get(QuoteWeight);
        let quoted_click = params.get(QuotedClickWeight);
        let quoted_vqv = params.get(QuotedVqvWeight);
        let cont_dwell_time = params.get(ContDwellTimeWeight);
        let cont_click_dwell_time = params.get(ContClickDwellTimeWeight);
        let follow_author = params.get(FollowAuthorWeight);
        let not_interested = params.get(NotInterestedWeight);
        let block_author = params.get(BlockAuthorWeight);
        let mute_author = params.get(MuteAuthorWeight);
        let report = params.get(ReportWeight);
        let not_dwelled = params.get(NotDwelledWeight);
        let min_video_duration_ms = params.get(MinVideoDurationMs);
        let enable_quoted_vqv_duration_check = params.get(EnableQuotedVqvDurationCheck);

        let positive_sum = favorite
            + reply
            + retweet
            + photo_expand
            + click
            + profile_click
            + vqv
            + share
            + share_via_dm
            + share_via_copy_link
            + dwell
            + quote
            + quoted_click
            + quoted_vqv
            + follow_author;
        let negative_sum = -(not_interested + block_author + mute_author + report + not_dwelled);
        let total_sum = positive_sum + negative_sum;

        Self {
            favorite,
            reply,
            retweet,
            photo_expand,
            click,
            profile_click,
            vqv,
            share,
            share_via_dm,
            share_via_copy_link,
            dwell,
            quote,
            quoted_click,
            quoted_vqv,
            cont_dwell_time,
            cont_click_dwell_time,
            follow_author,
            not_interested,
            block_author,
            mute_author,
            report,
            not_dwelled,
            negative_sum,
            total_sum,
            min_video_duration_ms,
            enable_quoted_vqv_duration_check,
        }
    }
}

pub struct RankingScorer;

impl RankingScorer {

    fn apply(score: Option<f64>, weight: f64) -> f64 {
        score.unwrap_or(0.0) * weight
    }

    fn compute_weighted_score(
        weights: &ScoringWeights,
        query: &ScoredPostsQuery,
        candidate: &PostCandidate,
    ) -> f64 {
        let scores: &PhoenixScores = &candidate.phoenix_scores;

        let vqv_weight = crate::util::candidates_util::vqv_weight(
            query,
            candidate,
            weights.min_video_duration_ms,
            weights.vqv,
        );

        let quoted_vqv_weight = crate::util::candidates_util::quoted_vqv_weight(
            candidate,
            weights.min_video_duration_ms,
            weights.quoted_vqv,
            weights.enable_quoted_vqv_duration_check,
        );

        let combined_score = Self::apply(scores.favorite_score, weights.favorite)
            + Self::apply(scores.reply_score, weights.reply)
            + Self::apply(scores.retweet_score, weights.retweet)
            + Self::apply(scores.photo_expand_score, weights.photo_expand)
            + Self::apply(scores.click_score, weights.click)
            + Self::apply(scores.profile_click_score, weights.profile_click)
            + Self::apply(scores.vqv_score, vqv_weight)
            + Self::apply(scores.share_score, weights.share)
            + Self::apply(scores.share_via_dm_score, weights.share_via_dm)
            + Self::apply(
                scores.share_via_copy_link_score,
                weights.share_via_copy_link,
            )
            + Self::apply(scores.dwell_score, weights.dwell)
            + Self::apply(scores.quote_score, weights.quote)
            + Self::apply(scores.quoted_click_score, weights.quoted_click)
            + Self::apply(scores.quoted_vqv_score, quoted_vqv_weight)
            + Self::apply(scores.dwell_time, weights.cont_dwell_time)
            + Self::apply(scores.click_dwell_time, weights.cont_click_dwell_time)
            + Self::apply(scores.follow_author_score, weights.follow_author)
            + Self::apply(scores.not_interested_score, weights.not_interested)
            + Self::apply(scores.block_author_score, weights.block_author)
            + Self::apply(scores.mute_author_score, weights.mute_author)
            + Self::apply(scores.report_score, weights.report)
            + Self::apply(scores.not_dwelled_score, weights.not_dwelled);

        Self::offset_score(combined_score, weights)
    }

    fn offset_score(combined_score: f64, w: &ScoringWeights) -> f64 {
        if w.total_sum == 0.0 {
            combined_score.max(0.0)
        } else if combined_score < 0.0 {
            (combined_score + w.negative_sum) / w.total_sum * NEGATIVE_SCORES_OFFSET
        } else {
            combined_score + NEGATIVE_SCORES_OFFSET
        }
    }


    fn diversity_multiplier(decay_factor: f64, floor: f64, position: usize) -> f64 {
        (1.0 - floor) * decay_factor.powf(position as f64) + floor
    }

    fn apply_author_diversity(
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
        weighted_scores: &[f64],
    ) -> Vec<f64> {
        let decay_factor = query.params.get(AuthorDiversityDecay);
        let floor = query.params.get(AuthorDiversityFloor);

        let mut indexed: Vec<(usize, f64)> = weighted_scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

        let mut adjusted = vec![0.0_f64; candidates.len()];
        let mut author_counts: HashMap<u64, usize> = HashMap::new();

        for (idx, weighted) in indexed {
            let author_id = candidates[idx].author_id;
            let position = author_counts.entry(author_id).or_insert(0);
            let multiplier = Self::diversity_multiplier(decay_factor, floor, *position);
            *position += 1;
            adjusted[idx] = weighted * multiplier;
        }

        adjusted
    }


    fn effective_oon_weight(query: &ScoredPostsQuery) -> f64 {
        if !query.topic_ids.is_empty() {
            return query.params.get(TopicOonWeightFactor);
        }

        let oon_weight_factor = query.params.get(OonWeightFactor);

        let new_user_age_threshold = Duration::from_secs(query.params.get(NewUserAgeThresholdSecs));

        let is_eligible_new_user = duration_since_creation_opt(query.user_id)
            .map(|age| age < new_user_age_threshold)
            .unwrap_or(false)
            && query.user_features.followed_user_ids.len() >= NEW_USER_MIN_FOLLOWING;

        if is_eligible_new_user {
            NEW_USER_OON_WEIGHT_FACTOR
        } else {
            oon_weight_factor
        }
    }
}

#[async_trait]
impl Scorer<ScoredPostsQuery, PostCandidate> for RankingScorer {
    fn enable(&self, _query: &ScoredPostsQuery) -> bool {
        true
    }

    async fn score(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let weights = ScoringWeights::from_params(&query.params);

        let weighted_scores: Vec<f64> = candidates
            .iter()
            .map(|c| {
                let raw = Self::compute_weighted_score(&weights, query, c);
                normalize_score(c, raw)
            })
            .collect();

        let diversity_adjusted = Self::apply_author_diversity(query, candidates, &weighted_scores);

        let effective_oon = Self::effective_oon_weight(query);

        candidates
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let after_diversity = diversity_adjusted[i];
                let final_score = match c.in_network {
                    Some(false) => after_diversity * effective_oon,
                    _ => after_diversity,
                };

                Ok(PostCandidate {
                    weighted_score: Some(weighted_scores[i]),
                    score: Some(final_score),
                    ..Default::default()
                })
            })
            .collect()
    }

    fn update(&self, candidate: &mut PostCandidate, scored: PostCandidate) {
        candidate.weighted_score = scored.weighted_score;
        candidate.score = scored.score;
    }
}
