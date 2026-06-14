use crate::models::candidate::PostCandidate;
use crate::models::candidate_features::TopicFilteringExperiment;
use crate::models::query::ScoredPostsQuery;
use crate::params::topics::*;
use std::collections::HashMap;
use std::collections::HashSet;
use tracing::warn;
use xai_candidate_pipeline::filter::{Filter, FilterResult};
use xai_stats_receiver::global_stats_receiver;

const EXCLUDED_TOPIC_METRIC: &str = "TopicIdsFilter.excluded_topic_id";

pub struct TopicIdsFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for TopicIdsFilter {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.is_topic_request() || query.has_excluded_topics()
    }

    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let (mut kept, mut removed) = if query.is_topic_request() {
            let query_topic_ids: HashSet<i64> = query.topic_ids.iter().copied().collect();
            let expanded = TopicIdExpansion::expand(&query_topic_ids);

            if query.is_bulk_topic_request() {
                let all_topic_ids = TopicIdExpansion::all_production_topic_ids();
                let excluded: HashSet<i64> = all_topic_ids.difference(&expanded).copied().collect();

                let (kept, removed): (Vec<_>, Vec<_>) =
                    candidates
                        .into_iter()
                        .partition(|c| match &c.filtered_topic_ids {
                            Some(candidate_topics) if !candidate_topics.is_empty() => {
                                !candidate_topics.iter().all(|tid| excluded.contains(tid))
                            }
                            _ => true,
                        });
                (kept, removed)
            } else {
                let supertopic_expanded: HashMap<i64, HashSet<i64>> = query
                    .topic_ids
                    .iter()
                    .map(|&tid| (tid, TopicIdExpansion::expand_supertopic(tid)))
                    .collect();

                let (kept, removed): (Vec<_>, Vec<_>) = candidates.into_iter().partition(|c| {
                    let exp_topics = c.filtered_topic_ids.as_deref().unwrap_or(&[]);
                    let unf_topics = c.unfiltered_topic_ids.as_deref().unwrap_or(&[]);

                    query.topic_ids.iter().any(|&tid| {
                        let tid_expanded = match TopicIdExpansion::category_ids(tid) {
                            Some(ids) => ids,
                            None => std::slice::from_ref(&tid),
                        };

                        let path1 = exp_topics.iter().any(|et| tid_expanded.contains(et));

                        if path1 {
                            return true;
                        }

                        if let Some(st_ids) = supertopic_expanded.get(&tid) {
                            let unf_has_topic =
                                unf_topics.iter().any(|ut| tid_expanded.contains(ut));
                            let exp_has_supertopic =
                                exp_topics.iter().any(|et| st_ids.contains(et));
                            unf_has_topic && exp_has_supertopic
                        } else {
                            false
                        }
                    })
                });
                (kept, removed)
            }
        } else {
            (candidates, vec![])
        };

        if query.has_excluded_topics() {
            if let Some(receiver) = global_stats_receiver() {
                for &tid in &query.excluded_topic_ids {
                    let tid_str = tid.to_string();
                    receiver.incr(EXCLUDED_TOPIC_METRIC, &[("topic_id", &tid_str)], 1);
                }
            }
            let excluded_ids: HashSet<i64> = query.excluded_topic_ids.iter().copied().collect();
            let mut excluded_expanded = TopicIdExpansion::expand(&excluded_ids);
            excluded_expanded.extend(&excluded_ids);

            let (new_kept, new_removed): (Vec<_>, Vec<_>) =
                kept.into_iter().partition(|c| match &c.filtered_topic_ids {
                    Some(candidate_topics) if !candidate_topics.is_empty() => !candidate_topics
                        .iter()
                        .any(|tid| excluded_expanded.contains(tid)),
                    _ => false,
                });
            kept = new_kept;
            removed.extend(new_removed);
        }

        FilterResult { kept, removed }
    }
}

pub struct TopicIdExpansion;

impl TopicIdExpansion {
    pub fn category_ids(topic_id: i64) -> Option<&'static [i64]> {
        match topic_id {
            SCIENCE_TECHNOLOGY => Some(&[
                XAI_SCIENCE,
                XAI_TECHNOLOGY,
                XAI_AI,
                XAI_SOFTWARE_DEVELOPMENT,
                XAI_ROBOTICS,
                XAI_SPACE,
                XAI_BIOTECH,
                XAI_ELECTRONICS,
            ]),
            ENTERTAINMENT => Some(&[
                XAI_MOVIES_TV,
                XAI_STREAMING,
                XAI_MUSIC,
                XAI_DANCE,
                XAI_CELEBRITY,
                XAI_GAMING,
                XAI_ANIME,
            ]),
            BUSINESS_FINANCE => Some(&[
                XAI_BUSINESS_FINANCE_REAL,
                XAI_CRYPTOCURRENCY,
                XAI_STOCKS,
                XAI_ENTREPRENEURSHIP,
                XAI_REAL_ESTATE,
                XAI_PERSONAL_FINANCE,
            ]),
            SPORTS => Some(&[
                XAI_SPORTS_REAL,
                XAI_ATHLETICS,
                XAI_SOCCER,
                XAI_BUNDESLIGA,
                XAI_LA_LIGA,
                XAI_LIGUE1,
                XAI_UEFA_CL,
                XAI_UEFA_EL,
                XAI_UEFA_ECL,
                XAI_BASKETBALL,
                XAI_NBA,
                XAI_WNBA,
                XAI_AMERICAN_FOOTBALL,
                XAI_NCAA_FOOTBALL,
                XAI_NFL,
                XAI_BASEBALL,
                XAI_MLB,
                XAI_TENNIS,
                XAI_CRICKET,
                XAI_IPL,
                XAI_MMA,
                XAI_WWE,
                XAI_BOXING,
                XAI_GOLF,
                XAI_RACING,
                XAI_FORMULA1,
                XAI_NASCAR,
                XAI_ICE_HOCKEY,
                XAI_NHL,
                XAI_OLYMPICS,
                XAI_RUGBY,
                XAI_CYCLING,
                XAI_SKIING,
                XAI_SNOWBOARDING,
                XAI_ESPORTS,
                XAI_MOTO_GP,
                XAI_PREMIER_LEAGUE,
                XAI_SERIE_A,
            ]),
            XAI_POLITICS => Some(&[XAI_POLITICS]),
            XAI_AI => Some(&[XAI_AI]),
            XAI_GAMING => Some(&[XAI_GAMING]),
            XAI_CRYPTOCURRENCY => Some(&[XAI_CRYPTOCURRENCY]),
            XAI_US_IRAN_WAR => Some(&[XAI_US_IRAN_WAR]),
            XAI_NEWS => Some(&[XAI_NEWS, XAI_NATURAL_DISASTERS]),
            XAI_MOVIES_TV => Some(&[XAI_MOVIES_TV, XAI_STREAMING]),
            XAI_MUSIC => Some(&[XAI_MUSIC]),
            XAI_DANCE => Some(&[XAI_DANCE]),
            XAI_CELEBRITY => Some(&[XAI_CELEBRITY]),
            XAI_ANIME => Some(&[XAI_ANIME]),
            XAI_SOCCER => Some(&[
                XAI_SOCCER,
                XAI_BUNDESLIGA,
                XAI_LA_LIGA,
                XAI_LIGUE1,
                XAI_UEFA_CL,
                XAI_UEFA_EL,
                XAI_UEFA_ECL,
            ]),
            XAI_BASKETBALL => Some(&[XAI_BASKETBALL, XAI_NBA, XAI_WNBA]),
            XAI_AMERICAN_FOOTBALL => Some(&[
                XAI_AMERICAN_FOOTBALL,
                XAI_NCAA_FOOTBALL,
                XAI_NFL,
            ]),
            XAI_BASEBALL => Some(&[XAI_BASEBALL, XAI_MLB]),
            XAI_TENNIS => Some(&[XAI_TENNIS]),
            XAI_CRICKET => Some(&[XAI_CRICKET, XAI_IPL]),
            XAI_MMA => Some(&[XAI_MMA, XAI_WWE]),
            XAI_BOXING => Some(&[XAI_BOXING]),
            XAI_GOLF => Some(&[XAI_GOLF]),
            XAI_RACING => Some(&[XAI_RACING, XAI_FORMULA1, XAI_NASCAR]),
            XAI_ICE_HOCKEY => Some(&[XAI_ICE_HOCKEY, XAI_NHL]),
            XAI_OLYMPICS => Some(&[XAI_OLYMPICS]),
            XAI_RUGBY => Some(&[XAI_RUGBY]),
            XAI_CYCLING => Some(&[XAI_CYCLING]),
            XAI_SKIING => Some(&[XAI_SKIING, XAI_SNOWBOARDING]),
            XAI_HEALTH_FITNESS => Some(&[
                XAI_HEALTH_FITNESS,
                XAI_NUTRITION,
                XAI_WORKOUTS,
            ]),
            XAI_TRAVEL => Some(&[XAI_TRAVEL, XAI_AVIATION]),
            XAI_FOOD => Some(&[
                XAI_FOOD,
                XAI_COOKING,
                XAI_BAKING,
                XAI_RESTAURANTS,
                XAI_DRINKS,
            ]),
            XAI_MEMES => Some(&[XAI_MEMES]),
            XAI_BEAUTY => Some(&[XAI_BEAUTY]),
            XAI_FASHION => Some(&[XAI_FASHION]),
            XAI_NATURE_OUTDOORS => Some(&[XAI_NATURE_OUTDOORS]),
            XAI_PETS => Some(&[XAI_PETS, XAI_CATS, XAI_DOGS]),
            XAI_HOME_GARDEN => Some(&[XAI_HOME_GARDEN]),
            XAI_ART => Some(&[XAI_ART]),
            XAI_RELIGION => Some(&[XAI_RELIGION]),
            XAI_SHOPPING => Some(&[XAI_SHOPPING]),
            XAI_EDUCATION => Some(&[XAI_EDUCATION]),
            XAI_CAREER => Some(&[XAI_CAREER]),
            XAI_CARS => Some(&[XAI_CARS]),
            XAI_MOTORCYCLES => Some(&[XAI_MOTORCYCLES]),
            XAI_RELATIONSHIPS => Some(&[XAI_RELATIONSHIPS]),
            XAI_FAMILY => Some(&[XAI_FAMILY, XAI_MARRIAGE, XAI_PARENTING]),
            XAI_PODCASTS => Some(&[XAI_PODCASTS]),
            XAI_POP => Some(&[XAI_POP]),
            XAI_K_POP => Some(&[XAI_K_POP]),
            XAI_COUNTRY_MUSIC => Some(&[XAI_COUNTRY_MUSIC]),
            XAI_ELECTRONIC => Some(&[XAI_ELECTRONIC]),
            XAI_J_POP => Some(&[XAI_J_POP]),
            XAI_ROCK => Some(&[XAI_ROCK]),
            XAI_PHOTOGRAPHY => Some(&[XAI_PHOTOGRAPHY]),
            XAI_DESIGN => Some(&[XAI_DESIGN]),
            XAI_SOFTWARE_DEVELOPMENT => Some(&[XAI_SOFTWARE_DEVELOPMENT]),
            XAI_ROBOTICS => Some(&[XAI_ROBOTICS]),
            XAI_SPACE => Some(&[XAI_SPACE]),
            XAI_BIOTECH => Some(&[XAI_BIOTECH]),
            XAI_ELECTRONICS => Some(&[XAI_ELECTRONICS]),
            XAI_STOCKS => Some(&[XAI_STOCKS]),
            XAI_ENTREPRENEURSHIP => Some(&[XAI_ENTREPRENEURSHIP]),
            XAI_REAL_ESTATE => Some(&[XAI_REAL_ESTATE]),
            XAI_PERSONAL_FINANCE => Some(&[XAI_PERSONAL_FINANCE]),
            XAI_CRIME => Some(&[XAI_CRIME]),
            XAI_DATING => Some(&[XAI_DATING]),
            XAI_ELECTIONS => Some(&[XAI_ELECTIONS]),
            XAI_ESPORTS => Some(&[XAI_ESPORTS]),
            XAI_HIP_HOP => Some(&[XAI_HIP_HOP]),
            XAI_JAZZ => Some(&[XAI_JAZZ]),
            XAI_CONCERTS => Some(&[XAI_CONCERTS]),
            XAI_MENTAL_HEALTH => Some(&[XAI_MENTAL_HEALTH]),
            XAI_MOTO_GP => Some(&[XAI_MOTO_GP]),
            XAI_PREMIER_LEAGUE => Some(&[XAI_PREMIER_LEAGUE]),
            XAI_SERIE_A => Some(&[XAI_SERIE_A]),
            XAI_CHRISTIANITY => Some(&[XAI_CHRISTIANITY]),
            XAI_BUDDHISM => Some(&[XAI_BUDDHISM]),
            XAI_HINDUISM => Some(&[XAI_HINDUISM]),
            XAI_ISLAM => Some(&[XAI_ISLAM]),
            XAI_JUDAISM => Some(&[XAI_JUDAISM]),
            XAI_DIGITAL_ART => Some(&[XAI_DIGITAL_ART]),
            XAI_SCIENCE => Some(&[XAI_SCIENCE]),
            XAI_TECHNOLOGY => Some(&[XAI_TECHNOLOGY]),
            XAI_BUSINESS_FINANCE_REAL => Some(&[XAI_BUSINESS_FINANCE_REAL]),
            XAI_SPORTS_REAL => Some(&[XAI_SPORTS_REAL]),
            XAI_ATHLETICS => Some(&[XAI_ATHLETICS]),
            XAI_STREAMING => Some(&[XAI_STREAMING]),
            XAI_STOCKS_ECONOMY => Some(&[XAI_STOCKS_ECONOMY]),
            _ => None,
        }
    }

    pub fn expand(topic_ids: &HashSet<i64>) -> HashSet<i64> {
        let mut expanded = HashSet::new();
        for &topic_id in topic_ids {
            match Self::category_ids(topic_id) {
                Some(ids) => {
                    for &id in ids {
                        expanded.insert(id);
                    }
                }
                None => {
                    expanded.insert(topic_id);
                }
            }
        }
        expanded
    }

    pub fn all_production_topic_ids() -> HashSet<i64> {
        let all_client_topics: HashSet<i64> = [
            SCIENCE_TECHNOLOGY,
            ENTERTAINMENT,
            BUSINESS_FINANCE,
            SPORTS,
            XAI_POLITICS,
            XAI_AI,
            XAI_GAMING,
            XAI_CRYPTOCURRENCY,
            XAI_US_IRAN_WAR,
            XAI_NEWS,
            XAI_MOVIES_TV,
            XAI_MUSIC,
            XAI_DANCE,
            XAI_CELEBRITY,
            XAI_ANIME,
            XAI_SOCCER,
            XAI_BASKETBALL,
            XAI_AMERICAN_FOOTBALL,
            XAI_BASEBALL,
            XAI_TENNIS,
            XAI_CRICKET,
            XAI_MMA,
            XAI_BOXING,
            XAI_GOLF,
            XAI_RACING,
            XAI_ICE_HOCKEY,
            XAI_OLYMPICS,
            XAI_RUGBY,
            XAI_CYCLING,
            XAI_SKIING,
            XAI_HEALTH_FITNESS,
            XAI_TRAVEL,
            XAI_FOOD,
            XAI_MEMES,
            XAI_BEAUTY,
            XAI_FASHION,
            XAI_NATURE_OUTDOORS,
            XAI_PETS,
            XAI_HOME_GARDEN,
            XAI_ART,
            XAI_RELIGION,
            XAI_SHOPPING,
            XAI_EDUCATION,
            XAI_CAREER,
            XAI_CARS,
            XAI_MOTORCYCLES,
            XAI_RELATIONSHIPS,
            XAI_FAMILY,
            XAI_PODCASTS,
            XAI_POP,
            XAI_K_POP,
            XAI_COUNTRY_MUSIC,
            XAI_ELECTRONIC,
            XAI_J_POP,
            XAI_ROCK,
            XAI_PHOTOGRAPHY,
            XAI_DESIGN,
            XAI_SOFTWARE_DEVELOPMENT,
            XAI_ROBOTICS,
            XAI_SPACE,
            XAI_BIOTECH,
            XAI_ELECTRONICS,
            XAI_STOCKS,
            XAI_ENTREPRENEURSHIP,
            XAI_REAL_ESTATE,
            XAI_PERSONAL_FINANCE,
            XAI_CRIME,
            XAI_DATING,
            XAI_ELECTIONS,
            XAI_ESPORTS,
            XAI_HIP_HOP,
            XAI_JAZZ,
            XAI_CONCERTS,
            XAI_MENTAL_HEALTH,
            XAI_MOTO_GP,
            XAI_PREMIER_LEAGUE,
            XAI_SERIE_A,
            XAI_CHRISTIANITY,
            XAI_BUDDHISM,
            XAI_HINDUISM,
            XAI_ISLAM,
            XAI_JUDAISM,
            XAI_DIGITAL_ART,
            XAI_STOCKS_ECONOMY,
        ]
        .into();
        Self::expand(&all_client_topics)
    }

    pub fn resolve_first(topic_id: i64) -> i64 {
        topic_id
    }

    pub fn supertopic(topic_id: i64) -> i64 {
        match topic_id {
            XAI_POP
            | XAI_K_POP
            | XAI_COUNTRY_MUSIC
            | XAI_ELECTRONIC
            | XAI_J_POP
            | XAI_ROCK
            | XAI_HIP_HOP
            | XAI_JAZZ
            | XAI_CONCERTS => XAI_MUSIC,

            XAI_DESIGN | XAI_DIGITAL_ART | XAI_PHOTOGRAPHY => XAI_ART,

            XAI_NATURAL_DISASTERS | XAI_CRIME => XAI_NEWS,

            XAI_COOKING | XAI_BAKING | XAI_RESTAURANTS | XAI_DRINKS => {
                XAI_FOOD
            }

            XAI_AVIATION => XAI_TRAVEL,

            XAI_CATS | XAI_DOGS => XAI_PETS,

            XAI_CHRISTIANITY
            | XAI_BUDDHISM
            | XAI_HINDUISM
            | XAI_ISLAM
            | XAI_JUDAISM => XAI_RELIGION,

            XAI_NUTRITION | XAI_WORKOUTS | XAI_MENTAL_HEALTH => {
                XAI_HEALTH_FITNESS
            }

            XAI_DATING | XAI_MARRIAGE => XAI_RELATIONSHIPS,

            XAI_PARENTING => XAI_FAMILY,

            XAI_ELECTIONS => XAI_POLITICS,

            XAI_CRYPTOCURRENCY
            | XAI_STOCKS
            | XAI_ENTREPRENEURSHIP
            | XAI_REAL_ESTATE
            | XAI_PERSONAL_FINANCE => XAI_BUSINESS_FINANCE_REAL,

            XAI_AI
            | XAI_SOFTWARE_DEVELOPMENT
            | XAI_ROBOTICS
            | XAI_ELECTRONICS => XAI_TECHNOLOGY,

            XAI_BIOTECH | XAI_SPACE => XAI_SCIENCE,

            XAI_BUNDESLIGA
            | XAI_LA_LIGA
            | XAI_LIGUE1
            | XAI_UEFA_CL
            | XAI_UEFA_EL
            | XAI_UEFA_ECL
            | XAI_PREMIER_LEAGUE
            | XAI_SERIE_A => XAI_SOCCER,

            XAI_NBA | XAI_WNBA => XAI_BASKETBALL,

            XAI_NCAA_FOOTBALL | XAI_NFL => XAI_AMERICAN_FOOTBALL,

            XAI_MLB => XAI_BASEBALL,

            XAI_IPL => XAI_CRICKET,

            XAI_NHL => XAI_ICE_HOCKEY,

            XAI_FORMULA1 | XAI_NASCAR | XAI_MOTO_GP => XAI_RACING,

            XAI_ESPORTS => XAI_GAMING,

            XAI_MOTORCYCLES => XAI_CARS,

            XAI_STREAMING => XAI_MOVIES_TV,

            XAI_AMERICAN_FOOTBALL
            | XAI_ATHLETICS
            | XAI_BASEBALL
            | XAI_BASKETBALL
            | XAI_BOXING
            | XAI_CRICKET
            | XAI_CYCLING
            | XAI_GOLF
            | XAI_ICE_HOCKEY
            | XAI_MMA
            | XAI_OLYMPICS
            | XAI_RACING
            | XAI_RUGBY
            | XAI_SKIING
            | XAI_SNOWBOARDING
            | XAI_SOCCER
            | XAI_TENNIS
            | XAI_WWE => XAI_SPORTS_REAL,

            _ => topic_id,
        }
    }

    pub fn expand_supertopic(topic_id: i64) -> HashSet<i64> {
        let st = Self::supertopic(topic_id);
        match Self::category_ids(st) {
            Some(ids) => ids.iter().copied().collect(),
            None => [st].into(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TopicFilteringOverrideMap {
    overrides: HashMap<i64, TopicFilteringExperiment>,
}

impl TopicFilteringOverrideMap {
    pub fn parse(raw: &str) -> Self {
        let mut overrides = HashMap::new();
        if raw.is_empty() {
            return Self { overrides };
        }
        for entry in raw.split(',') {
            let entry = entry.trim();
            if entry.is_empty() {
                continue;
            }
            if let Some((topic_str, experiment_str)) = entry.split_once('=') {
                match topic_str.trim().parse::<i64>() {
                    Ok(topic_id) => {
                        let experiment = TopicFilteringExperiment::parse(experiment_str.trim());
                        overrides.insert(topic_id, experiment);
                    }
                    Err(_) => {
                        warn!(
                            "TopicFilteringOverrides: invalid topic_id '{}' in entry '{}'",
                            topic_str, entry
                        );
                    }
                }
            } else {
                warn!(
                    "TopicFilteringOverrides: malformed entry '{}', expected 'topic_id=ExperimentId'",
                    entry
                );
            }
        }
        Self { overrides }
    }

    pub fn resolve(
        &self,
        query_topic_ids: &[i64],
        default: TopicFilteringExperiment,
    ) -> TopicFilteringExperiment {
        if self.overrides.is_empty() {
            return default;
        }
        for &tid in query_topic_ids {
            if let Some(&exp) = self.overrides.get(&tid) {
                return exp;
            }
        }
        default
    }
}
