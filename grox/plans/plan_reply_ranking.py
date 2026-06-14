from grox.plans.plan import Plan
from grox.tasks.task_pub import TaskWriteReplyRankingManhattan
from grox.schedules.types import TaskEligibility
from grox.tasks.task_media import TaskMediaHydration
from grox.tasks.task_filters import TaskReplyRankingFilter
from grox.tasks.task_rank_replies import TaskRankReplies
from grox.tasks.task_rate_limit import TaskRateLimitReplyRankingAnnotationWithPost


class PlanReplyRanking(Plan):
    REQUIRED_ELIGIBILITY = TaskEligibility.REPLY_RANKING

    TASKS = {
        "task_reply_ranking_filter": TaskReplyRankingFilter,
        "task_reply_ranking_annotation_rate_limit": TaskRateLimitReplyRankingAnnotationWithPost,
        "task_media_hydration": TaskMediaHydration,
        "task_rank_replies": TaskRankReplies,
        "task_write_reply_ranking_manhattan": TaskWriteReplyRankingManhattan,
    }

    TASK_DEPENDENCIES = {
        "task_reply_ranking_filter": set(),
        "task_reply_ranking_annotation_rate_limit": {"task_reply_ranking_filter"},
        "task_media_hydration": {"task_reply_ranking_annotation_rate_limit"},
        "task_rank_replies": {"task_media_hydration"},
        "task_write_reply_ranking_manhattan": {"task_rank_replies"},
    }
