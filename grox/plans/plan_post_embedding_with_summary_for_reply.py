from grox.plans.plan import Plan
from grox.schedules.types import TaskEligibility
from grox.tasks.task_filters import TaskPostEmbeddingWithSummaryForReplyFilter
from grox.tasks.task_media import TaskMediaHydration
from grox.tasks.task_multimodal_post_embedding import (
    TaskMultimodalPostEmbeddingWithSummary,
)
from grox.tasks.task_write_mm_embedding_sink import TaskWriteMMEmbeddingSinkV3
from grox.tasks.task_rate_limit import TaskRateLimitEmbeddingWithPostSummaryForReply
from grox.tasks.task_summarizer_for_post_embedding import TaskPostEmbeddingSummarizer


class PlanPostEmbeddingWithSummaryForReply(Plan):
    REQUIRED_ELIGIBILITY = TaskEligibility.POST_EMBEDDING_WITH_SUMMARY_FOR_REPLY

    TASKS = {
        "task_post_embedding_rate_limit_summary_for_reply": TaskRateLimitEmbeddingWithPostSummaryForReply,
        "task_post_embedding_with_summary_filter_for_reply": TaskPostEmbeddingWithSummaryForReplyFilter,
        "task_media_hydration": TaskMediaHydration,
        "task_post_embedding_summarizer": TaskPostEmbeddingSummarizer,
        "task_multimodal_post_embedding_with_summary": TaskMultimodalPostEmbeddingWithSummary,
        "task_write_post_embedding_sink_v3": TaskWriteMMEmbeddingSinkV3,
    }

    TASK_DEPENDENCIES = {
        "task_post_embedding_rate_limit_summary_for_reply": set(),
        "task_post_embedding_with_summary_filter_for_reply": {
            "task_post_embedding_rate_limit_summary_for_reply"
        },
        "task_media_hydration": {"task_post_embedding_with_summary_filter_for_reply"},
        "task_post_embedding_summarizer": {"task_media_hydration"},
        "task_multimodal_post_embedding_with_summary": {
            "task_post_embedding_summarizer"
        },
        "task_write_post_embedding_sink_v3": {
            "task_multimodal_post_embedding_with_summary"
        },
    }
