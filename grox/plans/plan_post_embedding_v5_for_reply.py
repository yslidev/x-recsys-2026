from grox.plans.plan import Plan
from grox.schedules.types import TaskEligibility
from grox.tasks.task_filters import TaskPostEmbeddingWithSummaryForReplyFilter
from grox.tasks.task_media import TaskMediaHydration
from grox.tasks.task_multimodal_post_embedding import TaskMultimodalPostEmbeddingV5
from grox.tasks.task_write_mm_embedding_sink import TaskWriteMMEmbeddingSinkV5
from grox.tasks.task_rate_limit import TaskRateLimitEmbeddingV5ForReply
from grox.tasks.task_asr import TaskASRTranscription


class PlanPostEmbeddingV5ForReply(Plan):
    REQUIRED_ELIGIBILITY = TaskEligibility.MM_EMB_V5_FOR_REPLY

    TASKS = {
        "task_post_embedding_rate_limit_v5_for_reply": TaskRateLimitEmbeddingV5ForReply,
        "task_post_embedding_filter_for_reply": TaskPostEmbeddingWithSummaryForReplyFilter,
        "task_media_hydration": TaskMediaHydration,
        "task_asr_transcription": TaskASRTranscription,
        "task_multimodal_post_embedding_v5": TaskMultimodalPostEmbeddingV5,
        "task_write_post_embedding_sink_v5": TaskWriteMMEmbeddingSinkV5,
    }

    TASK_DEPENDENCIES = {
        "task_post_embedding_rate_limit_v5_for_reply": set(),
        "task_post_embedding_filter_for_reply": {
            "task_post_embedding_rate_limit_v5_for_reply"
        },
        "task_media_hydration": {"task_post_embedding_filter_for_reply"},
        "task_asr_transcription": {"task_media_hydration"},
        "task_multimodal_post_embedding_v5": {"task_asr_transcription"},
        "task_write_post_embedding_sink_v5": {"task_multimodal_post_embedding_v5"},
    }
