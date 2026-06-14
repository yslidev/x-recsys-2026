from grox.plans.plan import Plan
from grox.tasks.task_pub import TaskPublishKafka, TaskWriteReplySpamManhattan
from grox.schedules.types import TaskEligibility
from grox.tasks.task_media import TaskMediaHydration
from grox.tasks.task_filters import TaskSpamFilter
from grox.tasks.task_spam_detection import TaskSpamDetection
from grox.tasks.task_rate_limit import TaskRateLimitReplySpamAnnotationWithPost


class PlanSpamComment(Plan):
    REQUIRED_ELIGIBILITY = TaskEligibility.SPAM_COMMENT

    TASKS = {
        "task_spam_filter": TaskSpamFilter,
        "task_reply_spam_annotation_rate_limit": TaskRateLimitReplySpamAnnotationWithPost,
        "task_media_hydration": TaskMediaHydration,
        "task_spam_detection": TaskSpamDetection,
        "task_publish_reply_spam_mh": TaskWriteReplySpamManhattan,
        "task_publish_kafka": TaskPublishKafka,
    }

    TASK_DEPENDENCIES = {
        "task_spam_filter": set(),
        "task_reply_spam_annotation_rate_limit": {"task_spam_filter"},
        "task_media_hydration": {"task_reply_spam_annotation_rate_limit"},
        "task_spam_detection": {"task_media_hydration"},
        "task_publish_reply_spam_mh": {"task_spam_detection"},
        "task_publish_kafka": {"task_spam_detection"},
    }
