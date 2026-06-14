from grox.plans.plan import Plan
from grox.tasks.task_pub import TaskUpsertTweetBoolMetadataToUnifiedPostAnnotation
from grox.schedules.types import TaskEligibility
from grox.tasks.task_filters import TaskPostSafetyDeluxeFilter
from grox.tasks.task_media import TaskMediaHydrationBanger
from grox.tasks.task_post_safety_screen_deluxe import TaskPostSafetyScreenDeluxe
from grox.tasks.task_rate_limit import TaskRateLimitPostSafetyAnnotationWithPost
from grox.tasks.task_grok_upa_action_with_labels import TaskGrokUpaActionWithLabels


class PlanPostSafety(Plan):
    REQUIRED_ELIGIBILITY = TaskEligibility.POST_SAFETY

    TASKS = {
        "task_post_safety_deluxe_filter": TaskPostSafetyDeluxeFilter,
        "task_post_safety_annotation_rate_limit": TaskRateLimitPostSafetyAnnotationWithPost,
        "task_media_hydration": TaskMediaHydrationBanger,
        "task_post_safety_screen_deluxe": TaskPostSafetyScreenDeluxe,
        "task_grok_upa_action_with_labels": TaskGrokUpaActionWithLabels,
        "task_upsert_tweet_bool_metadata_to_unified_post_annotations_manhattan": TaskUpsertTweetBoolMetadataToUnifiedPostAnnotation,
    }

    TASK_DEPENDENCIES = {
        "task_post_safety_deluxe_filter": set(),
        "task_post_safety_annotation_rate_limit": {"task_post_safety_deluxe_filter"},
        "task_media_hydration": {"task_post_safety_annotation_rate_limit"},
        "task_post_safety_screen_deluxe": {"task_media_hydration"},
        "task_grok_upa_action_with_labels": {"task_post_safety_screen_deluxe"},
        "task_upsert_tweet_bool_metadata_to_unified_post_annotations_manhattan": {
            "task_post_safety_screen_deluxe"
        },
    }
