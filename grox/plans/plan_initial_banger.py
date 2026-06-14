from grox.plans.plan import Plan
from grox.tasks.task_pub import (
    TaskPublishKafka,
    TaskPublishUnifiedPostAnnotationsManhattan,
)
from grox.schedules.types import TaskEligibility
from grox.tasks.task_media import TaskMediaHydrationBanger
from grox.tasks.task_filters import TaskInitialBangerFilter
from grox.tasks.task_banger_screen import TaskBangerScreen
from grox.tasks.task_rate_limit import TaskRateLimitBangerAnnotationWithPost
from grox.tasks.task_grok_upa_action_with_labels import TaskGrokUpaActionWithLabels


class PlanInitialBanger(Plan):
    REQUIRED_ELIGIBILITY = TaskEligibility.BANGER_INITIAL_SCREEN

    TASKS = {
        "task_initial_banger_filter": TaskInitialBangerFilter,
        "task_banger_annotation_rate_limit": TaskRateLimitBangerAnnotationWithPost,
        "task_media_hydration": TaskMediaHydrationBanger,
        "task_banger_screen_initial": TaskBangerScreen,
        "task_grok_upa_action_with_labels": TaskGrokUpaActionWithLabels,
        "task_publish_unified_post_annotations_manhattan": TaskPublishUnifiedPostAnnotationsManhattan,
        "task_publish_kafka": TaskPublishKafka,
    }

    TASK_DEPENDENCIES = {
        "task_initial_banger_filter": set(),
        "task_banger_annotation_rate_limit": {"task_initial_banger_filter"},
        "task_media_hydration": {"task_banger_annotation_rate_limit"},
        "task_banger_screen_initial": {"task_media_hydration"},
        "task_grok_upa_action_with_labels": {"task_banger_screen_initial"},
        "task_publish_unified_post_annotations_manhattan": {
            "task_banger_screen_initial"
        },
        "task_publish_kafka": {"task_publish_unified_post_annotations_manhattan"},
    }
