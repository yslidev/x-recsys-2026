from grox.plans.plan import Plan
from grox.schedules.types import TaskEligibility
from grox.tasks.task_media import TaskMediaHydration
from grox.tasks.task_filters import TaskSafetyPtosFilter
from grox.tasks.task_safety_ptos_category import TaskSafetyPtosCategoryDetection
from grox.tasks.task_safety_ptos_policy import TaskSafetyPtosPolicyDetection
from grox.tasks.task_rate_limit import TaskRateLimitSafetyPtosAnnotationWithPost
from grox.tasks.task_write_safety_post_annotations_result_sink import (
    TaskWriteSafetyPostAnnotationsResultSink,
)


class PlanSafetyPtos(Plan):
    REQUIRED_ELIGIBILITY = TaskEligibility.SAFETY_PTOS

    TASKS = {
        "task_safety_ptos_filter": TaskSafetyPtosFilter,
        "task_safety_ptos_annotation_rate_limit": TaskRateLimitSafetyPtosAnnotationWithPost,
        "task_media_hydration": TaskMediaHydration,
        "task_safety_ptos_category_detection": TaskSafetyPtosCategoryDetection,
        "task_safety_ptos_policy_detection": TaskSafetyPtosPolicyDetection,
        "task_write_safety_post_annotations_result_sink": TaskWriteSafetyPostAnnotationsResultSink,
    }

    TASK_DEPENDENCIES = {
        "task_safety_ptos_filter": {},
        "task_safety_ptos_annotation_rate_limit": {"task_safety_ptos_filter"},
        "task_media_hydration": {"task_safety_ptos_annotation_rate_limit"},
        "task_safety_ptos_category_detection": {"task_media_hydration"},
        "task_safety_ptos_policy_detection": {"task_safety_ptos_category_detection"},
        "task_write_safety_post_annotations_result_sink": {"task_safety_ptos_policy_detection"},
    }
