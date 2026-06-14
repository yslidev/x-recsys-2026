import logging

from grox.tasks.task import Task, TaskWithPost, TaskResultCategory
from monitor.metrics import Metrics
from grox.schedules.types import TaskContext
from grox.data_loaders.data_types import Post
from grox.classifiers.content.safety_ptos import SafetyPtosCategoryClassifier
from grox.config.config import ModelName, TaskGeneratorType

logger = logging.getLogger(__name__)


class TaskSafetyPtosCategoryDetection(TaskWithPost):
    classifier = SafetyPtosCategoryClassifier(ModelName.VLM_SAFETY)
    deluxe_classifier = SafetyPtosCategoryClassifier(
        ModelName.VLM_PRIMARY_CRITICAL, deluxe=True
    )

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        is_deluxe = ctx.payload.task_type == TaskGeneratorType.SAFETY_PTOS_DELUXE
        active_classifier = cls.deluxe_classifier if is_deluxe else cls.classifier
        metric_prefix = (
            "task.safety_ptos_deluxe_category"
            if is_deluxe
            else "task.safety_ptos_category"
        )

        safety_annotations = await active_classifier.classify_post(post)
        ctx.safety_annotations = safety_annotations

        safety_categories = safety_annotations.violatedPolicies or []
        violation_count = len(safety_categories)
        has_violations = violation_count > 0
        Metrics.counter(f"{metric_prefix}.classified.count").add(1)

        if has_violations:
            Metrics.counter(f"{metric_prefix}.has_violations.count").add(1)
            Metrics.counter(f"{metric_prefix}.violations.count").add(violation_count)

            violation_details = []
            for violation in safety_categories:
                Metrics.counter(f"{metric_prefix}.violations_by_category.count").add(
                    1, attributes={"category": violation.category.value}
                )
                violation_details.append(
                    f"{violation.category.value}({violation.score})"
                )

            mode = " (deluxe)" if is_deluxe else ""
            logger.info(
                f"Post {post.id}: Found {violation_count} violations{mode} - Details: {', '.join(violation_details)}"
            )
        else:
            Metrics.counter(f"{metric_prefix}.no_violations.count").add(1)
            mode = " (deluxe)" if is_deluxe else ""
            logger.info(f"Post {post.id}: No safety violations detected{mode}")

    @classmethod
    async def exec(cls, ctx: TaskContext) -> TaskResultCategory:
        return await Task.exec.__wrapped__(cls, ctx)
