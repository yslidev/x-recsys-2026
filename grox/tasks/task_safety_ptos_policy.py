import logging

from grox.tasks.task import Task, TaskWithPost, TaskResultCategory
from monitor.metrics import Metrics
from grox.schedules.types import TaskContext
from grox.data_loaders.data_types import (
    Post,
    SafetyPolicyCategory,
    SafetyPolicyType,
    SafetyPtosViolatedPolicy,
)
from grox.classifiers.content.safety_ptos import SafetyPtosPolicyClassifier
from grox.config.config import TaskGeneratorType

logger = logging.getLogger(__name__)


class TaskSafetyPtosPolicyDetection(TaskWithPost):
    violated_policy_classifier = SafetyPtosPolicyClassifier()
    deluxe_violated_policy_classifier = SafetyPtosPolicyClassifier(deluxe=True)

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        if not ctx.safety_annotations:
            return

        is_deluxe = ctx.payload.task_type == TaskGeneratorType.SAFETY_PTOS_DELUXE
        active_classifier = (
            cls.deluxe_violated_policy_classifier
            if is_deluxe
            else cls.violated_policy_classifier
        )
        metric_prefix = (
            "task.safety_ptos_deluxe_policy" if is_deluxe else "task.safety_ptos_policy"
        )

        violations = list(ctx.safety_annotations.violatedPolicies or [])

        injected_recheck = None
        if is_deluxe:
            if not any(
                v.category == SafetyPolicyCategory.AdultContent for v in violations
            ):
                injected_recheck = SafetyPtosViolatedPolicy(
                    category=SafetyPolicyCategory.AdultContent,
                    reason="adult content recheck",
                    score=50,
                )
                violations.append(injected_recheck)

        for violation in violations:
            violation.safetyPolicy = (
                await active_classifier.classify_policy_for_violation(post, violation)
            )
            if violation.safetyPolicy:
                cls._record_policy_metrics(metric_prefix, violation)

        if injected_recheck is not None:
            policy = injected_recheck.safetyPolicy
            if not policy or policy.policyType == SafetyPolicyType.NoViolation:
                violations.remove(injected_recheck)

        ctx.safety_annotations.violatedPolicies = violations

    @classmethod
    def _record_policy_metrics(
        cls, metric_prefix: str, violation: SafetyPtosViolatedPolicy
    ) -> None:
        Metrics.counter(f"{metric_prefix}.classified_total.count").add(1)
        category_key = {
            SafetyPolicyCategory.ViolentMedia: "violent_media",
            SafetyPolicyCategory.AdultContent: "adult_content",
            SafetyPolicyCategory.Spam: "spam",
            SafetyPolicyCategory.IllegalAndRegulatedBehaviors: "illegal_and_regulated_behaviors",
            SafetyPolicyCategory.HateOrAbuse: "hate_or_abuse",
            SafetyPolicyCategory.ViolentSpeech: "violent_speech",
            SafetyPolicyCategory.SuicideOrSelfHarm: "suicide_or_self_harm",
        }.get(violation.category)
        if category_key:
            Metrics.counter(
                f"{metric_prefix}.classified_{category_key}_violations.count"
            ).add(1)
            Metrics.counter(
                f"{metric_prefix}.classified_{category_key}_policy_types.count"
            ).add(1, attributes={"policy_type": violation.safetyPolicy.policyType.name})

    @classmethod
    async def exec(cls, ctx: TaskContext) -> TaskResultCategory:
        return await Task.exec.__wrapped__(cls, ctx)
