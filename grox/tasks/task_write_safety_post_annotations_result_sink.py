import logging
import time

from grox.tasks.task import Task
from grox.tasks.disable_rules import DisableTaskForNonPtosProd
from monitor.metrics import Metrics
from grox.schedules.types import TaskContext
from grox.data_loaders.data_types import (
    Image,
    Video,
    SafetyPolicyCategory,
    SafetyPolicyType,
)
from strato_http.queries.data_types import (
    SafetyPostAnnotations,
    SafetyPostAnnotationsResult,
    SafetyBoolMetadata,
    SafetyPtosViolatedPolicy,
    SafetyPolicy,
    FoundMetadata,
)
from strato_http.queries.safety_post_annotations_result import (
    StratoSafetyPostAnnotationsResultMh,
    StratoSafetyPostAnnotationsResultDirectMh,
    StratoSafetyPostAnnotationsResultKafka,
)
from strato_http.queries.grok_ptos_action_with_labels import (
    StratoGrokPtosActionWithLabels,
)
from strato_http.queries.grok_ptos_delete_labels import StratoGrokPtosDeleteLabels

logger = logging.getLogger(__name__)


class TaskWriteSafetyPostAnnotationsResultSink(Task):
    DISABLE_RULES = [DisableTaskForNonPtosProd]

    _strato_mh = StratoSafetyPostAnnotationsResultMh()
    _strato_direct_mh = StratoSafetyPostAnnotationsResultDirectMh()
    _strato_kafka = StratoSafetyPostAnnotationsResultKafka()
    _strato_grok_ptos_action_with_labels = StratoGrokPtosActionWithLabels()
    _strato_grok_ptos_delete_labels = StratoGrokPtosDeleteLabels()

    @staticmethod
    def _build_found_metadata(post) -> FoundMetadata:
        return FoundMetadata(
            imageCount=sum(1 for m in post.media if isinstance(m, Image))
            if post.media
            else 0,
            videoCount=sum(1 for m in post.media if isinstance(m, Video))
            if post.media
            else 0,
            cardV2Count=len(post.cardsV2) if post.cardsV2 else 0,
        )

    @classmethod
    def _compute_bool_metadata_from_violations(
        cls, safety_annotations
    ) -> SafetyBoolMetadata:
        is_gore = False
        is_nsfw = False
        is_soft_nsfw = False
        is_spam = False

        if safety_annotations and safety_annotations.violatedPolicies:
            for violation in safety_annotations.violatedPolicies:
                if (
                    violation.category == SafetyPolicyCategory.ViolentMedia
                    and violation.safetyPolicy
                    and violation.safetyPolicy.policyType
                    != SafetyPolicyType.NoViolation
                ):
                    is_gore = True
                    Metrics.counter(
                        "task.write_safety_post_annotations_result_sink.detected_gore.count"
                    ).add(1)

                if (
                    violation.category == SafetyPolicyCategory.AdultContent
                    and violation.safetyPolicy
                    and violation.safetyPolicy.policyType
                    == SafetyPolicyType.AdultContentSexualHard
                ):
                    is_nsfw = True
                    Metrics.counter(
                        "task.write_safety_post_annotations_result_sink.detected_nsfw.count"
                    ).add(1)

                if (
                    violation.category == SafetyPolicyCategory.AdultContent
                    and violation.safetyPolicy
                    and violation.safetyPolicy.policyType
                    == SafetyPolicyType.AdultContentSexualSoft
                ):
                    is_soft_nsfw = True
                    Metrics.counter(
                        "task.write_safety_post_annotations_result_sink.detected_soft_nsfw.count"
                    ).add(1)

                if (
                    violation.category == SafetyPolicyCategory.Spam
                    and violation.safetyPolicy
                    and violation.safetyPolicy.policyType
                    != SafetyPolicyType.NoViolation
                ):
                    is_spam = True
                    Metrics.counter(
                        "task.write_safety_post_annotations_result_sink.detected_spam.count"
                    ).add(1)

                if (
                    violation.category
                    == SafetyPolicyCategory.IllegalAndRegulatedBehaviors
                    and violation.safetyPolicy
                    and violation.safetyPolicy.policyType
                    != SafetyPolicyType.NoViolation
                ):
                    is_spam = True
                    Metrics.counter(
                        "task.write_safety_post_annotations_result_sink.detected_spam_illegal.count"
                    ).add(1)

        return SafetyBoolMetadata(
            isGore=is_gore, isNsfw=is_nsfw, isSoftNsfw=is_soft_nsfw, isSpam=is_spam
        )

    @classmethod
    def _merge_bool_metadata(
        cls, existing: SafetyBoolMetadata | None, new: SafetyBoolMetadata
    ) -> SafetyBoolMetadata:
        if existing is None:
            return new
        return SafetyBoolMetadata(
            isGore=True if (existing.isGore or new.isGore) else False,
            isNsfw=True if (existing.isNsfw or new.isNsfw) else False,
            isSoftNsfw=True if (existing.isSoftNsfw or new.isSoftNsfw) else False,
            isSpam=True if (existing.isSpam or new.isSpam) else False,
        )

    @classmethod
    async def _exec(cls, ctx: TaskContext) -> None:
        Metrics.counter("task.write_safety_post_annotations_result_sink.count").add(1)

        post = ctx.payload.post
        if not post:
            return

        safety_annotations = ctx.safety_annotations
        if not safety_annotations:
            return

        post_id = int(post.id)

        existing_result = await cls._strato_direct_mh.fetch(post_id)
        if existing_result:
            Metrics.counter(
                "task.write_safety_post_annotations_result_sink.existing_found.count"
            ).add(1)
        else:
            Metrics.counter(
                "task.write_safety_post_annotations_result_sink.existing_not_found.count"
            ).add(1)

        if safety_annotations.violatedPolicies:
            safety_annotations.violatedPolicies.sort(
                key=lambda x: x.score or 0, reverse=True
            )

        task_type_suffix = (
            ctx.payload.task_type.value if ctx.payload.task_type else "unknown"
        )
        identifier = f"{task_type_suffix}"
        timestamp_ms = int(time.time() * 1000)
        found_metadata = cls._build_found_metadata(post)

        new_annotation = SafetyPostAnnotations(
            tweetId=post_id,
            violatedPolicies=[
                policy.model_dump()
                for policy in (safety_annotations.violatedPolicies or [])
            ],
            foundMetadata=found_metadata,
            identifier=identifier,
            timestamp=timestamp_ms,
        )

        annotations_list = (
            list(existing_result.safetyPostAnnotations)
            if existing_result and existing_result.safetyPostAnnotations
            else []
        )
        annotations_list.append(new_annotation)

        new_bool_metadata = cls._compute_bool_metadata_from_violations(
            safety_annotations
        )
        existing_bool_metadata = (
            existing_result.safetyBoolMetadata if existing_result else None
        )
        merged_bool_metadata = cls._merge_bool_metadata(
            existing_bool_metadata, new_bool_metadata
        )

        violation_details = []
        for v in safety_annotations.violatedPolicies:
            policy_type = v.safetyPolicy.policyType.name if v.safetyPolicy else "none"
            violation_details.append(f"{v.category.value}:{policy_type}")
        violations_summary = ", ".join(violation_details)

        action_result = await cls._strato_grok_ptos_action_with_labels.execute(
            new_annotation
        )
        if action_result and len(action_result.applied_labels) > 0:
            logger.info(
                f"grokPtosActionWithLabels applied labels: debugString='{action_result.debug_string}', appliedLabels={action_result.applied_labels} for post {post_id} (result_sink), violations=[{violations_summary}]"
            )
            Metrics.counter(
                "task.write_safety_post_annotations_result_sink.grok_ptos_action_with_labels.count"
            ).add(1)
            for label in action_result.applied_labels:
                Metrics.counter(
                    "task.write_safety_post_annotations_result_sink.grok_ptos_action_with_labels.applied_label.count"
                ).add(1, attributes={"label": label})
        elif action_result:
            logger.info(
                f"grokPtosActionWithLabels did not apply any labels: (debugString='{action_result.debug_string}') for post {post_id} (result_sink), violations=[{violations_summary}] "
            )
            Metrics.counter(
                "task.write_safety_post_annotations_result_sink.grok_ptos_action_with_labels.empty.count"
            ).add(1)
        else:
            logger.info(
                f"grokPtosActionWithLabels failed for post {post_id} (result_sink), violations=[{violations_summary}] "
            )
            Metrics.counter(
                "task.write_safety_post_annotations_result_sink.grok_ptos_action_with_labels.failed.count"
            ).add(1)

        ptos_already_nsfw = new_bool_metadata.isNsfw
        if ctx.safemodel_sex_nudity.positive and not ptos_already_nsfw:
            safemodel_confidence_int = round(ctx.safemodel_sex_nudity.confidence * 100)
            safemodel_annotation = SafetyPostAnnotations(
                tweetId=post_id,
                violatedPolicies=[
                    SafetyPtosViolatedPolicy(
                        category=SafetyPolicyCategory.AdultContent,
                        score=safemodel_confidence_int,
                        reason="safemodel sex-and-nudity classifier detected adult content",
                        safetyPolicy=SafetyPolicy(
                            policyType=SafetyPolicyType.AdultContentSexualHard,
                            confidenceScore=safemodel_confidence_int,
                            reason="safemodel sex-and-nudity classifier detected adult content",
                        ),
                    ).model_dump(),
                ],
                foundMetadata=found_metadata,
                identifier="safemodel-sex-nudity",
                timestamp=timestamp_ms,
            )
            annotations_list.append(safemodel_annotation)
            merged_bool_metadata = cls._merge_bool_metadata(
                merged_bool_metadata,
                SafetyBoolMetadata(
                    isGore=False, isNsfw=True, isSoftNsfw=False, isSpam=False
                ),
            )

            Metrics.counter(
                "task.write_safety_post_annotations_result_sink.safemodel_enforced.count"
            ).add(1)
            safemodel_action_result = (
                await cls._strato_grok_ptos_action_with_labels.execute(
                    safemodel_annotation
                )
            )
            if (
                safemodel_action_result
                and len(safemodel_action_result.applied_labels) > 0
            ):
                logger.info(
                    f"safemodel enforce: grokPtosActionWithLabels applied labels: debugString='{safemodel_action_result.debug_string}', "
                    f"appliedLabels={safemodel_action_result.applied_labels} for post {post_id}"
                )
                Metrics.counter(
                    "task.write_safety_post_annotations_result_sink.safemodel_action.count"
                ).add(1)
                for label in safemodel_action_result.applied_labels:
                    Metrics.counter(
                        "task.write_safety_post_annotations_result_sink.safemodel_action.applied_label.count"
                    ).add(1, attributes={"label": label})
            elif safemodel_action_result:
                logger.info(
                    f"safemodel enforce: grokPtosActionWithLabels no labels applied (debugString='{safemodel_action_result.debug_string}') for post {post_id}"
                )
                Metrics.counter(
                    "task.write_safety_post_annotations_result_sink.safemodel_action.empty.count"
                ).add(1)
            else:
                logger.info(
                    f"safemodel enforce: grokPtosActionWithLabels returned None for post {post_id}"
                )
                Metrics.counter(
                    "task.write_safety_post_annotations_result_sink.safemodel_action.failed.count"
                ).add(1)

        final_result = SafetyPostAnnotationsResult(
            tweetId=post_id,
            safetyPostAnnotations=annotations_list,
            safetyBoolMetadata=merged_bool_metadata,
        )

        delete_labels_result = await cls._strato_grok_ptos_delete_labels.execute(
            final_result
        )
        if delete_labels_result:
            logger.info(
                f"grokPtosDeleteLabels returned '{delete_labels_result}' for post {post_id} (result_sink)"
            )
            Metrics.counter(
                "task.write_safety_post_annotations_result_sink.grok_ptos_delete_labels.count"
            ).add(1)
        else:
            logger.info(
                f"grokPtosDeleteLabels returned no result for post {post_id} (result_sink)"
            )
            Metrics.counter(
                "task.write_safety_post_annotations_result_sink.grok_ptos_delete_labels.empty.count"
            ).add(1)

        await cls._strato_mh.put(post_id, final_result)
        Metrics.counter(
            "task.write_safety_post_annotations_result_sink.mh.success.count"
        ).add(1)

        await cls._strato_kafka.insert(post_id, final_result)
        Metrics.counter(
            "task.write_safety_post_annotations_result_sink.kafka.success.count"
        ).add(1)

        Metrics.counter(
            "task.write_safety_post_annotations_result_sink.success.count"
        ).add(1)
