import asyncio

from grox.plans.plan import Plan
from grox.schedules.types import TaskResult, TaskPayload
from grox.plans.plan_spam_comment import PlanSpamComment
from grox.plans.plan_initial_banger import PlanInitialBanger
from grox.plans.plan_post_embedding_with_summary import PlanPostEmbeddingWithSummary
from grox.plans.plan_post_embedding_v5 import PlanPostEmbeddingV5
from grox.plans.plan_post_embedding_v5_for_reply import PlanPostEmbeddingV5ForReply
from grox.plans.plan_post_embedding_with_summary_for_reply import (
    PlanPostEmbeddingWithSummaryForReply,
)
from grox.plans.plan_post_safety import PlanPostSafety
from grox.plans.plan_reply_ranking import PlanReplyRanking
from grox.plans.plan_safety_ptos import PlanSafetyPtos


class PlanMaster:
    ALL_PLANS: list[Plan] = [
        PlanInitialBanger(),
        PlanPostSafety(),
        PlanSpamComment(),
        PlanPostEmbeddingWithSummary(),
        PlanPostEmbeddingWithSummaryForReply(),
        PlanPostEmbeddingV5(),
        PlanPostEmbeddingV5ForReply(),
        PlanReplyRanking(),
        PlanSafetyPtos(),
    ]

    @classmethod
    async def exec(cls, task: TaskPayload) -> TaskResult:
        results = await asyncio.gather(*[p.execute(task) for p in cls.ALL_PLANS])
        result = cls.merge_results(task, [r for r in results if r is not None])
        return result

    @classmethod
    def merge_results(cls, task: TaskPayload, results: list[TaskResult]) -> TaskResult:
        multimodal_post_embedding = [
            r.multimodal_post_embedding
            for r in results
            if r.multimodal_post_embedding is not None
        ]
        if multimodal_post_embedding:
            multimodal_post_embedding = multimodal_post_embedding[0]
        else:
            multimodal_post_embedding = None

        return TaskResult(
            task=task,
            content_categories=[
                c.model_copy() for r in results for c in r.content_categories
            ],
            task_started_at=min(r.task_started_at for r in results),
            task_finished_at=max(r.task_finished_at for r in results),
            multimodal_post_embedding=multimodal_post_embedding,
            reason="\n".join([r.reason for r in results if r.reason]),
            success=all(r.success for r in results),
            error="\n".join(
                [r.error or "unknown error" for r in results if not r.success]
            ),
        )
