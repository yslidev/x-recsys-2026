import logging

from grox.tasks.task import Task
from grox.tasks.disable_rules import DisableTaskForNonProd
from monitor.metrics import Metrics
from grox.schedules.types import TaskContext
from strato_http.queries.grok_upa_action_with_labels import (
    StratoGrokUpaActionWithLabels,
)

logger = logging.getLogger(__name__)


class TaskGrokUpaActionWithLabels(Task):
    DISABLE_RULES = [DisableTaskForNonProd]

    _strato_grok_upa_action_with_labels = StratoGrokUpaActionWithLabels()

    @classmethod
    async def _exec(cls, ctx: TaskContext) -> None:
        Metrics.counter("task.grok_upa_action_with_labels.count").add(1)

        post = ctx.payload.post
        if not post:
            return

        results = ctx.content_categories
        if not results:
            return

        grok_response = next((r for r in results if r.tweet_bool_metadata), None)
        if not grok_response or not grok_response.tweet_bool_metadata:
            return

        tweet_id = int(post.id)
        tweet_bool_metadata = grok_response.tweet_bool_metadata.model_dump()

        action_result = await cls._strato_grok_upa_action_with_labels.execute(
            tweet_id, tweet_bool_metadata
        )
        if action_result and len(action_result.applied_labels) > 0:
            logger.info(
                f"grokUpaActionWithLabels applied labels: debugString='{action_result.debug_string}', appliedLabels={action_result.applied_labels} for post {tweet_id}"
            )
            Metrics.counter("task.grok_upa_action_with_labels.applied.count").add(1)
            for label in action_result.applied_labels:
                Metrics.counter(
                    "task.grok_upa_action_with_labels.applied_label.count"
                ).add(1, attributes={"label": label})
        elif action_result:
            logger.info(
                f"grokUpaActionWithLabels no labels applied: debugString='{action_result.debug_string}' for post {tweet_id}"
            )
            Metrics.counter("task.grok_upa_action_with_labels.empty.count").add(1)
        else:
            logger.info(f"grokUpaActionWithLabels failed for post {tweet_id}")
            Metrics.counter("task.grok_upa_action_with_labels.failed.count").add(1)
