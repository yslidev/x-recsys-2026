import logging

from grox.tasks.task import Task, TaskWithPost, TaskResultCategory
from monitor.metrics import Metrics
from grox.schedules.types import TaskContext
from grox.data_loaders.data_types import Post
from grox.classifiers.content.post_safety_screen_deluxe import (
    PostSafetyDeluxeClassifier,
)

logger = logging.getLogger(__name__)


class TaskPostSafetyScreenDeluxe(TaskWithPost):
    classifier = PostSafetyDeluxeClassifier()

    @classmethod
    async def exec(cls, ctx: TaskContext) -> TaskResultCategory:
        return await Task.exec.__wrapped__(cls, ctx)

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        Metrics.counter("task.post_safety_screen_deluxe.total.count").add(1)
        res = await cls.classifier.classify(post)
        ctx.content_categories.extend(res)
