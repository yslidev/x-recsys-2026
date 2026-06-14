import logging

from grox.tasks.task import Task, TaskWithPost, TaskResultCategory
from grox.tasks.task_load_post_with_not_found_retry import TaskLoadPostWithNotFoundRetry
from monitor.metrics import Metrics
from grox.schedules.types import TaskContext, TaskPayload
from grox.data_loaders.data_types import Post
from grox.summarizer.post_embedding_summarizer import PostEmbeddingSummarizer

logger = logging.getLogger(__name__)


class TaskPostEmbeddingSummarizer(TaskWithPost):
    summarizer = PostEmbeddingSummarizer(prompt_file="")

    @classmethod
    async def exec(cls, ctx: TaskContext) -> TaskResultCategory:
        return await Task.exec.__wrapped__(cls, ctx)

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        res = await cls.summarizer.summarize(post)
        assert res is not None
        post.summary = res
        Metrics.counter("task.post_embedding_summarizer.count").add(1)
