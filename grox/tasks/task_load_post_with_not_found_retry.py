import time
from grox.tasks.task import Task, TaskWithPost, TaskResultCategory

from grox.schedules.types import TaskContext
from grox.data_loaders.data_types import Post
from grox.data_loaders.strato_loader import TweetStratoLoader
from grox.schedules.types import TaskPayload
from grox.tasks.task import TaskStopExecution
from monitor.metrics import Metrics
from tenacity import retry, wait_chain, wait_fixed, stop_after_attempt


class TaskLoadPostWithNotFoundRetry(TaskWithPost):
    @classmethod
    @retry(stop=stop_after_attempt(3), wait=wait_chain(wait_fixed(1), wait_fixed(1)))
    async def exec(cls, ctx: TaskContext) -> TaskResultCategory:
        return await Task.exec.__wrapped__(cls, ctx)

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        start_time = time.perf_counter_ns()
        loaded_post = await TweetStratoLoader.load_post(post.id)
        if loaded_post is None:
            task_type = (
                ctx.payload.task_type.value
                if ctx.payload and ctx.payload.task_type
                else "None"
            )
            if "recovery" in task_type:
                raise TaskStopExecution(f"Post not found: {post.id}")
            else:
                raise ValueError(f"Post not found: {post.id}")
        ctx.payload.post = loaded_post
        duration_ms = (time.perf_counter_ns() - start_time) / 1_000
        Metrics.histogram("task.embedding_load_post.duration_ms").record(duration_ms)
