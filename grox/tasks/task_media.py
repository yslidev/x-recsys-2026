from grox.tasks.task import TaskWithPost
from grox.schedules.types import TaskContext
from grox.data_loaders.data_types import Post
from grox.data_loaders.media_processor import MediaProcessor
from monitor.metrics import Metrics


class TaskMediaHydration(TaskWithPost):
    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        Metrics.counter("task.media_hydration.total.count").add(1)
        await MediaProcessor.process(post, video_duration_limit_minutes=360)
        Metrics.counter("task.media_hydration.passed.count").add(1)


class TaskMediaHydrationBanger(TaskWithPost):
    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        Metrics.counter("task.media_hydration_banger.total.count").add(1)
        await MediaProcessor.process(post, video_duration_limit_minutes=360)
        Metrics.counter("task.media_hydration_banger.passed.count").add(1)
