import time
import asyncio
import logging
import traceback
from abc import ABC
from functools import cache

from grox.lib.utils import camel_to_snake
from grox.tasks.task import Task, TaskResultCategory
from monitor.metrics import Metrics
from grox.schedules.types import TaskResult, TaskContext, TaskPayload, TaskEligibility

logger = logging.getLogger(__name__)


class Plan(ABC):
    TASKS: dict[str, type[Task]] = {}
    TASK_DEPENDENCIES: dict[str, set[str]] = {}
    REQUIRED_ELIGIBILITY: TaskEligibility

    def __init__(self):
        self.deps = set([d for deps in self.TASK_DEPENDENCIES.values() for d in deps])
        if any(t not in self.TASKS for t in self.deps) or any(
            t not in self.TASKS for t in self.TASK_DEPENDENCIES.keys()
        ):
            raise ValueError("Not every task in TASK_DEPENDENCIES is defined in TASKS")

    async def execute(self, task: TaskPayload) -> TaskResult | None:
        if not self._eligible(task):
            return None
        Metrics.counter("plan.execute.count").add(
            1, attributes={"plan_name": self.get_name()}
        )
        logger.debug(f"Creating execution plan for graph: {self.TASK_DEPENDENCIES}")
        loop = asyncio.get_running_loop()
        dependencies = {task: loop.create_future() for task in self.deps}
        start = time.perf_counter()
        ctx = TaskContext(task)
        try:
            await asyncio.gather(
                *[self._execute_task(t, ctx, dependencies) for t in self.TASKS.keys()]
            )
            Metrics.counter("plan.execute.success.count").add(
                1, attributes={"plan_name": self.get_name()}
            )
        except Exception as e:
            logger.error(f"Error executing plan: {traceback.format_exc()}")
            ctx.errors.append(e)
            Metrics.counter("plan.execute.failed.count").add(
                1, attributes={"plan_name": self.get_name()}
            )
        finally:
            duration = time.perf_counter() - start
            Metrics.histogram("plan.execute.duration").record(
                duration, attributes={"plan_name": self.get_name()}
            )
            for fut in dependencies.values():
                try:
                    if not fut.done():
                        fut.cancel()
                except Exception:
                    logger.error(
                        f"Error canceling dependency future: {traceback.format_exc()}"
                    )
            dependencies.clear()
        return TaskResult(
            task=task,
            content_categories=[c.model_copy() for c in ctx.content_categories],
            task_started_at=ctx.start_time,
            task_finished_at=time.perf_counter(),
            multimodal_post_embedding=ctx.multimodal_post_embedding,
            reason=ctx.reason,
            success=len(ctx.errors) == 0,
            error="\n".join([str(e) for e in ctx.errors]),
        )

    def _eligible(self, ctx: TaskPayload) -> bool:
        return self.REQUIRED_ELIGIBILITY in ctx.eligibilities

    async def _execute_task(
        self, task_name: str, ctx: TaskContext, dependencies: dict[str, asyncio.Future]
    ):
        logger.debug(f"Waiting for task to become ready: {task_name}")
        task = self.TASKS[task_name]
        deps = self.TASK_DEPENDENCIES.get(task_name, set())
        dep_futures = [dependencies[d] for d in deps]
        dep_results = await asyncio.gather(*dep_futures)
        task_future = dependencies.get(task_name, None)
        if any(r == TaskResultCategory.SKIPPED for r in dep_results):
            if task_future is not None:
                task_future.set_result(TaskResultCategory.SKIPPED)
            return
        logger.debug(f"Started executing task: {task_name}")
        try:
            res = await task.exec(ctx)
        except Exception as e:
            if task_future is not None:
                task_future.set_exception(e)
            raise e
        if task_future is not None:
            task_future.set_result(res)
        logger.debug(f"Finished executing task: {task_name}")

    @cache
    def get_name(self) -> str:
        return camel_to_snake(self.__class__.__name__)
