import logging
import time
from typing import override

from grox.tasks.task import Task, TaskWithPost, TaskResultCategory
from monitor.metrics import Metrics
from grox.schedules.types import TaskContext
from grox.data_loaders.data_types import Post, ContentCategoryType
from grox.classifiers.content.banger_initial_screen import BangerInitialScreenClassifier
from strato_http.queries.grok_topics import StratoGrokTopics

logger = logging.getLogger(__name__)

LOG_EVERY_N = 10000
CACHE_TTL_SECONDS = 3600


class TaskBangerScreen(TaskWithPost):
    classifier = BangerInitialScreenClassifier()
    _cached_topics = None
    _cache_timestamp = None

    @classmethod
    async def exec(cls, ctx: TaskContext) -> TaskResultCategory:
        return await Task.exec.__wrapped__(cls, ctx)

    @override
    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        Metrics.counter("task.banger_initial_screen.total.count").add(1)

        if cls._cached_topics is None or (
            cls._cache_timestamp is not None
            and time.time() - cls._cache_timestamp > CACHE_TTL_SECONDS
        ):
            logger.info("Fetching grok topics for cache")
            Metrics.counter("task.banger_initial_screen.grok_cache.new_load.count").add(
                1
            )
            query = StratoGrokTopics()
            fetched_topics = await query.fetch()
            if fetched_topics:
                cls._cached_topics = fetched_topics
                cls._cache_timestamp = time.time()
                logger.info(f"Cached {len(cls._cached_topics)} categories with topics")
                Metrics.counter(
                    "task.banger_initial_screen.grok_cache.new_load.success.count"
                ).add(1)
            else:
                logger.warning("Failed to fetch grok topics")
                Metrics.counter(
                    "task.banger_initial_screen.grok_cache.new_load.failure.count"
                ).add(1)
        else:
            Metrics.counter(
                "task.banger_initial_screen.grok_cache.cache_hit.count"
            ).add(1)

        if cls._cached_topics and len(cls._cached_topics) > 0:
            Metrics.counter("task.banger_initial_screen.with_cached_topics.count").add(
                1
            )
        else:
            Metrics.counter(
                "task.banger_initial_screen.without_cached_topics.count"
            ).add(1)

        res = await cls.classifier.classify(post, topics=cls._cached_topics)
        ctx.content_categories.extend(res)
        ctx.available_topics = cls._cached_topics
        passed = any(
            r.positive
            for r in res
            if r.category == ContentCategoryType.BANGER_INITIAL_SCREEN
        )
        if passed:
            Metrics.counter("task.banger_initial_screen.passed.count").add(1)
        else:
            Metrics.counter("task.banger_initial_screen.failed.count").add(1)
