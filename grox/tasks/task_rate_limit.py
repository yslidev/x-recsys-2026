import logging
from abc import abstractmethod
from typing import Awaitable, override

from cachetools import TTLCache
from grox.tasks.task import Task, TaskContext, TaskStopExecution
from grox.config.config import TaskGeneratorType
from monitor.metrics import Metrics
from grox.data_loaders.data_types import Post, User

logger = logging.getLogger(__name__)


class TaskRateLimit(Task):
    @classmethod
    async def _exec(cls, ctx: TaskContext) -> None:
        eligible = await cls._eligible(ctx)
        Metrics.counter("task.rate_limit.count").add(
            1, attributes={"task_name": cls.get_name(), "passed": eligible}
        )
        if not eligible:
            raise TaskStopExecution()

    @classmethod
    @abstractmethod
    def _eligible(cls, ctx: TaskContext) -> Awaitable[bool]:
        raise NotImplementedError()


class TaskRateLimitWithPost(TaskRateLimit):
    @override
    @classmethod
    async def _eligible(cls, ctx: TaskContext) -> bool:
        if not ctx.payload.post:
            return False
        return await cls._eligible_with_post(ctx.payload.post, ctx)

    @classmethod
    @abstractmethod
    def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> Awaitable[bool]:
        raise NotImplementedError()


class TaskRateLimitWithUser(TaskRateLimit):
    @override
    @classmethod
    async def _eligible(cls, ctx: TaskContext) -> bool:
        if not ctx.payload.user:
            return False
        return await cls._eligible_with_user(ctx.payload.user, ctx)

    @classmethod
    @abstractmethod
    def _eligible_with_user(cls, user: User, ctx: TaskContext) -> Awaitable[bool]:
        raise NotImplementedError()


class TaskRateLimitEmbeddingWithPostSummary(TaskRateLimitWithPost):
    POST_CACHE_FOR_MM_EMB_SUMMARY = TTLCache(maxsize=10_000, ttl=60)

    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        post_id = post.id
        if post_id not in cls.POST_CACHE_FOR_MM_EMB_SUMMARY:
            cls.POST_CACHE_FOR_MM_EMB_SUMMARY[post_id] = True
            return True
        logger.info(f"Post {post_id} already hit rate limit for mm emb with summary")
        return False


class TaskRateLimitEmbeddingWithPostSummaryForReply(TaskRateLimitWithPost):
    POST_CACHE_FOR_MM_EMB_SUMMARY_REPLY = TTLCache(maxsize=10_000, ttl=60)

    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        post_id = post.id
        if post_id not in cls.POST_CACHE_FOR_MM_EMB_SUMMARY_REPLY:
            cls.POST_CACHE_FOR_MM_EMB_SUMMARY_REPLY[post_id] = True
            return True
        logger.info(
            f"Post {post_id} already hit rate limit for mm emb with summary for reply"
        )
        return False


class TaskRateLimitEmbeddingV5(TaskRateLimitWithPost):
    POST_CACHE_FOR_MM_EMB_V5 = TTLCache(maxsize=10_000, ttl=60)

    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        post_id = post.id
        if post_id not in cls.POST_CACHE_FOR_MM_EMB_V5:
            cls.POST_CACHE_FOR_MM_EMB_V5[post_id] = True
            return True
        logger.info(f"Post {post_id} already hit rate limit for mm emb v5")
        return False


class TaskRateLimitEmbeddingV5ForReply(TaskRateLimitWithPost):
    POST_CACHE_FOR_MM_EMB_V5_REPLY = TTLCache(maxsize=10_000, ttl=60)

    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        post_id = post.id
        if post_id not in cls.POST_CACHE_FOR_MM_EMB_V5_REPLY:
            cls.POST_CACHE_FOR_MM_EMB_V5_REPLY[post_id] = True
            return True
        logger.info(f"Post {post_id} already hit rate limit for mm emb v5 for reply")
        return False


class TaskRateLimitBangerAnnotationWithPost(TaskRateLimitWithPost):
    POST_CACHE_FOR_BANGER = TTLCache(maxsize=10_000, ttl=60)

    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        post_id = post.id
        if post_id not in cls.POST_CACHE_FOR_BANGER:
            cls.POST_CACHE_FOR_BANGER[post_id] = True
            return True
        logger.info(f"Post {post_id} already hit rate limit for banger")
        return False


class TaskRateLimitReplySpamAnnotationWithPost(TaskRateLimitWithPost):
    POST_CACHE_FOR_REPLY_SPAM = TTLCache(maxsize=10_000, ttl=60)

    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        post_id = post.id
        if post_id not in cls.POST_CACHE_FOR_REPLY_SPAM:
            cls.POST_CACHE_FOR_REPLY_SPAM[post_id] = True
            return True
        logger.info(f"Post {post_id} already hit rate limit for reply spam")
        return False


class TaskRateLimitReplyRankingAnnotationWithPost(TaskRateLimitWithPost):
    POST_CACHE_FOR_REPLY_RANKING = TTLCache(maxsize=10_000, ttl=60)

    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        post_id = post.id
        if post_id not in cls.POST_CACHE_FOR_REPLY_RANKING:
            cls.POST_CACHE_FOR_REPLY_RANKING[post_id] = True
            return True
        logger.info(f"Post {post_id} already hit rate limit for reply ranking")
        return False


class TaskRateLimitPostSafetyAnnotationWithPost(TaskRateLimitWithPost):
    POST_CACHE_FOR_POST_SAFETY = TTLCache(maxsize=10_000, ttl=60)

    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        post_id = post.id
        if post_id not in cls.POST_CACHE_FOR_POST_SAFETY:
            cls.POST_CACHE_FOR_POST_SAFETY[post_id] = True
            return True
        logger.info(f"Post {post_id} already hit rate limit for post safety")
        return False


class TaskRateLimitSafetyPtosAnnotationWithPost(TaskRateLimitWithPost):
    POST_CACHE_FOR_SAFETY_PTOS = TTLCache(maxsize=10_000, ttl=60)
    POST_CACHE_FOR_SAFETY_PTOS_DELUXE = TTLCache(maxsize=10_000, ttl=60)

    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        post_id = post.id
        is_deluxe = ctx.payload.task_type == TaskGeneratorType.SAFETY_PTOS_DELUXE
        cache = (
            cls.POST_CACHE_FOR_SAFETY_PTOS_DELUXE
            if is_deluxe
            else cls.POST_CACHE_FOR_SAFETY_PTOS
        )
        label = "safety ptos deluxe" if is_deluxe else "safety ptos"
        if post_id not in cache:
            cache[post_id] = True
            return True
        logger.info(f"Post {post_id} already hit rate limit for {label}")
        return False
