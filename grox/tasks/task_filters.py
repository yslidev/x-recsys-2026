import logging
from abc import abstractmethod
from typing import override
from datetime import datetime, timezone

from grox.config.config import grox_config, TaskGeneratorType
from grox.tasks.task import Task, TaskStopExecution
from monitor.metrics import Metrics
from grox.schedules.types import TaskContext
from grox.data_loaders.data_types import Post, User

logger = logging.getLogger(__name__)


class TaskFilter(Task):
    @classmethod
    async def _exec(cls, ctx: TaskContext) -> None:
        if not await cls._eligible(ctx):
            raise TaskStopExecution()

    @classmethod
    @abstractmethod
    async def _eligible(cls, ctx: TaskContext) -> bool:
        raise NotImplementedError()


class TaskFilterWithUser(TaskFilter):
    @override
    @classmethod
    async def _eligible(cls, ctx: TaskContext) -> bool:
        if not ctx.payload.user:
            return False
        return await cls._eligible_with_user(ctx.payload.user, ctx)

    @classmethod
    @abstractmethod
    async def _eligible_with_user(cls, user: User, ctx: TaskContext) -> bool:
        raise NotImplementedError()


class TaskFilterWithPost(TaskFilter):
    @override
    @classmethod
    async def _eligible(cls, ctx: TaskContext) -> bool:
        if not ctx.payload.post:
            return False
        return await cls._eligible_with_post(ctx.payload.post, ctx)

    @classmethod
    @abstractmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        raise NotImplementedError()


class TaskSpamFilter(TaskFilterWithPost):
    FOLLOWER_COUNT_THRESHOLD_FOR_SPAM_DETECTION = ""

    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        if not post.ancestors:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "spam_detection", "reason": "not_reply"}
            )
            return False
        if not post.user:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "spam_detection", "reason": "no_user"}
            )
            return False
        if post.user.id == 0:
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={"filter": "spam_detection", "reason": "is_system_account"},
            )
            return False
        if post.user.id == 0:
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={"filter": "spam_detection", "reason": "is_system_account"},
            )
            return False
        if not post.ancestors[-1].user:
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={
                    "filter": "spam_detection",
                    "reason": "previous_post_no_user",
                },
            )
            return False
        if post.user.id == post.ancestors[-1].user.id:
            logger.info(
                f"Skipping reply spam since the replier is same as reply target post {post.id}"
            )
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "spam_detection", "reason": "same_user_reply"}
            )
            return False
        if not post.ancestors[0].user:
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={"filter": "spam_detection", "reason": "root_post_no_user"},
            )
            return False
        if post.user.id == post.ancestors[0].user.id:
            logger.info(
                f"Skipping reply spam since the replier is same as reply root post {post.id}"
            )
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={
                    "filter": "spam_detection",
                    "reason": "same_user_reply_as_root",
                },
            )
            return False
        in_reply_user_follower_count = post.ancestors[-1].user.follower_count or 0
        root_user_follower_count = post.ancestors[0].user.follower_count or 0
        if (
            in_reply_user_follower_count
            > cls.FOLLOWER_COUNT_THRESHOLD_FOR_SPAM_DETECTION
            or root_user_follower_count
            > cls.FOLLOWER_COUNT_THRESHOLD_FOR_SPAM_DETECTION
        ):
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={
                    "filter": "spam_detection",
                    "reason": "reply_ranking_target",
                },
            )
            return False
        return True


class TaskReplyRankingFilter(TaskFilterWithPost):
    FOLLOWER_COUNT_THRESHOLD_FOR_REPLY_RANKING = ""

    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        if not post.ancestors:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "reply_ranking", "reason": "not_reply"}
            )
            return False
        if not post.user:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "reply_ranking", "reason": "no_user"}
            )
            return False
        if not post.ancestors[-1].user:
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={
                    "filter": "reply_ranking",
                    "reason": "previous_post_no_user",
                },
            )
            return False
        if not post.ancestors[0].user:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "reply_ranking", "reason": "root_post_no_user"}
            )
            return False
        in_reply_user_follower_count = post.ancestors[-1].user.follower_count or 0
        root_user_follower_count = post.ancestors[0].user.follower_count or 0
        if (
            in_reply_user_follower_count
            <= cls.FOLLOWER_COUNT_THRESHOLD_FOR_REPLY_RANKING
            and root_user_follower_count
            <= cls.FOLLOWER_COUNT_THRESHOLD_FOR_REPLY_RANKING
        ):
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "reply_ranking", "reason": "low_blast_radius"}
            )
            return False
        if post.user.id == post.ancestors[-1].user.id:
            logger.info(
                f"Skipping reply ranking since the replier is same as reply target post {post.id}"
            )
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "reply_ranking", "reason": "same_user_reply"}
            )
            return False
        if post.user.id == post.ancestors[0].user.id:
            logger.info(
                f"Skipping reply ranking since the replier is same as reply root post {post.id}"
            )
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={
                    "filter": "reply_ranking",
                    "reason": "same_user_reply_as_root",
                },
            )
            return False

        Metrics.counter("task.reply_ranking.eligible.count").add(1)
        return True


class TaskPostEmbeddingWithSummaryFilter(TaskFilterWithPost):
    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        if post.ancestors:
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={
                    "filter": "post_embedding_with_summary",
                    "reason": "is_reply",
                },
            )
            return False
        if not post.user:
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={
                    "filter": "post_embedding_with_summary",
                    "reason": "no_user",
                },
            )
            return False
        if post.user.is_protected:
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={
                    "filter": "post_embedding_with_summary",
                    "reason": "private_account",
                },
            )
            return False
        Metrics.counter("task.post_embedding_with_summary.eligible.count").add(1)
        return True


class TaskPostEmbeddingWithSummaryForReplyFilter(TaskFilterWithPost):
    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        if not post.ancestors:
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={
                    "filter": "post_embedding_with_summary_for_reply",
                    "reason": "is_original",
                },
            )
            return False
        if not post.user:
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={
                    "filter": "post_embedding_with_summary_for_reply",
                    "reason": "no_user",
                },
            )
            return False
        in_reply_user_protected = post.ancestors[-1].user.is_protected
        root_user_protected = post.ancestors[0].user.is_protected
        if in_reply_user_protected or root_user_protected or post.user.is_protected:
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={
                    "filter": "post_embedding_with_summary_for_reply",
                    "reason": "private_account",
                },
            )
            return False
        Metrics.counter(
            "task.post_embedding_with_summary_for_reply.eligible.count"
        ).add(1)
        return True


class TaskSafetyPtosFilter(TaskFilterWithPost):
    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        is_deluxe = ctx.payload.task_type == TaskGeneratorType.SAFETY_PTOS_DELUXE
        filter_name = (
            "safety_ptos_deluxe_detection" if is_deluxe else "safety_ptos_detection"
        )

        Metrics.counter(f"task.{filter_name}.request.count").add(1)
        if not post.user:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": filter_name, "reason": "no_user"}
            )
            return False

        Metrics.counter(f"task.{filter_name}.eligible.count").add(1)
        return True


class TaskPostSafetyDeluxeFilter(TaskFilterWithPost):
    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        if not post.user:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "post_safety_deluxe", "reason": "no_user"}
            )
            return False

        if post.ancestors:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "post_safety_deluxe", "reason": "reply"}
            )
            logger.info(f"Skipping post {post.id} because it is a reply")
            return False

        filter_reason = cls._get_hardcoded_filter_reason(post)
        if filter_reason:
            logger.info(
                f"Skipping upa deluxe {post.id} because it is hit by hardcoded filters, reason: {filter_reason}"
            )
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "post_safety_deluxe", "reason": filter_reason}
            )
            return False

        Metrics.counter("task.post_safety_deluxe.eligible.count").add(1)
        return True

    @classmethod
    def _get_hardcoded_filter_reason(cls, post: Post) -> str | None:
        if not post.user:
            return None
        if post.user.is_protected:
            return "private_account"
        return None


class TaskInitialBangerFilter(TaskFilterWithPost):
    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        Metrics.counter("task.initial_banger_filter.count").add(1)
        if not post.user:
            return False
        if post.ancestors:
            Metrics.counter("task.filter.skipped.count").add(
                1, attributes={"filter": "content_understanding", "reason": "reply"}
            )
            logger.info(f"Skipping post {post.id} because it is a reply")
            return False
        filter_reason = cls._get_hardcoded_filter_reason(post)
        if filter_reason:
            logger.info(
                f"Skipping post {post.id} because it is hit by hardcoded filters, reason: {filter_reason}"
            )
            Metrics.counter("task.filter.skipped.count").add(
                1,
                attributes={"filter": "content_understanding", "reason": filter_reason},
            )
            return False
        logger.info(f"Post {post.id} is eligible for initial banger")
        Metrics.counter("task.initial_banger_filter.eligible.count").add(1)
        return True

    @classmethod
    def _get_hardcoded_filter_reason(cls, post: Post) -> str | None:
        if not post.user:
            return None
        if post.user.is_protected:
            return "private_account"
        return None
