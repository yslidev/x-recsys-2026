import logging

from grox.tasks.task import Task, TaskWithPost, TaskResultCategory
from grox.schedules.types import TaskContext
from grox.data_loaders.data_types import Post
from grox.classifiers.content.reply_ranking import ReplyScorer

logger = logging.getLogger(__name__)


class TaskRankReplies(TaskWithPost):
    scorer = ReplyScorer()

    @classmethod
    async def exec(cls, ctx: TaskContext) -> TaskResultCategory:
        return await Task.exec.__wrapped__(cls, ctx)

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        user = post.user
        logger.info(
            f"[task_rank_replies] {post.id=} "
            f"is_pasted={post.is_pasted} "
            f"user_agent={post.user_agent!r} "
            f"composition_source={post.composition_source!r} "
            f"app_attestation_status={post.app_attestation_status!r} "
            f"has_risky_user_safety_label={user.has_risky_user_safety_label if user else None} "
            f"num_legit_blocks_received_last_24hrs={user.num_legit_blocks_received_last_24hrs if user else None}"
        )
        res = await cls.scorer.score(post)
        ctx.reply_ranking_results.extend(res)
