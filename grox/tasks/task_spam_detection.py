import logging
from typing import override

from grox.tasks.task import Task, TaskWithPost, TaskResultCategory
from monitor.metrics import Metrics
from grox.schedules.types import TaskContext
from grox.data_loaders.data_types import Post, ContentCategoryType
from grox.classifiers.content.spam import SpamEapiLowFollowerClassifier

logger = logging.getLogger(__name__)


class TaskSpamDetection(TaskWithPost):
    eapi_low_follower_classifier = SpamEapiLowFollowerClassifier()

    @classmethod
    def get_follower_bucket_string(cls, post: Post) -> str:
        if not post.ancestors:
            return "invalid"
        in_reply_user_follower_count = post.ancestors[-1].user.follower_count or 0
        root_user_follower_count = post.ancestors[0].user.follower_count or 0
        if in_reply_user_follower_count <= 100 and root_user_follower_count <= 100:
            return "lte_100"
        elif in_reply_user_follower_count <= 500 and root_user_follower_count <= 500:
            return "lte_500"
        elif in_reply_user_follower_count <= 1000 and root_user_follower_count <= 1000:
            return "lte_1000"
        else:
            return "gt_1000"

    @classmethod
    async def exec(cls, ctx: TaskContext) -> TaskResultCategory:
        return await Task.exec.__wrapped__(cls, ctx)

    @override
    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        res = await cls.eapi_low_follower_classifier.classify(post)
        ctx.content_categories.extend(res)
        passed = any(
            r.positive for r in res if r.category == ContentCategoryType.SPAM_COMMENT
        )

        follower_bucket_string = cls.get_follower_bucket_string(post)
        if passed and follower_bucket_string != "gt_1000":
            logger.info(
                f"Reply Spam Found for lower than 1000 follower bucket. The post_id is {post.id} and the follower bucket is {follower_bucket_string}"
            )

        if passed:
            Metrics.counter("task.spam_comment_detection.positive.count").add(
                1, attributes={"reason": follower_bucket_string}
            )
        else:
            Metrics.counter("task.spam_comment_detection.negative.count").add(
                1, attributes={"reason": follower_bucket_string}
            )
