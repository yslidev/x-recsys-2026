import logging

from grox.data_loaders.data_types import Post
from grox.data_loaders.strato_loader import UserRecentPostsLoader
from grox.schedules.types import TaskContext
from grox.tasks.task import TaskWithPost, TaskStopExecution

logger = logging.getLogger(__name__)


class TaskLoadUserRecentPosts(TaskWithPost):
    RECENT_POSTS_LIMIT = ""

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        if not post.user or not post.user.id:
            raise TaskStopExecution("Post has no author user to load recent posts for")

        recent_posts = await UserRecentPostsLoader.load(
            post.user.id, limit=cls.RECENT_POSTS_LIMIT
        )
        post.user.recent_posts = recent_posts
        logger.info(f"Loaded {len(recent_posts)} recent posts for user {post.user.id}")
