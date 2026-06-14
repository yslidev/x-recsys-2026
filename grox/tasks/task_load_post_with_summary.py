from grox.tasks.task import TaskWithPost

from grox.schedules.types import TaskContext
from grox.data_loaders.data_types import Post
from grox.data_loaders.strato_loader import TweetStratoLoader
from grox.tasks.task import TaskStopExecution
from strato_http.queries.post_multimodal_embedding_mh_searchai import (
    StratoPostMultimodalEmbeddingGrokSummaryMh,
)


class TaskLoadPostWithSummary(TaskWithPost):
    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        loaded_post = await TweetStratoLoader.load_post(post.id)
        if loaded_post is None:
            raise TaskStopExecution(f"Post not found: {post.id}")
        stratoPostMultimodalEmbeddingGrokSummaryMh = (
            StratoPostMultimodalEmbeddingGrokSummaryMh()
        )
        summary = await stratoPostMultimodalEmbeddingGrokSummaryMh.fetch(
            int(post.id), "v3"
        )
        if summary is None:
            raise TaskStopExecution(f"Summary not found: {post.id}")
        loaded_post.summary = summary
        ctx.payload.post = loaded_post
