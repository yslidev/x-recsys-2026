import logging
from typing import override

from grox.tasks.task import TaskWithPost
from monitor.metrics import Metrics
from grox.schedules.types import TaskContext, TaskEligibility
from grox.data_loaders.data_types import Post, Video
from grox.embedder.multimodal_post_embedder_v2 import MultimodalPostEmbedderV2
from grox.embedder.multimodal_post_embedder_v5 import MultimodalPostEmbedderV5

logger = logging.getLogger(__name__)


class TaskMultimodalPostEmbeddingWithSummary(TaskWithPost):
    embedder = MultimodalPostEmbedderV2(
        model="qwen3", renderer_version="lite", use_post_context_summary=True
    )

    @override
    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        _, embedding = await cls.embedder.embed(post)
        ctx.multimodal_post_embedding_dict["v3"] = embedding
        logger.info(
            f"TaskMultimodalPostEmbeddingWithSummary Embedding Added, length: {len(embedding)}"
        )
        Metrics.counter("task.multimodal_post_embedding_with_summary.count").add(1)


class TaskMultimodalPostEmbeddingRecsysV4(TaskWithPost):
    embedder = MultimodalPostEmbedderV2(
        model="v4",
        renderer_version="lite",
        use_post_context_summary=True,
    )

    @override
    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        try:
            _, embedding = await cls.embedder.embed(post)
        except Exception as e:
            Metrics.counter("task.multimodal_post_embedding_recsys_v4.error").add(1)
            logger.warning(
                f"TaskMultimodalPostEmbeddingRecsysV4 failed for post {post.id}: {e}"
            )
            return
        ctx.multimodal_post_embedding_dict["v4"] = embedding
        logger.info(
            f"TaskMultimodalPostEmbeddingRecsysV4 Embedding Added, length: {len(embedding)}"
        )
        Metrics.counter("task.multimodal_post_embedding_recsys_v4.count").add(1)


class TaskMultimodalPostEmbeddingV5(TaskWithPost):
    embedder = MultimodalPostEmbedderV5()

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        try:
            transcripts = []
            if post.media:
                for m in post.media:
                    if (
                        isinstance(m, Video)
                        and m.convo_video
                        and m.convo_video.asr_transcript
                    ):
                        transcripts.append(m.convo_video.asr_transcript)
            transcript = "\n".join(transcripts) if transcripts else None
            _, embedding = await cls.embedder.embed(post, transcript=transcript)
        except Exception as e:
            Metrics.counter("task.multimodal_post_embedding_v5.error").add(1)
            logger.warning(
                f"TaskMultimodalPostEmbeddingV5 failed for post {post.id}: {e}"
            )
            raise
        ctx.multimodal_post_embedding_dict["v5_1"] = embedding
        logger.info(
            f"TaskMultimodalPostEmbeddingV5 Embedding Added, length: {len(embedding)}, has_transcript={transcript is not None}"
        )
        Metrics.counter("task.multimodal_post_embedding_v5.count").add(1)
        if transcript:
            Metrics.counter(
                "task.multimodal_post_embedding_v5.with_transcript.count"
            ).add(1)
