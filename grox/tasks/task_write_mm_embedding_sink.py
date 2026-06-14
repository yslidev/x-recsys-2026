import logging
import time

from grox.tasks.task import Task, TaskWithPost, TaskResultCategory
from grox.tasks.disable_rules import DisableTaskForNonMmEmbProd
from monitor.metrics import Metrics
from grox.schedules.types import TaskContext
from grox.data_loaders.data_types import Post
from strato_http.queries.post_multimodal_embedding_sink import (
    StratoPostMultimodalEmbeddingSink,
)
from tenacity import retry, wait_chain, wait_fixed, stop_after_attempt
from strato_http.queries.post_multimodal_embedding_mh_searchai import (
    StratoPostMultimodalEmbeddingMhSearchAi,
    TweetEmbedding,
    StratoPostMultimodalEmbeddingMhSearchAiNoCache,
    StratoPostMultimodalEmbeddingGrokSummaryMh,
    StratoMultiModalEmbeddingTopic,
)
from grox.tasks.task_embedding_pub import (
    TaskPublishEmbeddingV4Kafka,
    TaskPublishEmbeddingV5Kafka,
)

logger = logging.getLogger(__name__)


class TaskWriteMMEmbeddingSinkBase(TaskWithPost):
    model_version: str

    DISABLE_RULES = [DisableTaskForNonMmEmbProd]

    @classmethod
    @retry(stop=stop_after_attempt(3), wait=wait_chain(wait_fixed(1), wait_fixed(2)))
    async def exec(cls, ctx: TaskContext) -> TaskResultCategory:
        return await Task.exec.__wrapped__(cls, ctx)


class TaskWriteMMEmbeddingSinkExperiment(TaskWriteMMEmbeddingSinkBase):
    model_version = "v2"

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        start_time = time.perf_counter_ns()
        embedding = ctx.multimodal_post_embedding_dict[cls.model_version]
        assert embedding is not None
        query = StratoPostMultimodalEmbeddingMhSearchAi()
        await query.put(
            int(post.id),
            cls.model_version,
            TweetEmbedding(tweetId=int(post.id), embedding1=embedding),
        )
        logger.info(
            f"wrote post embedding to strato sink for post {post.id} (model: {cls.model_version})"
        )
        duration_ms = (time.perf_counter_ns() - start_time) / 1_000
        Metrics.histogram(
            "task.write_post_embedding_sink_experiment.duration_ms"
        ).record(duration_ms)
        Metrics.counter("task.write_post_embedding_sink_experiment.count").add(1)


class TaskWriteMMEmbeddingSinkV3(TaskWriteMMEmbeddingSinkBase):
    model_version = "v3"

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        start_time = time.perf_counter_ns()

        summary = post.summary
        assert summary is not None
        stratoPostMultimodalEmbeddingGrokSummaryMh = (
            StratoPostMultimodalEmbeddingGrokSummaryMh()
        )
        await stratoPostMultimodalEmbeddingGrokSummaryMh.put(
            int(post.id), cls.model_version, summary
        )

        embedding = ctx.multimodal_post_embedding_dict[cls.model_version]
        assert embedding is not None
        query = StratoPostMultimodalEmbeddingMhSearchAi()
        await query.put(
            int(post.id),
            cls.model_version,
            TweetEmbedding(tweetId=int(post.id), embedding1=embedding),
        )

        stratoMultiModalEmbeddingTopic = StratoMultiModalEmbeddingTopic()
        await stratoMultiModalEmbeddingTopic.insert(
            TweetEmbedding(tweetId=int(post.id), embedding1=embedding)
        )

        logger.info(
            f"wrote post embedding to strato sink for post {post.id} (model: {cls.model_version})"
        )
        duration_ms = (time.perf_counter_ns() - start_time) / 1_000
        Metrics.histogram("task.write_post_embedding_sink_v3.duration_ms").record(
            duration_ms
        )
        Metrics.counter("task.write_post_embedding_sink_v3.count").add(1)


class TaskWriteMMEmbeddingSinkV4(TaskWriteMMEmbeddingSinkBase):
    model_version = "v4"

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        start_time = time.perf_counter_ns()

        embedding = ctx.multimodal_post_embedding_dict[cls.model_version]
        assert embedding is not None
        query = StratoPostMultimodalEmbeddingMhSearchAiNoCache()
        await query.put(
            int(post.id),
            cls.model_version,
            TweetEmbedding(tweetId=int(post.id), embedding1=embedding),
        )

        await TaskPublishEmbeddingV4Kafka._publish_to_kafka(post, embedding)

        logger.info(
            f"wrote post embedding to strato sink for post {post.id} (model: {cls.model_version})"
        )
        duration_ms = (time.perf_counter_ns() - start_time) / 1_000
        Metrics.histogram("task.write_post_embedding_sink_v4.duration_ms").record(
            duration_ms
        )
        Metrics.counter("task.write_post_embedding_sink_v4.count").add(1)


class TaskWriteMMEmbeddingSinkV5(TaskWriteMMEmbeddingSinkBase):
    model_version = "v5_1"

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        start_time = time.perf_counter_ns()

        embedding = ctx.multimodal_post_embedding_dict[cls.model_version]
        assert embedding is not None
        query = StratoPostMultimodalEmbeddingMhSearchAiNoCache()
        await query.put(
            int(post.id),
            cls.model_version,
            TweetEmbedding(tweetId=int(post.id), embedding1=embedding),
        )

        await TaskPublishEmbeddingV5Kafka._publish_to_kafka(post, embedding)

        logger.info(
            f"wrote post embedding to strato sink for post {post.id} (model: {cls.model_version})"
        )
        duration_ms = (time.perf_counter_ns() - start_time) / 1_000
        Metrics.histogram("task.write_post_embedding_sink_v5.duration_ms").record(
            duration_ms
        )
        Metrics.counter("task.write_post_embedding_sink_v5.count").add(1)


class TaskWriteMMEmbeddingSinkV5SkipKafkaForReplies(TaskWriteMMEmbeddingSinkBase):
    model_version = "v5_1"

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        start_time = time.perf_counter_ns()

        embedding = ctx.multimodal_post_embedding_dict[cls.model_version]
        assert embedding is not None
        query = StratoPostMultimodalEmbeddingMhSearchAiNoCache()
        await query.put(
            int(post.id),
            cls.model_version,
            TweetEmbedding(tweetId=int(post.id), embedding1=embedding),
        )

        is_reply = bool(post.ancestors)
        if not is_reply:
            await TaskPublishEmbeddingV5Kafka._publish_to_kafka(post, embedding)
        else:
            Metrics.counter(
                "task.write_post_embedding_sink_v5.kafka_skipped_reply.count"
            ).add(1)
            logger.info(
                f"Skipping Kafka publish for reply post {post.id} (written to Manhattan only)"
            )

        logger.info(
            f"wrote post embedding to strato sink for post {post.id} (model: {cls.model_version}, kafka={'yes' if not is_reply else 'no'})"
        )
        duration_ms = (time.perf_counter_ns() - start_time) / 1_000
        Metrics.histogram("task.write_post_embedding_sink_v5.duration_ms").record(
            duration_ms
        )
        Metrics.counter("task.write_post_embedding_sink_v5.count").add(1)
