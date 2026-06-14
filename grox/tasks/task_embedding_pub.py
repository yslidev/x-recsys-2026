import logging
import os
import time
import traceback
from functools import cache

from thrifts.gen.twitter.strato.columns.content_understanding.content_understanding.ttypes import (
    SimpleTweetEmbedding,
)
from thrifts.serdes import Serializer
from grox.tasks.task import TaskWithPost
from grox.tasks.disable_rules import DisableTaskForNonMmEmbProd
from grox.schedules.types import TaskContext
from grox.data_loaders.data_types import Post
from grox.config.config import grox_config, KafkaTopicName
from kafka_cli.producer import ScramKafkaProducer
from monitor.metrics import Metrics

logger = logging.getLogger(__name__)


class TaskPublishEmbeddingKafka(TaskWithPost):
    DISABLE_RULES = [DisableTaskForNonMmEmbProd]
    KAFKA_TOPIC_NAME: KafkaTopicName

    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        embedding = ctx.multimodal_post_embedding
        if embedding is None:
            Metrics.counter("task.publish_embedding_kafka.skipped.count").add(
                1, attributes={"reason": "no_embedding"}
            )
            logger.info(f"No embedding available for post {post.id}, skipping")
            return

        Metrics.counter("task.publish_embedding_kafka.intaken.count").add(1)
        try:
            await cls._publish_to_kafka(post, embedding)
            Metrics.counter("task.publish_embedding_kafka.success.count").add(1)
            if post.created_at:
                latency = time.time() - post.created_at.timestamp()
                Metrics.histogram("task.publish_embedding_kafka.e2e_latency").record(
                    latency
                )
        except Exception:
            Metrics.counter("task.publish_embedding_kafka.failed.count").add(1)
            logger.error(
                f"Failed to publish embedding to Kafka: {traceback.format_exc()}"
            )
            raise

    @classmethod
    async def _publish_to_kafka(cls, post: Post, embedding: list[float]) -> None:
        tweet_embedding = SimpleTweetEmbedding(
            tweetId=int(post.id),
            embedding1=embedding,
        )
        serialized_bytes = Serializer.serialize(tweet_embedding)
        await cls._get_kafka_producer().send(id=post.id, value=serialized_bytes)
        logger.info(
            f"Published embedding for post {post.id} to {cls.KAFKA_TOPIC_NAME.value}"
        )

    @classmethod
    @cache
    def _get_kafka_producer(cls) -> ScramKafkaProducer:
        producer_config = grox_config.get_kafka_producer_topic(cls.KAFKA_TOPIC_NAME)
        logger.info(
            f"Creating embedding kafka producer with config: {producer_config.model_dump()}"
        )
        return ScramKafkaProducer(producer_config)


class TaskPublishEmbeddingV4Kafka(TaskPublishEmbeddingKafka):
    KAFKA_TOPIC_NAME = KafkaTopicName.GROX_MULTIMODAL_EMBEDDING_V4


class TaskPublishEmbeddingV5Kafka(TaskPublishEmbeddingKafka):
    KAFKA_TOPIC_NAME = KafkaTopicName.GROX_MULTIMODAL_EMBEDDING_V5
