import abc
import logging
from collections.abc import AsyncGenerator

from grox.config.config import KafkaTopicName, TaskGeneratorType
from grox.schedules.types import TaskResult, TaskPayload, TaskEligibility
from grox.data_loaders.kafka_loader import (
    KafkaAdPostLoader,
    KafkaPostLoader,
    KafkaPostEmbeddingRequestLoader,
)
from grox.generators.task_generator import TaskGenerator
from grox.data_loaders.message_queue_loader import MessageQueueLoader

logger = logging.getLogger(__name__)


class StreamTaskGenerator(TaskGenerator, metaclass=abc.ABCMeta):
    ELIGIBILITIES_TO_INJECT: set[TaskEligibility]

    def __init__(self, max_qps: int | None):
        super().__init__(max_qps)
        self._loader = self._get_loader()

    @abc.abstractmethod
    def _get_loader(self) -> MessageQueueLoader:
        pass

    async def start(self) -> None:
        logger.info("Starting StreamTaskGenerator")
        await self._loader.start()

    async def stop(self) -> None:
        logger.info("Stopping StreamTaskGenerator")
        await super().stop()
        await self._loader.stop()

    async def _poll(self) -> AsyncGenerator[TaskPayload | None, None]:
        async for payload in self._loader.poll():
            if not payload:
                yield None
                continue
            yield TaskPayload(
                payload_id=payload.mid,
                post=payload.post,
                user=payload.user,
                user_context=payload.user_context,
                deadline_ts_secs=payload.deadline_ts_secs,
                task_type=self.TASK_GENERATOR_TYPE,
                eligibilities=self.ELIGIBILITIES_TO_INJECT.copy(),
                grox_content_analysis=payload.grox_content_analysis,
            )

    async def ack(self, result: TaskResult):
        await self._loader.ack(result.task.payload_id, result.success)


class PostStreamTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = TaskGeneratorType.POST_STREAM
    ELIGIBILITIES_TO_INJECT = {
        TaskEligibility.SPAM_COMMENT,
        TaskEligibility.REPLY_RANKING,
    }

    def _get_loader(self):
        return KafkaPostLoader(
            KafkaTopicName.CONTENT_UNDERSTANDING_REALTIME_UNIFIED_POSTS
        )


class MinTractionPostStreamForGroxTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = TaskGeneratorType.POST_MIN_TRACTION_STREAM_FOR_GROX
    ELIGIBILITIES_TO_INJECT = {TaskEligibility.BANGER_INITIAL_SCREEN}

    def _get_loader(self):
        return KafkaPostLoader(
            KafkaTopicName.CONTENT_UNDERSTANDING_REALTIME_UNIFIED_POSTS_MIN_TRACTION_FOR_GROX
        )


class MinTractionPostStreamForGroxPtosTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = TaskGeneratorType.POST_MIN_TRACTION_STREAM_FOR_GROX_PTOS
    ELIGIBILITIES_TO_INJECT = {TaskEligibility.SAFETY_PTOS}

    def _get_loader(self):
        return KafkaPostLoader(
            KafkaTopicName.CONTENT_UNDERSTANDING_REALTIME_UNIFIED_POSTS_MIN_TRACTION_FOR_GROX_PTOS
        )


class MinTractionPostStreamForGroxMultiModalTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = (
        TaskGeneratorType.POST_MIN_TRACTION_STREAM_FOR_GROX_MULTI_MODAL
    )
    ELIGIBILITIES_TO_INJECT = {TaskEligibility.POST_EMBEDDING_WITH_SUMMARY_FOR_REPLY}

    def _get_loader(self):
        return KafkaPostLoader(
            KafkaTopicName.CONTENT_UNDERSTANDING_REALTIME_UNIFIED_POSTS_MIN_TRACTION_FOR_GROX_MULTI_MODAL
        )


class PostEmbeddingRequestWithSummaryForReplyRecoveryStreamTaskGenerator(
    StreamTaskGenerator
):
    TASK_GENERATOR_TYPE = (
        TaskGeneratorType.POST_EMBEDDING_REQUEST_STREAM_WITH_SUMMARY_FOR_REPLY_RECOVERY
    )
    ELIGIBILITIES_TO_INJECT = {TaskEligibility.POST_EMBEDDING_WITH_SUMMARY_FOR_REPLY}

    def _get_loader(self):
        return KafkaPostLoader(
            KafkaTopicName.GROX_MULTIMODAL_EMBEDDING_REQUESTS_WITH_SUMMARY_FOR_REPLY_RECOVERY
        )


class PostStreamRecoveryTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = TaskGeneratorType.POST_STREAM_RECOVERY
    ELIGIBILITIES_TO_INJECT = {TaskEligibility.BANGER_INITIAL_SCREEN}

    def _get_loader(self):
        return KafkaPostLoader(
            KafkaTopicName.CONTENT_UNDERSTANDING_REALTIME_UNIFIED_POSTS_RECOVERY
        )


class SafetyPtosRecoveryStreamTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = TaskGeneratorType.SAFETY_PTOS_RECOVERY
    ELIGIBILITIES_TO_INJECT = {TaskEligibility.SAFETY_PTOS}

    def _get_loader(self):
        return KafkaPostLoader(KafkaTopicName.SAFETY_PTOS_RECOVERY)


class SafetyPtosDeluxeStreamTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = TaskGeneratorType.SAFETY_PTOS_DELUXE
    ELIGIBILITIES_TO_INJECT = {TaskEligibility.SAFETY_PTOS}

    def _get_loader(self):
        return KafkaPostLoader(KafkaTopicName.SAFETY_PTOS_DELUXE)


class PostStreamTestTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = TaskGeneratorType.POST_STREAM_TEST
    ELIGIBILITIES_TO_INJECT = {TaskEligibility.BANGER_INITIAL_SCREEN}

    def _get_loader(self):
        return KafkaPostLoader(
            KafkaTopicName.CONTENT_UNDERSTANDING_REALTIME_UNIFIED_POSTS_TEST
        )


class PostSafetyStreamTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = TaskGeneratorType.POST_SAFETY_STREAM
    ELIGIBILITIES_TO_INJECT = {TaskEligibility.POST_SAFETY}

    def _get_loader(self):
        return KafkaPostLoader(
            KafkaTopicName.CONTENT_UNDERSTANDING_REALTIME_UNIFIED_POSTS_POPULAR
        )


class PostStreamDelayedTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = TaskGeneratorType.POST_STREAM_DELAYED
    ELIGIBILITIES_TO_INJECT = {}

    def _get_loader(self):
        return KafkaPostLoader(
            KafkaTopicName.CONTENT_UNDERSTANDING_REALTIME_UNIFIED_POSTS_DELAYED
        )


class ReplyRankingRecoveryTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = TaskGeneratorType.REPLY_RANKING_RECOVERY
    ELIGIBILITIES_TO_INJECT = {TaskEligibility.REPLY_RANKING}

    def _get_loader(self):
        return KafkaPostLoader(KafkaTopicName.REPLY_RANKING_RECOVERY)


class PostEmbeddingRequestWithSummaryStreamTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = TaskGeneratorType.POST_EMBEDDING_REQUEST_STREAM_WITH_SUMMARY
    ELIGIBILITIES_TO_INJECT = {TaskEligibility.POST_EMBEDDING_WITH_SUMMARY}

    def _get_loader(self):
        return KafkaPostLoader(
            KafkaTopicName.GROX_MULTIMODAL_EMBEDDING_REQUESTS_WITH_SUMMARY
        )


class PostEmbeddingRequestWithSummaryRecoveryStreamTaskGenerator(StreamTaskGenerator):
    TASK_GENERATOR_TYPE = (
        TaskGeneratorType.POST_EMBEDDING_REQUEST_STREAM_WITH_SUMMARY_RECOVERY
    )
    ELIGIBILITIES_TO_INJECT = {TaskEligibility.POST_EMBEDDING_WITH_SUMMARY}

    def _get_loader(self):
        return KafkaPostLoader(
            KafkaTopicName.GROX_MULTIMODAL_EMBEDDING_REQUESTS_WITH_SUMMARY_RECOVERY
        )


class PostEmbeddingV5StreamTaskGenerator(StreamTaskGenerator):

    TASK_GENERATOR_TYPE = TaskGeneratorType.POST_EMBEDDING_V5_STREAM
    ELIGIBILITIES_TO_INJECT = {TaskEligibility.MM_EMB_V5}

    def _get_loader(self):
        return KafkaPostLoader(
            KafkaTopicName.GROX_MULTIMODAL_EMBEDDING_REQUESTS_WITH_SUMMARY
        )


class PostEmbeddingV5ForReplyStreamTaskGenerator(StreamTaskGenerator):

    TASK_GENERATOR_TYPE = TaskGeneratorType.POST_EMBEDDING_V5_FOR_REPLY_STREAM
    ELIGIBILITIES_TO_INJECT = {TaskEligibility.MM_EMB_V5_FOR_REPLY}

    def _get_loader(self):
        return KafkaPostLoader(
            KafkaTopicName.CONTENT_UNDERSTANDING_REALTIME_UNIFIED_POSTS_MIN_TRACTION_FOR_GROX_MULTI_MODAL
        )
