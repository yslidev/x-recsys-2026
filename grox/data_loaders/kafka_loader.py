import struct
import time
import uuid
import asyncio
import logging
import traceback
from abc import abstractmethod
from typing import override
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

from aiokafka import TopicPartition
from kafka_cli.config import KafkaMessage
from grox.config.config import KafkaTopicName, grox_config
from kafka_cli.consumer import KafkaConsumer
from grox.data_loaders.data_types import Post, User, GroxContentAnalysis
from grox.data_loaders.message_queue_loader import (
    MessageQueueLoader,
    MessageQueuePayload,
)
from monitor.metrics import Metrics
from thrifts.serdes import SerDesError


logger = logging.getLogger(__name__)
MAX_WORKING_THREADS = 12


class _Payload(MessageQueuePayload):
    tp: TopicPartition
    offset: int


class KafkaLoader(MessageQueueLoader):
    def __init__(self, topic_name: KafkaTopicName):
        super().__init__()
        self._initialized = False
        self._shutdown_event = asyncio.Event()
        self.topic_name = topic_name
        self.loader_config = grox_config.get_kafka_loader_topic(topic_name)
        self.consumer_config = grox_config.get_kafka_consumer_topic(topic_name)
        self.consumer = KafkaConsumer(self.consumer_config)
        self.loaded_messages: dict[str, tuple[TopicPartition, int]] = {}
        self.queue: asyncio.Queue[MessageQueuePayload] = asyncio.Queue()
        self._prefetcher_task: asyncio.Task | None = None

    def _is_shutdown(self) -> bool:
        try:
            return self._shutdown_event.is_set()
        except Exception:
            logger.error(
                f"Error checking if KafkaLoader is shutdown: {traceback.format_exc()}"
            )
            return True

    async def start(self):
        logger.info(f"Initializing KafkaLoader, topic: {self.topic_name}")
        self._initialized = True
        await self.consumer.start()
        self._prefetcher_task = asyncio.create_task(self._prefetcher())
        self._initialized = True
        logger.info(f"KafkaLoader initialized, topic: {self.topic_name}")

    async def stop(self):
        logger.warning(f"Stopping KafkaLoader, topic: {self.topic_name}")
        self._shutdown_event.set()
        try:
            if self._prefetcher_task:
                await asyncio.wait_for(self._prefetcher_task, 5)
        except asyncio.TimeoutError:
            logger.warning(
                f"Waiting prefetcher to stop timed out, topic: {self.topic_name}"
            )
        await self.consumer.stop()
        logger.warning(f"KafkaLoader stopped, topic: {self.topic_name}")

    async def poll(self) -> AsyncGenerator[MessageQueuePayload | None, None]:
        while not self._shutdown_event.is_set() or not self.queue.empty():
            try:
                yield self.queue.get_nowait()
            except asyncio.QueueEmpty:
                logger.debug(
                    f"Queue is empty, waiting for prefetcher to fill, topic: {self.topic_name}"
                )
                yield None
            except Exception:
                logger.error(
                    f"Error polling from kafka, topic: {self.topic_name}, error: {traceback.format_exc()}"
                )
                yield None

    async def ack(self, mid: str, success: bool = True):
        pass

    @abstractmethod
    def _messages_to_payloads(self, messages: list[KafkaMessage]) -> list[_Payload]:
        pass

    def _process_messages(self, messages: list[KafkaMessage]) -> list[_Payload]:
        group_size = max(1, len(messages) // MAX_WORKING_THREADS)
        message_groups = [
            messages[i : i + group_size] for i in range(0, len(messages), group_size)
        ]

        with ThreadPoolExecutor(max_workers=MAX_WORKING_THREADS) as executor:
            payloads = []
            for result in executor.map(self._messages_to_payloads, message_groups):
                payloads.extend(result)
        return payloads

    async def _prefetcher(self) -> None:
        logger.info(f"Starting prefetcher, topic: {self.topic_name}")
        prefetching_threshold = self.loader_config.prefetching_threshold
        prefetching_batch_size = self.loader_config.prefetching_batch_size
        while not self._is_shutdown():
            if self.queue.qsize() < prefetching_threshold:
                logger.debug(
                    f"Inventory low at {self.queue.qsize()}, prefetching {prefetching_batch_size} messages, topic: {self.topic_name}"
                )
                try:
                    messages = await self.consumer.poll(prefetching_batch_size)
                    try:
                        payloads = self._process_messages(messages)
                    except SerDesError:
                        logger.error(
                            f"Error processing messages, error: {traceback.format_exc()}"
                        )
                        raise
                    await asyncio.gather(
                        *[
                            self.queue.put(
                                MessageQueuePayload(
                                    mid=payload.mid,
                                    user=payload.user,
                                    post=payload.post,
                                    user_context=payload.user_context,
                                    grox_content_analysis=payload.grox_content_analysis,
                                    deadline_ts_secs=payload.deadline_ts_secs,
                                )
                            )
                            for payload in payloads
                        ]
                    )
                    logger.debug(
                        f"Prefetched {prefetching_batch_size} messages, inventory now at {self.queue.qsize()}, topic: {self.topic_name}"
                    )
                except Exception:
                    logger.error(
                        f"Error prefetching messages, error: {traceback.format_exc()}"
                    )
                    await asyncio.sleep(0.1)
            else:
                await asyncio.sleep(0.1)
        logger.warning("Prefetcher stopped")


class KafkaPostLoader(KafkaLoader):
    def __init__(self, topic_name: KafkaTopicName):
        super().__init__(topic_name)

    @override
    def _messages_to_payloads(self, messages: list[KafkaMessage]) -> list[_Payload]:
        return [
            _Payload(
                mid=uuid.uuid4().hex,
                post=Post.from_thrift_content_understanding_metadata(message.value),
                tp=message.tp,
                offset=message.offset,
                deadline_ts_secs=int(time.time())
                + self.loader_config.task_deadline_secs,
            )
            for message in messages
        ]


class KafkaPostEmbeddingRequestLoader(KafkaLoader):
    def __init__(self, topic_name: KafkaTopicName):
        super().__init__(topic_name)

    @override
    def _messages_to_payloads(self, messages: list[KafkaMessage]) -> list[_Payload]:
        return [
            _Payload(
                mid=uuid.uuid4().hex,
                post=Post.from_thrift_post_embedding_request(message.value),
                tp=message.tp,
                offset=message.offset,
                deadline_ts_secs=int(time.time())
                + self.loader_config.task_deadline_secs,
            )
            for message in messages
        ]


class KafkaGroxContentAnalysisLoader(KafkaLoader):
    def __init__(self, topic_name: KafkaTopicName):
        super().__init__(topic_name)

    @override
    def _messages_to_payloads(self, messages: list[KafkaMessage]) -> list[_Payload]:
        return [
            _Payload(
                mid=uuid.uuid4().hex,
                grox_content_analysis=GroxContentAnalysis.from_thrift_content_understanding_metadata(
                    message.value
                ),
                tp=message.tp,
                offset=message.offset,
                deadline_ts_secs=int(time.time())
                + self.loader_config.task_deadline_secs,
            )
            for message in messages
        ]


class KafkaTweetEmbeddingLoader(KafkaLoader):
    def __init__(self, topic_name: KafkaTopicName):
        super().__init__(topic_name)

    @override
    def _messages_to_payloads(self, messages: list[KafkaMessage]) -> list[_Payload]:
        return [
            _Payload(
                mid=uuid.uuid4().hex,
                post=Post.from_thrift_tweet_embedding(message.value),
                tp=message.tp,
                offset=message.offset,
                deadline_ts_secs=int(time.time())
                + self.loader_config.task_deadline_secs,
            )
            for message in messages
        ]
