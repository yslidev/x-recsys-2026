import asyncio
import logging
import traceback
import multiprocessing
from queue import Empty, Queue
from threading import Event
from multiprocessing import Process

from tenacity import retry, wait_incrementing, stop_after_attempt
from monitor.metrics import Metrics

from grox.config.config import TaskGeneratorType, grox_config
from grox.schedules.init import init_proc
from grox.schedules.types import TaskResult, TaskPayload
from grox.schedules.context import ScheduleContext
from grox.generators.task_generator import TaskGenerator, PriorityTaskGenerator
from grox.generators.stream_generator import (
    PostStreamTaskGenerator,
    PostStreamRecoveryTaskGenerator,
    PostStreamTestTaskGenerator,
    PostStreamDelayedTaskGenerator,
    PostSafetyStreamTaskGenerator,
    ReplyRankingRecoveryTaskGenerator,
    PostEmbeddingRequestWithSummaryStreamTaskGenerator,
    PostEmbeddingRequestWithSummaryRecoveryStreamTaskGenerator,
    MinTractionPostStreamForGroxTaskGenerator,
    MinTractionPostStreamForGroxPtosTaskGenerator,
    PostEmbeddingV5StreamTaskGenerator,
    PostEmbeddingV5ForReplyStreamTaskGenerator,
    MinTractionPostStreamForGroxMultiModalTaskGenerator,
    PostEmbeddingRequestWithSummaryForReplyRecoveryStreamTaskGenerator,
    SafetyPtosRecoveryStreamTaskGenerator,
    SafetyPtosDeluxeStreamTaskGenerator,
)

logger = logging.getLogger(__name__)


class Dispatcher:
    def __init__(self, context: ScheduleContext):
        self.config = grox_config.dispatcher
        self.context = context
        self._task_queue: Queue[TaskPayload] = self.context["task_queue"]
        self._resp_queue: Queue[TaskResult] = self.context["resp_queue"]
        self._shutdown_event: Event = self.context["shutdown_event"]
        self._queue_connection_shutdown_event: Event = self.context[
            "queue_connection_shutdown_event"
        ]
        self._process = None

    def _is_shutdown(self) -> bool:
        try:
            return self._shutdown_event.is_set()
        except BrokenPipeError:
            logger.error("Broken pipe error, assuming shutdown")
            return True
        except Exception:
            logger.error(
                f"Error checking shutdown event, assuming shutdown: {traceback.format_exc()}"
            )
            return True

    def _is_queue_connection_shutdown(self) -> bool:
        try:
            return self._queue_connection_shutdown_event.is_set()
        except BrokenPipeError:
            logger.error("Broken pipe error, assuming queue connection shutdown")
            return True
        except Exception:
            logger.error(
                f"Error checking shutdown event, assuming queue connection shutdown: {traceback.format_exc()}"
            )
            return True

    @retry(
        stop=stop_after_attempt(3), wait=wait_incrementing(start=1, increment=3, max=9)
    )
    async def _init_run(self):
        await init_proc("dispatcher")
        self._in_flights: set[str] = set()
        self._task_generator = self._get_task_generators()
        await self._task_generator.start()

    def _get_task_generators(self) -> TaskGenerator:
        generators: list[tuple[TaskGenerator, int]] = []
        for task_generator_config in self.config.task_generators:
            match task_generator_config.type:
                case TaskGeneratorType.POST_STREAM:
                    generators.append(
                        (
                            PostStreamTaskGenerator(task_generator_config.max_qps),
                            task_generator_config.weight,
                        )
                    )
                case TaskGeneratorType.POST_STREAM_RECOVERY:
                    generators.append(
                        (
                            PostStreamRecoveryTaskGenerator(
                                task_generator_config.max_qps
                            ),
                            task_generator_config.weight,
                        )
                    )
                case TaskGeneratorType.POST_STREAM_TEST:
                    generators.append(
                        (
                            PostStreamTestTaskGenerator(task_generator_config.max_qps),
                            task_generator_config.weight,
                        )
                    )
                case TaskGeneratorType.POST_STREAM_DELAYED:
                    generators.append(
                        (
                            PostStreamDelayedTaskGenerator(
                                task_generator_config.max_qps
                            ),
                            task_generator_config.weight,
                        )
                    )
                case TaskGeneratorType.POST_SAFETY_STREAM:
                    generators.append(
                        (
                            PostSafetyStreamTaskGenerator(
                                task_generator_config.max_qps
                            ),
                            task_generator_config.weight,
                        )
                    )
                case TaskGeneratorType.POST_MIN_TRACTION_STREAM_FOR_GROX:
                    generators.append(
                        (
                            MinTractionPostStreamForGroxTaskGenerator(
                                task_generator_config.max_qps
                            ),
                            task_generator_config.weight,
                        )
                    )
                case TaskGeneratorType.POST_MIN_TRACTION_STREAM_FOR_GROX_PTOS:
                    generators.append(
                        (
                            MinTractionPostStreamForGroxPtosTaskGenerator(
                                task_generator_config.max_qps
                            ),
                            task_generator_config.weight,
                        )
                    )
                case TaskGeneratorType.POST_MIN_TRACTION_STREAM_FOR_GROX_MULTI_MODAL:
                    generators.append(
                        (
                            MinTractionPostStreamForGroxMultiModalTaskGenerator(
                                task_generator_config.max_qps
                            ),
                            task_generator_config.weight,
                        )
                    )
                case TaskGeneratorType.POST_EMBEDDING_REQUEST_STREAM_WITH_SUMMARY_FOR_REPLY_RECOVERY:
                    generators.append(
                        (
                            PostEmbeddingRequestWithSummaryForReplyRecoveryStreamTaskGenerator(
                                task_generator_config.max_qps
                            ),
                            task_generator_config.weight,
                        )
                    )
                case TaskGeneratorType.SAFETY_PTOS_RECOVERY:
                    generators.append(
                        (
                            SafetyPtosRecoveryStreamTaskGenerator(
                                task_generator_config.max_qps
                            ),
                            task_generator_config.weight,
                        )
                    )
                case TaskGeneratorType.SAFETY_PTOS_DELUXE:
                    generators.append(
                        (
                            SafetyPtosDeluxeStreamTaskGenerator(
                                task_generator_config.max_qps
                            ),
                            task_generator_config.weight,
                        )
                    )
                case TaskGeneratorType.POST_EMBEDDING_V5_STREAM:
                    generators.append(
                        (
                            PostEmbeddingV5StreamTaskGenerator(
                                task_generator_config.max_qps
                            ),
                            task_generator_config.weight,
                        )
                    )
                case TaskGeneratorType.POST_EMBEDDING_V5_FOR_REPLY_STREAM:
                    generators.append(
                        (
                            PostEmbeddingV5ForReplyStreamTaskGenerator(
                                task_generator_config.max_qps
                            ),
                            task_generator_config.weight,
                        )
                    )
                case TaskGeneratorType.REPLY_RANKING_RECOVERY:
                    generators.append(
                        (
                            ReplyRankingRecoveryTaskGenerator(
                                task_generator_config.max_qps
                            ),
                            task_generator_config.weight,
                        )
                    )
                case TaskGeneratorType.POST_EMBEDDING_REQUEST_STREAM_WITH_SUMMARY:
                    generators.append(
                        (
                            PostEmbeddingRequestWithSummaryStreamTaskGenerator(
                                task_generator_config.max_qps
                            ),
                            task_generator_config.weight,
                        )
                    )
                case TaskGeneratorType.POST_EMBEDDING_REQUEST_STREAM_WITH_SUMMARY_RECOVERY:
                    generators.append(
                        (
                            PostEmbeddingRequestWithSummaryRecoveryStreamTaskGenerator(
                                task_generator_config.max_qps
                            ),
                            task_generator_config.weight,
                        )
                    )
                case _:
                    raise ValueError(
                        f"Invalid task generator type: {task_generator_config.type}"
                    )
        return PriorityTaskGenerator(generators)

    async def _submit_task(self, task_payload: TaskPayload) -> None:
        inflight_gauge = Metrics.gauge("dispatcher.inflight.count")
        self._in_flights.add(task_payload.payload_id)
        inflight_gauge.set(len(self._in_flights))
        Metrics.counter("dispatcher.task.sent.count").add(
            1,
            attributes={
                "task_type": task_payload.task_type.value
                if task_payload.task_type
                else "none"
            },
        )
        self._task_queue.put(task_payload)
        logger.debug(
            f"Submitted task: {task_payload.payload_id}, queue size: {self._task_queue.qsize()}"
        )

    async def _fill_loop(self):
        logger.info("Starting fill loop")
        while not self._is_shutdown():
            try:
                async for task_payload in self._task_generator.poll():
                    if task_payload is None:
                        await asyncio.sleep(0.01)
                        continue
                    while len(self._in_flights) >= self.config.max_in_flight:
                        await asyncio.sleep(0.01)
                        continue
                    await self._submit_task(task_payload)
            except Exception:
                logger.error(
                    f"Error polling from task queues: {traceback.format_exc()}"
                )

    async def _poll_result(self) -> TaskResult | None:
        try:
            res = self._resp_queue.get(block=False)
            logger.debug(f"Dispatcher received result: {res.task.payload_id}")
            Metrics.counter("dispatcher.result.received.count").add(1)
            return res
        except Empty:
            return None
        except BrokenPipeError:
            logger.error("Broken pipe error, shutting down")
            return None
        except Exception:
            logger.error(f"failed to poll result: {traceback.format_exc()}")
            return None

    async def _result_loop(self) -> None:
        logger.info("Starting result loop")
        max_attempts = self.config.max_attempts
        inflight_gauge = Metrics.gauge("dispatcher.inflight.count")
        while not self._is_shutdown() or self._in_flights:
            try:
                result = await self._poll_result()
                if result is None:
                    await asyncio.sleep(0.01)
                    continue
                task = result.task
                if result.success:
                    Metrics.counter("dispatcher.result.success.count").add(1)
                    if task.payload_id in self._in_flights:
                        self._in_flights.remove(task.payload_id)
                        inflight_gauge.set(len(self._in_flights))
                    await self._task_generator.ack(result)
                else:
                    if task.attempt < max_attempts:
                        Metrics.counter("dispatcher.result.failed.count").add(1)
                        logger.warning(
                            f"Task {task.payload_id} failed, retrying... (attempt {task.attempt})"
                        )
                        task.attempt += 1
                        await self._submit_task(task)
                    else:
                        if task.payload_id in self._in_flights:
                            self._in_flights.remove(task.payload_id)
                            inflight_gauge.set(len(self._in_flights))
                        logger.error(
                            f"Task {task.payload_id} failed after {max_attempts} attempts, error is {result.error}"
                        )
                        origin = self._task_generator.identify_task_origin(result)
                        Metrics.counter("dispatcher.result.failed.final.count").add(
                            1,
                            attributes={
                                "origin": origin.value if origin else "unknown"
                            },
                        )
                        if origin is None:
                            logger.warning(
                                f"No origin found for task {task.payload_id}, skipping ack"
                            )
                        else:
                            await self._task_generator.ack(result)
            except Empty:
                await asyncio.sleep(0.1)
            except BrokenPipeError:
                logger.error("Broken pipe error, shutting down")
                break
        logger.warning("Result loop finished")

    async def _wait_for_queue_connection_shutdown(self):
        while not self._is_queue_connection_shutdown():
            await asyncio.sleep(1)
        logger.warning("Shutdowning task generators")
        await self._task_generator.stop()
        logger.warning("Task generators stopped")

    async def _run(self, started_event: Event):
        await self._init_run()
        started_event.set()
        await asyncio.gather(
            self._fill_loop(),
            self._result_loop(),
            self._wait_for_queue_connection_shutdown(),
        )

    def run(self, started_event: Event):
        asyncio.run(self._run(started_event))

    async def start(self):
        logger.info("Starting Grox dispatcher...")
        started_event = multiprocessing.Event()
        self._process = Process(
            target=self.run, args=(started_event,), name="grox-dispatcher"
        )
        self._process.start()
        started_event.wait()
        logger.info("Grox dispatcher started")

    async def stop(self):
        logger.warning("Stopping Grox dispatcher...")
        if self._process and self._process.is_alive():
            self._process.join(self.config.graceful_shutdown_timeout)
        else:
            logger.warning("Dispatcher process is not alive, skipping join")
        logger.warning("Dispatcher stopped")
