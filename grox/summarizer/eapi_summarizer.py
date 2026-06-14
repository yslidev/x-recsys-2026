import logging
import time
import traceback

from abc import ABC, abstractmethod
from grok_sampler.eapi_sampler import EapiSampler
from grox.config.config import EapiModelConfig
from monitor.metrics import Metrics

logger = logging.getLogger(__name__)
from typing import TypeVar, Generic, Any

T = TypeVar("T")


class EapiSummarizer(ABC, Generic[T]):
    def __init__(self, eapi_config: EapiModelConfig, eapi_sampler: EapiSampler):
        self.eapi_config = eapi_config
        self.eapi_sampler = eapi_sampler

    async def summarize(self, input: T) -> Any:
        logger.info(f"[{self.__class__.__name__}] started processing summarize request")
        Metrics.counter("summarize.request.count").add(1)

        start = time.perf_counter()
        try:
            res = await self._summarize(input)
        except Exception:
            Metrics.counter("summarize.error.count").add(1)
            logger.error(
                f"[{self.__class__.__name__}] error processing summarize request: {traceback.format_exc()}"
            )
            raise
        Metrics.counter("summarize.success.count").add(1)
        end = time.perf_counter()
        logger.info(
            f"[{self.__class__.__name__}] finished processing summarize request in {end - start:.2f} seconds"
        )
        Metrics.histogram("summarize.latency.seconds").record(end - start)
        return res

    @abstractmethod
    async def _summarize(self, input: T) -> Any:
        pass
