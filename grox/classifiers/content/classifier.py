import logging
import time
import traceback

from abc import ABC, abstractmethod
from grok_sampler.llm import LiteLLM
from grox.data_loaders.data_types import (
    ContentCategoryResult,
    ContentCategoryType,
    Post,
)
from grox.lm.convo import Conversation
from monitor.metrics import Metrics
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ContentClassifier(ABC):
    def __init__(self, categories: list[ContentCategoryType], llm: LiteLLM):
        self.categories = categories
        self.llm = llm

    @property
    def model_name(self) -> str:
        return self.llm.model_config.model_name

    async def classify(self, post: Post) -> list[ContentCategoryResult]:
        logger.info(
            f"[{self.__class__.__name__}] started processing content classify request: {post.id}"
        )
        for category in self.categories:
            Metrics.counter(f"content.classification.request.count").add(
                1, attributes={"category": category.value.lower()}
            )
        for category in self.categories:
            Metrics.counter("content.classification.intake.count").add(
                1, attributes={"category": category.value.lower()}
            )
        start = time.perf_counter()
        try:
            res = await self._classify(post)
        except Exception:
            for category in self.categories:
                Metrics.counter(f"content.classification.error.count").add(
                    1, attributes={"category": category.value.lower()}
                )
            logger.error(
                f"[{self.__class__.__name__}] error processing content classify request: {post.id} {traceback.format_exc()}"
            )
            raise
        for category in self.categories:
            Metrics.counter(f"content.classification.success.count").add(
                1, attributes={"category": category.value.lower()}
            )
        end = time.perf_counter()
        logger.info(
            f"[{self.__class__.__name__}] finished processing content classify request: {post.id} in {end - start:.2f} seconds"
        )
        Metrics.histogram(f"content.classification.latency").record(
            end - start, attributes={"class": self.__class__.__name__}
        )
        self._post_process_for_logging(res, start, end)
        return res

    def _post_process_for_logging(
        self, res: list[BaseModel], start_time, end_time
    ) -> None:
        for category in self.categories:
            Metrics.histogram(f"content.classification.latency").record(
                end_time - start_time, attributes={"category": category.value.lower()}
            )

        for result in res:
            Metrics.counter(f"content.classification.result.count").add(
                1,
                attributes={
                    "category": result.category.value.lower(),
                    "positive": str(result.positive),
                },
            )

    @abstractmethod
    async def _to_convo(self, post: Post) -> Conversation:
        pass

    @abstractmethod
    async def _sample(self, convo: Conversation) -> str:
        pass

    @abstractmethod
    async def _parse(self, post: Post, output: str) -> list[ContentCategoryResult]:
        pass

    async def _classify(self, post: Post) -> list[ContentCategoryResult]:
        convo = await self._to_convo(post)
        output = await self._sample(convo)
        return await self._parse(post, output)
