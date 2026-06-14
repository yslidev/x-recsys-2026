import logging

from abc import ABC, abstractmethod
from grox.data_loaders.data_types import Post, User, UserContext, GroxContentAnalysis
from collections.abc import AsyncGenerator
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MessageQueuePayload(BaseModel):
    mid: str
    post: Post | None = None
    user: User | None = None
    user_context: UserContext | None = None
    grox_content_analysis: GroxContentAnalysis | None = None

    deadline_ts_secs: int


class MessageQueueLoader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    async def start(self):
        pass

    @abstractmethod
    async def stop(self):
        pass

    @abstractmethod
    def poll(self) -> AsyncGenerator[MessageQueuePayload | None, None]:
        pass

    @abstractmethod
    async def ack(self, mid: str, success: bool = True):
        pass
