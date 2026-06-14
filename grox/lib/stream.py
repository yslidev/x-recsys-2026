from typing import AsyncIterator, AsyncGenerator, TypeVar
from asyncio import Queue, create_task
from enum import Enum

import logging

T = TypeVar("T")


class StreamStatus(Enum):
    STOP = "Stop"


logger = logging.getLogger(__name__)


async def parallel_merge(*streams: AsyncIterator[T]) -> AsyncGenerator[T, None]:
    if not streams:
        return
    queue: Queue[T | StreamStatus | Exception] = Queue()

    async def enqueue(ait: AsyncIterator[T]):
        try:
            async for item in ait:
                await queue.put(item)
        except GeneratorExit:
            pass
        except Exception as e:
            await queue.put(e)
        finally:
            await queue.put(StreamStatus.STOP)

    _enq_tasks = [create_task(enqueue(s)) for s in streams]

    nstreams_done = 0
    while True:
        item = await queue.get()
        if item == StreamStatus.STOP:
            nstreams_done += 1
        elif isinstance(item, Exception):
            raise item
        else:
            yield item
        queue.task_done()
        if nstreams_done == len(streams):
            break
