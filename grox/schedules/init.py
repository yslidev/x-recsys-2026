import asyncio
import gc
import logging
import random
import time
import setproctitle

from grox.config.config import grox_config
from grox.schedules.context import prevent_default
from monitor.logging import Logging
from monitor.metrics import Metrics

logger = logging.getLogger(__name__)


async def init_proc(proc_name: str):
    prevent_default()
    Logging.config(grox_config.logging)
    Metrics.init(proc_name, grox_config.metrics)
    setproctitle.setproctitle(proc_name)
    logger.info(f"Changed process title to {proc_name}")
    asyncio.create_task(periodic_gc(proc_name))


async def periodic_gc(proc_name: str):
    while True:
        seconds = grox_config.periodic_gc.interval + random.randint(
            0, int(grox_config.periodic_gc.jitter)
        )
        await asyncio.sleep(seconds)
        logger.info(f"Running periodic GC for {proc_name}")
        start = time.perf_counter()
        gc.collect()
        end = time.perf_counter()
        logger.info(f"GC for {proc_name} took {end - start:.2f} seconds")
