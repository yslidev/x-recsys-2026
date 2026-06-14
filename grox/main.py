import signal
import asyncio
import logging

from grox.engine import Engine
from grox.service import GrpcServer
from grox.dispatcher import Dispatcher
from grox.config.config import grox_config
from grox.schedules.init import init_proc
from grox.schedules.context import (
    cleanup,
    new_context,
    shutdown_context,
    queue_connection_shutdown_context,
)

logger = logging.getLogger(__name__)
shutdown = asyncio.Event()


async def serve():
    await init_proc("main")
    logger.info(f"Starting grox server...")
    context = new_context()
    engine = Engine(context)
    dispatcher = Dispatcher(context)
    grpc_server = GrpcServer(context)

    await engine.start()
    await dispatcher.start()
    await grpc_server.start()

    logger.info("Grox server started")
    event_loop = asyncio.get_running_loop()
    event_loop.add_signal_handler(signal.SIGINT, lambda: shutdown.set())
    event_loop.add_signal_handler(signal.SIGTERM, lambda: shutdown.set())

    await shutdown.wait()
    logger.warning("Grox server shutting down...")
    queue_connection_shutdown_context(context)
    await asyncio.sleep(300)

    shutdown_context(context)
    await asyncio.gather(
        grpc_server.stop(),
        dispatcher.stop(),
        engine.stop(),
    )
    cleanup()
    logger.warning("Grox server stopped")


if __name__ == "__main__":
    asyncio.run(serve())
