import logging

from grox.data_loaders.asr_processor import ASRProcessor
from grox.data_loaders.data_types import Post, Video
from grox.schedules.types import TaskContext
from grox.tasks.task import TaskWithPost
from monitor.metrics import Metrics

logger = logging.getLogger(__name__)


def _get_video_url(video: Video) -> tuple[str | None, bool]:
    if video.animatedGifInfo:
        v = video.animatedGifInfo.get_best_variant()
        if v and v.url:
            return v.url, True
    if video.videoInfo:
        v = video.videoInfo.get_best_variant()
        if v and v.url:
            return v.url, False
    if video.url:
        return video.url, False
    return None, False


class TaskASRTranscription(TaskWithPost):
    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        Metrics.counter("task.asr_transcription.total.count").add(1)

        if not post.media:
            logger.debug(f"Post {post.id} has no media, skipping ASR")
            Metrics.counter("task.asr_transcription.skipped.count").add(
                1, attributes={"reason": "no_media"}
            )
            return

        videos = [m for m in post.media if isinstance(m, Video)]
        if not videos:
            logger.debug(f"Post {post.id} has no video, skipping ASR")
            Metrics.counter("task.asr_transcription.skipped.count").add(
                1, attributes={"reason": "no_video"}
            )
            return

        Metrics.counter("task.asr_transcription.has_video.count").add(1)

        for video in videos:
            video_url, is_animated_gif = _get_video_url(video)
            if not video_url:
                continue

            if is_animated_gif:
                logger.debug(
                    f"Post {post.id} video {video.id} is an animated GIF, skipping ASR (no audio)"
                )
                Metrics.counter("task.asr_transcription.skipped.count").add(
                    1, attributes={"reason": "animated_gif"}
                )
                continue

            transcript = await ASRProcessor.process(post.id, video_url)

            if transcript:
                if video.convo_video is not None:
                    video.convo_video.asr_transcript = transcript
                Metrics.counter("task.asr_transcription.success.count").add(1)
                logger.info(
                    f"ASR completed for post {post.id} video {video.id}, transcript_len={len(transcript)}"
                )
            else:
                Metrics.counter("task.asr_transcription.failed.count").add(
                    1, attributes={"reason": "processor_error"}
                )
