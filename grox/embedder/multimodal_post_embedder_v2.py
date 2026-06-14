import os
import time
import logging
import json
import numpy as np
from grox.lm.post import LitePostRenderer, MMEmbedPostRenderer, EvalPostRenderer
from grox.lm.convo import Image as ConvoImage, Video as ConvoVideo, Content
from embed.embed_cli import XaiEmbeddingClient
from monitor.metrics import Metrics
from grox.config.config import ModelName, grox_config
from grox.data_loaders.data_types import Post
from grox.data_loaders.media_loader import MediaLoader
from grox.data_loaders.strato_loader import TweetStratoLoader
from grox.data_loaders.media_description_loader import MediaDescriptionLoader
from strato_http.queries.post_multimodal_embedding_mh_searchai import (
    StratoContentUnderstandingUnifiedPostAnnotations,
)
from grox.config.config import EmbeddingModelConfig

logger = logging.getLogger(__name__)


class MultimodalPostEmbedderV2:
    def __init__(
        self,
        model: str = "qwen3",
        use_grok_summary: bool = False,
        renderer_version="mmembed_summary",
        use_media_descriptions: bool = False,
        use_post_context_summary: bool = False,
        use_grok_summary_path: str = "",
        custom_endpoint: str = "",
        instruction: str = "",
    ):
        embed_config = grox_config.get_embedding_model(ModelName.EMBED_PRIMARY)
        video_embed_config = grox_config.get_embedding_model(ModelName.EMBED_PRIMARY_VIDEO)
        embed_config.text_max_len = 8192
        video_embed_config.text_max_len = 8192
        qwen_3_embed_06b_config = grox_config.get_embedding_model(
            ModelName.EMBED_SMALL
        )
        qwen_3_embed_8b_config = grox_config.get_embedding_model(
            ModelName.EMBED_LARGE
        )
        recsys_v4_embed_config = grox_config.get_embedding_model(
            ModelName.RECSYS_EMBED_V4
        )
        qwen_3_embed_06b_config.text_max_len = 4096
        qwen_3_embed_8b_config.text_max_len = 4096
        recsys_v4_embed_config.text_max_len = 4096
        if custom_endpoint:
            custom_embed_config = EmbeddingModelConfig(
                model_name="custom", endpoint=custom_endpoint, text_max_len=4096
            )
            self._custom_embed_client = XaiEmbeddingClient(config=custom_embed_config)
        self.use_custom_embed = True if custom_endpoint else False
        self.renderer_version = renderer_version
        self._client = XaiEmbeddingClient(config=embed_config)
        self._video_client = XaiEmbeddingClient(config=video_embed_config)
        self._qwen_3_embed_06b_client = XaiEmbeddingClient(
            config=qwen_3_embed_06b_config
        )
        self._qwen_3_embed_8b_client = XaiEmbeddingClient(config=qwen_3_embed_8b_config)
        self._recsys_v4_embed_client = XaiEmbeddingClient(config=recsys_v4_embed_config)
        self.model = model
        self.use_grok_summary = use_grok_summary
        self.use_media_descriptions = use_media_descriptions
        self.use_grok_summary_versioned = False
        self.instruction = instruction
        self.use_post_context_summary = use_post_context_summary

        if use_grok_summary_path:
            assert os.path.exists(use_grok_summary_path), (
                f"Grok summary path {use_grok_summary_path} does not exist"
            )
            assert use_grok_summary_path.endswith(".jsonl"), (
                f"Grok summary path {use_grok_summary_path} is not a jsonl file"
            )
            self.grok_summary_versioned: dict[str, str] = {}
            self.use_grok_summary_versioned = True
            with open(use_grok_summary_path, "r") as f:
                for line in f:
                    json_line = json.loads(line)
                    self.grok_summary_versioned[str(json_line["post_id"]).strip()] = (
                        json_line["summary"]
                    )

    def _get_client(
        self, num_text: int, num_image: int, num_video: int
    ) -> XaiEmbeddingClient:
        if self.use_custom_embed:
            return self._custom_embed_client
        if self.model == "qwen3":
            return self._qwen_3_embed_06b_client
        if self.model == "qwen3_8b":
            return self._qwen_3_embed_8b_client
        if self.model == "v4":
            return self._recsys_v4_embed_client

        if num_video > 0:
            logger.info(
                f"Using video client for post with {num_text} text, {num_image} images, and {num_video} videos"
            )
            return self._video_client
        return self._client

    def document_original(
        self, content: list[Content]
    ) -> tuple[list[tuple[str, str | bytes]], int, int, int]:
        def get_convo_video_instruction(video: ConvoVideo) -> str:
            res = [f"The video lasts for {video.total_duration:.2f} seconds."]
            bucket_times = [i * video.duration for i in range(len(video.frames))]
            res.append(
                f"The following frames are sampled at every {video.duration:.2f} second interval."
            )
            for i, frame in enumerate(video.frames):
                subtitle = (
                    video.subtitles[i]
                    if video.subtitles and i < len(video.subtitles)
                    else None
                )
                subtitle_str = (
                    f"with subtitle: {subtitle}" if subtitle else "(no subtitles)"
                )
                res.append(f"At {bucket_times[i]:.2f} seconds, {subtitle_str}.")
            res.append("The frames are listed below:")
            return " ".join(res)

        document = []
        num_text = 0
        num_image = 0
        num_video = 0
        new_text_part = ""
        for c in content:
            if isinstance(c, ConvoImage):
                document.append(("text", f"Image: \n"))
                document.append(("image", c.content))
                num_image += 1
            elif isinstance(c, ConvoVideo):
                if c.combined_video_bytes:
                    new_text_part += get_convo_video_instruction(c)
                    document.append(("video", c.combined_video_bytes))
                    num_video += 1
            elif isinstance(c, str):
                new_text_part += c
                num_text += 1
        new_text_part = new_text_part.strip()
        document.append(
            ("text", "")
        )
        document.append(("text", new_text_part))
        return document, num_text, num_image, num_video

    def document_v1(
        self, content: list[Content]
    ) -> tuple[list[tuple[str, str | bytes]], int, int, int]:
        def video_frames(
            video: ConvoVideo, index: int
        ) -> list[tuple[str, str | bytes]]:
            res: list[tuple[str, str | bytes]] = []
            for i, frame in enumerate(video.frames):
                res.append(("image", frame))
                if (
                    video.subtitles
                    and i < len(video.subtitles)
                    and video.subtitles[i] is not None
                    and video.subtitles[i].strip() != ""
                ):
                    res.append(("text", "subtitle: " + video.subtitles[i] + " "))
            return res

        document = []
        num_text = 0
        num_image = 0
        num_video = 0

        for c in content:
            if isinstance(c, str):
                document.append(("text", c.strip()))
                num_text += 1
            else:
                if isinstance(c, ConvoImage):
                    document.append(("image", c.content))
                    num_image += 1
                elif isinstance(c, ConvoVideo):
                    document.extend(video_frames(c, num_video))
                    num_video += 1

        return document, num_text, num_image, num_video

    async def _create_embeddings_for_post(
        self,
        content: list[Content],
        is_query: bool = False,
        document_version: str = "v1",
    ) -> tuple[list[tuple[str, str | bytes]], np.ndarray]:
        if document_version == "default":
            document_fn = self.document_original
        elif document_version == "v1":
            document_fn = self.document_v1
        else:
            raise ValueError(f"document_version not found: {document_version}")

        document, num_text, num_image, num_video = document_fn(content)
        logger.info(
            f"creating embeddings for post with {num_text} text, {num_image} images, and {num_video} videos"
        )
        client = self._get_client(num_text, num_image, num_video)
        return document, await client.create_embeddings_async(
            [document], is_query=is_query
        )

    async def hydrate_grok_post_summary(self, post: Post):
        query = StratoContentUnderstandingUnifiedPostAnnotations()
        res = await query.fetch(int(post.id))
        if res:
            description = res["annotations"]["description"]
            post.summary = description

    def get_detailed_instruct(self, instruction: str) -> str:
        return f"Instruct: {instruction}\nQuery: Please embed the following post:"

    def _get_document_fn(self, document_version: str):
        if document_version == "default":
            return self.document_original
        if document_version == "v1":
            return self.document_v1
        raise ValueError(f"document_version not found: {document_version}")

    async def embed_texts_batch(
        self, texts: list[str], is_query: bool = True, document_version: str = "v1"
    ) -> list[list[float]]:
        if not texts:
            return []
        document_fn = self._get_document_fn(document_version)
        documents: list[list[tuple[str, str | bytes]]] = []
        for text in texts:
            document, _, _, _ = document_fn([text])
            documents.append(document)
        client = self._get_client(num_text=1, num_image=0, num_video=0)
        embeddings = await client.create_embeddings_async(documents, is_query=is_query)
        return [embedding.flatten().tolist() for embedding in embeddings]

    async def embed(
        self, post: Post, is_query: bool = False, document_version: str = "v1"
    ) -> tuple[list[tuple[str, str | bytes]], list[float]]:
        if self.instruction:
            content: list[Content] = [self.get_detailed_instruct(self.instruction)]

        if self.renderer_version == "lite":
            content = LitePostRenderer.render_for_embedding(post)
        elif self.renderer_version == "eval":
            content = EvalPostRenderer.render_for_embedding(post)
        elif self.renderer_version == "mmembed_summary":
            content = await MMEmbedPostRenderer.render_for_embedding(
                post, use_grok_summary=self.use_grok_summary
            )

        if self.use_grok_summary_versioned:
            if str(post.id) in self.grok_summary_versioned:
                content.append(
                    f"\nPost summary and description: {self.grok_summary_versioned[str(post.id)]}"
                )

        if self.use_grok_summary and not self.use_grok_summary_versioned:
            await self.hydrate_grok_post_summary(post)
            content.append(f"\nPost summary and description: {post.summary}")

        if self.use_post_context_summary:
            content.append(f"\nPost summary and description: {post.summary}")

        if self.use_media_descriptions:
            await MediaDescriptionLoader.hydrate_media_descriptions(post)
            content.append(
                f"\nThe post has these associated media descriptions: \n{post.media_descriptions}"
            )

        start_time = time.perf_counter_ns()

        document, embedding = await self._create_embeddings_for_post(
            content, is_query, document_version
        )

        duration_ms = (time.perf_counter_ns() - start_time) / 1_000_000
        logger.info(f"Embedding finished in {duration_ms:.2f} ms")
        Metrics.histogram("post_embedding_duration_ms").record(duration_ms)
        return document, embedding.flatten().tolist()
