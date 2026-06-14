import uuid
import logging

from grox.lm.post import PostRenderer
from grox.lm.user import UserRenderer
from grox.config.config import grox_config, ModelName
from grox.summarizer.summarizer import Summarizer
from grox.data_loaders.data_types import Post
from grox.lm.convo import Conversation, Message, Role
from grok_sampler.config import GrokModelConfig
from grok_sampler.vision_sampler import VisionSampler
import os

logger = logging.getLogger(__name__)


class PostEmbeddingSummarizer(Summarizer):
    def __init__(self, prompt_file: str):
        vlm_config = grox_config.get_model(ModelName.VLM_MINI_CRITICAL)
        vlm = VisionSampler(GrokModelConfig(**vlm_config.model_dump()))
        self.prompt_file: str = prompt_file
        if not os.path.exists(self.prompt_file):
            raise FileNotFoundError(f"Prompt file {self.prompt_file} not found")
        super().__init__(vlm_config, vlm)

    async def _summarize(self, post: Post) -> str:
        convo = await self._render_vlm_conversation(post)
        result = await self.vlm.sample(
            convo.interleave(), conversation_id=convo.conversation_id
        )
        result_section = result.split("<description>")[1].split("</description>")[0]
        return result_section

    async def _render_vlm_conversation(
        self, post: Post, disable_thinking: bool = True
    ) -> Conversation:
        convo = Conversation(conversation_id=uuid.uuid4().hex)
        prompt = ""
        with open(self.prompt_file, "r") as f:
            prompt = f.read()
        convo.messages.append(Message(role=Role.SYSTEM, content=[prompt]))
        convo.messages.append(await self._build_task_message(post))
        if disable_thinking:
            convo.messages.append(
                Message(role=Role.ASSISTANT, content=[""], separator="")
            )
        else:
            convo.messages.append(Message(role=Role.ASSISTANT))
        return convo

    async def _build_task_message(self, post: Post) -> Message:
        msg: Message = Message(role=Role.USER, content=[])
        msg.content.extend(UserRenderer.render(post.user))
        msg.content.extend(PostRenderer.render(post))
        return msg
