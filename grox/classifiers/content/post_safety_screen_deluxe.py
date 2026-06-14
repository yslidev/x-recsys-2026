import re
import uuid
import logging
import asyncio


from grox.data_loaders.media_loader import MediaLoader
from grox.data_loaders.strato_loader import TweetStratoLoader
from grox.lm.post import PostRenderer
from grox.lm.user import UserRenderer
from grox.lm.convo import Role, Message, Conversation
from grox.config.config import ModelName, grox_config
from grok_sampler.config import GrokModelConfig
from grox.prompts.template import PostSafetyDeluxe
from grok_sampler.vision_sampler import VisionSampler
from grox.data_loaders.data_types import (
    Post,
    ContentCategoryType,
    ContentCategoryResult,
    TweetBoolMetadata,
)
from grox.classifiers.content.classifier import ContentClassifier
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PostSafetyScreenResult(BaseModel):
    tweet_bool_metadata: TweetBoolMetadata


class PostSafetyDeluxeClassifier(ContentClassifier):
    result_pattern = re.compile(r"(.*)<json>(.*)</json>", re.DOTALL)

    def __init__(self):
        vlm_config = grox_config.get_model(ModelName.VLM_PRIMARY_CRITICAL)
        vlm_config.temperature = 0.000001
        vlm = VisionSampler(GrokModelConfig(**vlm_config.model_dump()))
        super().__init__(
            categories=[
                ContentCategoryType.POST_SAFETY_SCREEN,
            ],
            llm=vlm,
        )

    @staticmethod
    def build_convo(post: Post) -> Conversation:
        convo = Conversation(conversation_id=uuid.uuid4().hex)
        convo.messages.append(
            Message(role=Role.SYSTEM, content=[PostSafetyDeluxe().render()])
        )

        user_msg = Message(role=Role.USER, content=[])
        user_msg.content.extend(UserRenderer.render(post.user))
        user_msg.content.extend(PostRenderer.render(post))
        user_msg.content.append(
            f"\n\nAnalyze the post {post.id} and provide the requested JSON object for the post."
        )
        convo.messages.append(user_msg)

        convo.messages.append(Message(role=Role.ASSISTANT, content=[]))
        return convo

    async def _to_convo(self, post: Post) -> Conversation:
        return self.build_convo(post)

    async def _sample(self, convo: Conversation) -> str:
        return await self.llm.sample(
            convo.interleave(), conversation_id=convo.conversation_id
        )

    async def _parse(self, post: Post, output: str) -> list[ContentCategoryResult]:
        match = self.result_pattern.search(output)
        if match:
            reasoning = match.group(1).strip()
            logger.info(
                f"Post Safety Screen reasoning for post {post.id} : {reasoning}"
            )
            result = PostSafetyScreenResult.model_validate_json(match.group(2).strip())
            logger.info(
                f"Post Safety Screen result for post {post.id} : {result}"
            )
            return [
                ContentCategoryResult(
                    category=ContentCategoryType.POST_SAFETY_SCREEN,
                    positive=False,
                    score=0.0,
                    tweet_bool_metadata=result.tweet_bool_metadata,
                )
            ]
        else:
            raise ValueError(f"Invalid output: {output}")
