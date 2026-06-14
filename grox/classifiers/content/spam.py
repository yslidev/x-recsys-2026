import logging
import re
import uuid
from pydantic import ValidationError

from grox.lm.convo import Role, Message, Conversation
from grox.lm.thread import ThreadRenderer
from grox.config.config import ModelName, grox_config
from grok_sampler.config import GrokModelConfig
from grox.prompts.template import SpamSystemLowFollower
from grok_sampler.vision_sampler import VisionSampler
from grox.data_loaders.data_types import (
    Post,
    ContentCategoryType,
    ContentCategoryResult,
    SpamSampleResult,
)
from grox.classifiers.content.classifier import ContentClassifier
from grox.data_loaders.strato_loader import TweetStratoLoader
from grox.data_loaders.media_loader import MediaLoader

logger = logging.getLogger(__name__)


class SpamEapiLowFollowerClassifier(ContentClassifier):
    def __init__(self, model_name: ModelName = ModelName.VLM_PRIMARY):
        vlm_config = grox_config.get_model(model_name)
        vlm_config.temperature = 0.000001
        vlm = VisionSampler(GrokModelConfig(**vlm_config.model_dump()))
        super().__init__(categories=[ContentCategoryType.SPAM_COMMENT], llm=vlm)

    @property
    def model_name(self) -> str:
        return "grox"

    async def _classify(self, post: Post) -> list[ContentCategoryResult]:
        convo = await self._to_convo(post)
        result = await self._sample(convo)
        parsed = await self._parse(post, result)
        filtered_parsed = [
            res for res in parsed if res.category == ContentCategoryType.SPAM_COMMENT
        ]
        assert len(filtered_parsed) == 1
        return filtered_parsed

    async def _to_convo(self, post: Post) -> Conversation:
        convo = Conversation(conversation_id=uuid.uuid4().hex)
        convo.messages.append(
            Message(role=Role.SYSTEM, content=[SpamSystemLowFollower().render()])
        )
        convo.messages.append(
            ThreadRenderer.render(post, role=Role.HUMAN, separator="\n\n")
        )
        return convo

    async def _sample(self, convo: Conversation) -> str:
        return await self.llm.sample(
            convo.interleave(), conversation_id=convo.conversation_id
        )

    async def _clean_output(self, output: str) -> str:
        if output.endswith("<|eos|>"):
            output = output.removesuffix("<|eos|>")
        output = output.strip()
        if output.startswith("```json"):
            output = output[7:]
        elif output.startswith("```"):
            output = output[3:]
        if output.endswith("```"):
            output = output[:-3]
        output = output.strip()
        return output

    async def _parse(self, post: Post, output: str) -> list[ContentCategoryResult]:
        decision = None
        summary = ""

        cleaned_result = await self._clean_output(output)
        try:
            result = SpamSampleResult.model_validate_json(cleaned_result)
            decision = result.decision
            summary = result.reason
        except ValidationError:
            match = re.search(r'"decision":\s*"(.*?)"', cleaned_result)
            if match:
                decision = match.group(1).strip()

        if not decision:
            raise ValueError(f"Invalid output format: {output}")

        is_spam = decision == "spam"
        score = 1.0 if is_spam else 0.0

        if is_spam:
            logger.info(f"Spam found for low follower user: {post.id}")

        return [
            ContentCategoryResult(
                category=ContentCategoryType.SPAM_COMMENT,
                positive=is_spam,
                score=score,
                summary=summary,
            )
        ]
