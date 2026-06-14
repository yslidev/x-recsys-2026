import re
import uuid
import logging


from grox.data_loaders.media_loader import MediaLoader
from grox.data_loaders.strato_loader import TweetStratoLoader
from grox.lm.post import PostRenderer
from grox.lm.user import UserRenderer
from grox.lm.convo import Role, Message, Conversation
from grox.config.config import ModelName, grox_config
from grok_sampler.config import GrokModelConfig
from grox.prompts.template import BangerMiniVlmScreenScore
from grok_sampler.vision_sampler import VisionSampler
from grox.data_loaders.data_types import (
    Post,
    ContentCategoryType,
    ContentCategoryResult,
    TweetBoolMetadata,
    ContentCategoryScore,
)
from grox.classifiers.content.classifier import ContentClassifier
from monitor.metrics import Metrics
from pydantic import BaseModel
from strato_http.queries.grok_topics import StratoGrokTopics

logger = logging.getLogger(__name__)


class BangerInitialScreenResult(BaseModel):
    quality_score: float
    description: str
    tags: list[str]
    taxonomy_categories: list[dict] | None = None
    tweet_bool_metadata: TweetBoolMetadata | None = None
    is_image_editable_by_grok: bool | None = None
    slop_score: int | None = None
    has_minor_score: float | None = None


class BangerInitialScreenClassifier(ContentClassifier):
    result_pattern = re.compile(r"(.*)<json>(.*)</json>", re.DOTALL)

    def __init__(self):
        vlm_config = grox_config.get_model(ModelName.VLM_PRIMARY)
        vlm_config.temperature = 0.000001
        vlm = VisionSampler(GrokModelConfig(**vlm_config.model_dump()))
        super().__init__(
            categories=[
                ContentCategoryType.BANGER_INITIAL_SCREEN,
                ContentCategoryType.GROK_RANKER,
            ],
            llm=vlm,
        )
        self._topics = None

    @staticmethod
    def build_convo(post: Post, topics: list | None = None) -> Conversation:
        convo = Conversation(conversation_id=uuid.uuid4().hex)
        convo.messages.append(
            Message(
                role=Role.SYSTEM,
                content=[BangerMiniVlmScreenScore().render(params={"topics": topics})],
            )
        )

        user_msg = Message(role=Role.USER, content=[])
        user_msg.content.extend(UserRenderer.render(post.user))
        user_msg.content.extend(PostRenderer.render(post))
        user_msg.content.append(
            f"\n\nAnalyze the post {post.id} and provide the requested JSON object for the post."
        )
        convo.messages.append(user_msg)

        convo.messages.append(Message(role=Role.ASSISTANT, content=[""], separator=""))
        return convo

    async def classify(
        self, post: Post, topics: list | None = None
    ) -> list[ContentCategoryResult]:
        self._topics = topics
        return await super().classify(post)

    async def _classify_for_eval(self, post: Post) -> str:
        self._topics = None
        convo = await self._to_convo(post)
        logger.info(f"Banger initial screen conversation for post {post.id}")
        result = await self.llm.sample(
            convo.interleave(), conversation_id=convo.conversation_id
        )
        logger.info(f"Banger initial screen result for post {post.id}: {result}")
        return result

    async def _to_convo(self, post: Post) -> Conversation:
        return self.build_convo(post, topics=self._topics)

    async def _sample(self, convo: Conversation) -> str:
        return await self.llm.sample(
            convo.interleave(), conversation_id=convo.conversation_id
        )

    async def _parse(self, post: Post, output: str) -> list[ContentCategoryResult]:
        match = self.result_pattern.search(output)
        if match:
            reasoning = match.group(1).strip()
            logger.info(
                f"Banger initial screen result reasoning for post {post.id}: {reasoning}"
            )
            result = BangerInitialScreenResult.model_validate_json(
                match.group(2).strip()
            )
            score = result.quality_score
            Metrics.histogram(
                "banger_initial_screen_score",
                explicit_bucket_boundaries_advisory=[
                    0,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1,
                ],
            ).record(score)
            banger_initial_positive = score >= 0.4

            taxonomy_categories = []
            if result.taxonomy_categories:
                for tc in result.taxonomy_categories:
                    taxonomy_categories.append(
                        ContentCategoryScore(
                            id=tc["id"],
                            name=tc["name"],
                            score=tc["score"],
                            category_id=None,
                        )
                    )

            return [
                ContentCategoryResult(
                    category=cat,
                    positive=banger_initial_positive,
                    score=score,
                    reason=reasoning,
                    summary=result.description,
                    tags=result.tags,
                    taxonomy_categories=taxonomy_categories,
                    tweet_bool_metadata=result.tweet_bool_metadata,
                    is_image_editable_by_grok=result.is_image_editable_by_grok,
                    slop_score=result.slop_score,
                    has_minor_score=result.has_minor_score,
                )
                for cat in self.categories
            ]
        else:
            raise ValueError(f"Invalid output: {output}")
