import logging
import re
import traceback
import uuid
from typing import List


from grox.data_loaders.media_loader import MediaLoader
from grox.data_loaders.strato_loader import TweetStratoLoader
from grox.lm.convo import Role, Message, Conversation
from grox.lm.post import PostRenderer
from grox.lm.user import UserRenderer
from grox.config.config import ModelName, grox_config
from grok_sampler.config import GrokModelConfig, EapiModelConfig
from grox.prompts.template import (
    SafetyPtos,
    ViolentMediaPolicy,
    AdultContentPolicy,
    SpamPolicy,
    IllegalAndRegulatedBehaviorsPolicy,
    HateOrAbusePolicy,
    ViolentSpeechPolicy,
    SuicideOrSelfHarmPolicy,
)
from grok_sampler.vision_sampler import VisionSampler
from grok_sampler.eapi_sampler import EapiSampler
from grox.data_loaders.data_types import (
    Post,
    SafetyPostAnnotations,
    ContentCategoryResult,
    ContentCategoryType,
    SafetyPtosViolatedPolicy,
    SafetyPolicy,
    SafetyPolicyCategory,
)
from grox.classifiers.content.classifier import ContentClassifier

logger = logging.getLogger(__name__)

_THINKING_RESTRICTION_LINES = {
    "",
    "",
}


def _strip_thinking_restrictions(text: str) -> str:
    lines = text.splitlines(keepends=True)
    return "".join(
        line for line in lines if line.strip() not in _THINKING_RESTRICTION_LINES
    ).lstrip("\n")


def _render_safety_ptos_for_reasoning() -> str:
    return _strip_thinking_restrictions(SafetyPtos().render())


class SafetyPtosCategoryClassifier(ContentClassifier):
    result_pattern = re.compile(r"(.*)<json>(.*)</json>", re.DOTALL)

    def __init__(
        self,
        model_name: ModelName = ModelName.VLM_SAFETY,
        deluxe: bool = False,
    ):
        self.deluxe = deluxe
        vlm_config = grox_config.get_model(model_name)
        vlm_config.temperature = 0.000001
        vlm = VisionSampler(GrokModelConfig(**vlm_config.model_dump()))
        super().__init__(categories=[ContentCategoryType.SAFETY_PTOS], llm=vlm)

    def build_convo(self, post: Post) -> Conversation:
        convo = Conversation(conversation_id=uuid.uuid4().hex)

        if self.deluxe:
            convo.messages.append(
                Message(role=Role.SYSTEM, content=[_render_safety_ptos_for_reasoning()])
            )
        else:
            convo.messages.append(
                Message(role=Role.SYSTEM, content=[SafetyPtos().render()])
            )

        user_msg = Message(role=Role.USER, content=[])
        user_msg.content.extend(UserRenderer.render(post.user))
        user_msg.content.extend(PostRenderer.render(post, include_reply_to=True))
        user_msg.content.append(
            f"\n\nAnalyze the post {post.id} and provide the requested JSON object for the post."
        )
        convo.messages.append(user_msg)

        if self.deluxe:
            convo.messages.append(Message(role=Role.ASSISTANT, content=[]))
        else:
            convo.messages.append(
                Message(role=Role.ASSISTANT, content=[""], separator="")
            )

        return convo

    async def classify_post(self, post: Post) -> SafetyPostAnnotations:
        convo = await self._to_convo(post)
        result = await self._sample(convo, post)
        mode = "deluxe" if self.deluxe else "standard"
        logger.info(
            f"safety ptos category classifier ({mode}) result for {post.id}: {result}"
        )

        match = self.result_pattern.search(result)
        if match:
            json_str = match.group(2).strip()
            return SafetyPostAnnotations.model_validate_json(json_str)
        else:
            raise ValueError(
                f"Invalid output for safety ptos category classifier ({mode}): {result}"
            )

    async def _to_convo(self, post: Post) -> Conversation:
        return self.build_convo(post)

    async def _sample(self, convo: Conversation, post: Post = None) -> str:
        return await self.llm.sample(
            convo.interleave(), conversation_id=convo.conversation_id
        )

    async def _parse(self, post: Post, output: str) -> List[ContentCategoryResult]:
        match = self.result_pattern.search(output)
        if match:
            return [
                ContentCategoryResult(
                    category=ContentCategoryType.SAFETY_PTOS, positive=True, score=0.0
                )
            ]
        else:
            mode = "deluxe" if self.deluxe else "standard"
            raise ValueError(
                f"Invalid parsing for safety ptos category classifier ({mode}): {output}"
            )


class SafetyPtosPolicyClassifier(ContentClassifier):
    result_pattern = re.compile(r"(.*)<json>(.*)</json>", re.DOTALL)

    def __init__(self, deluxe: bool = False):
        self.deluxe = deluxe

        vlm_config = grox_config.get_model(ModelName.VLM_PRIMARY_CRITICAL)
        vlm_config.temperature = 0.000001
        vlm = VisionSampler(GrokModelConfig(**vlm_config.model_dump()))
        super().__init__(categories=[ContentCategoryType.SAFETY_PTOS], llm=vlm)

        if deluxe:
            eapi_config_reasoning = grox_config.get_eapi_model(
                ModelName.EAPI_REASONING_INTERNAL
            )
            self.eapi_reasoning = EapiSampler(
                EapiModelConfig(**eapi_config_reasoning.model_dump())
            )

            eapi_config_reasoning_x_algo = grox_config.get_eapi_model(
                ModelName.EAPI_REASONING
            )
            self.eapi_reasoning_x_algo = EapiSampler(
                EapiModelConfig(**eapi_config_reasoning_x_algo.model_dump())
            )

    @staticmethod
    def _get_policy_prompt(violation: SafetyPtosViolatedPolicy) -> str:
        if violation.category == SafetyPolicyCategory.ViolentMedia:
            return ViolentMediaPolicy().render()
        elif violation.category == SafetyPolicyCategory.AdultContent:
            return AdultContentPolicy().render()
        elif violation.category == SafetyPolicyCategory.Spam:
            return SpamPolicy().render()
        elif violation.category == SafetyPolicyCategory.IllegalAndRegulatedBehaviors:
            return IllegalAndRegulatedBehaviorsPolicy().render()
        elif violation.category == SafetyPolicyCategory.HateOrAbuse:
            return HateOrAbusePolicy().render()
        elif violation.category == SafetyPolicyCategory.ViolentSpeech:
            return ViolentSpeechPolicy().render()
        elif violation.category == SafetyPolicyCategory.SuicideOrSelfHarm:
            return SuicideOrSelfHarmPolicy().render()
        else:
            raise ValueError(
                f"No policy prompt available for category: {violation.category.value}"
            )

    def build_convo(
        self, post: Post, violation: SafetyPtosViolatedPolicy
    ) -> Conversation:
        content = self._get_policy_prompt(violation)
        if self.deluxe:
            content = _strip_thinking_restrictions(content)

        convo = Conversation(conversation_id=uuid.uuid4().hex)
        convo.messages.append(Message(role=Role.SYSTEM, content=[content]))

        user_msg = Message(role=Role.USER, content=[])
        user_msg.content.extend(UserRenderer.render(post.user))
        user_msg.content.extend(PostRenderer.render(post, include_reply_to=True))
        user_msg.content.append(
            f"\n\nAnalyze the post {post.id} for the specific safety policy violation category: {violation.category.value}"
        )
        user_msg.content.append(
            f"\n\nProvide the requested JSON object for the specific safety policy type."
        )
        convo.messages.append(user_msg)

        if self.deluxe:
            convo.messages.append(Message(role=Role.ASSISTANT, content=[]))
        else:
            convo.messages.append(
                Message(role=Role.ASSISTANT, content=[""], separator="")
            )

        return convo

    SUPPORTED_POLICY_CATEGORIES = {
        SafetyPolicyCategory.ViolentMedia,
        SafetyPolicyCategory.AdultContent,
        SafetyPolicyCategory.Spam,
        SafetyPolicyCategory.IllegalAndRegulatedBehaviors,
        SafetyPolicyCategory.HateOrAbuse,
        SafetyPolicyCategory.ViolentSpeech,
        SafetyPolicyCategory.SuicideOrSelfHarm,
    }

    DELUXE_4_2_CATEGORIES = {
        SafetyPolicyCategory.AdultContent,
        SafetyPolicyCategory.ViolentMedia,
    }

    async def classify_policy_for_violation(
        self, post: Post, violation: SafetyPtosViolatedPolicy
    ) -> SafetyPolicy | None:

        if violation.category not in self.SUPPORTED_POLICY_CATEGORIES:
            return None

        convo = await self._to_convo(post, violation)

        if (
            and self.deluxe
            and violation.category in self.DELUXE_4_2_CATEGORIES
        ):
            mode = "deluxe-4.2"
            result = await self._sample_4_2(convo, post)
        else:
            mode = "deluxe" if self.deluxe else "standard"
            result = await self._sample(convo, post)

        logger.info(
            f"safety ptos policy classifier ({mode}) result for post {post.id}, violation {violation.category}: {result}"
        )

        match = self.result_pattern.search(result)
        if match:
            json_str = match.group(2).strip()
            return SafetyPolicy.model_validate_json(json_str)
        else:
            raise ValueError(
                f"Invalid output for safety ptos policy ({mode}): {result}"
            )

    async def _to_convo(
        self, post: Post, violation: SafetyPtosViolatedPolicy
    ) -> Conversation:
        return self.build_convo(post, violation)

    async def _sample_4_2(self, convo: Conversation, post: Post) -> str:
        try:
            return await self.eapi_reasoning_x_algo.sample(
                convo.interleaveToEapi(), conversation_id=convo.conversation_id
            )
        except Exception:
            logger.error(
                f"Failed to call 4.2 reasoning, error: {traceback.format_exc()}"
            )
            return await self.llm.sample(
                convo.interleave(), conversation_id=convo.conversation_id
            )

    async def _sample(self, convo: Conversation, post: Post = None) -> str:
        return await self.llm.sample(
            convo.interleave(), conversation_id=convo.conversation_id
        )

    async def _parse(self, post: Post, output: str) -> List[ContentCategoryResult]:
        return []
