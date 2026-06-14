import uuid
import json
import logging
import re


import json_repair
from grox.data_loaders.media_loader import MediaLoader
from grox.lm.convo import Role, Message, Conversation
from grox.lm.thread import ThreadRenderer
from grox.config.config import ModelName, grox_config
from grok_sampler.config import GrokModelConfig
from grox.prompts.template import ReplyScoringSystem
from grok_sampler.vision_sampler import VisionSampler
from grox.data_loaders.data_types import (
    Post,
    ReplyScoreResult,
)
from monitor.metrics import Metrics
from pydantic import ValidationError
from grox.data_loaders.strato_loader import TweetStratoLoader

logger = logging.getLogger(__name__)


class ReplyScorer:
    model_name = "GROK"

    def __init__(self):
        vlm_config = grox_config.get_model(ModelName.VLM_MINI_CRITICAL)
        vlm_config.temperature = 0.000001
        self.vlm = VisionSampler(GrokModelConfig(**vlm_config.model_dump()))

        vlm_fallback_config = grox_config.get_model(
            ModelName.VLM_PRIMARY_CRITICAL
        )
        vlm_fallback_config.temperature = 0.000001
        self.vlm_fallback = VisionSampler(
            GrokModelConfig(**vlm_fallback_config.model_dump())
        )

    async def score(self, post: Post) -> list[ReplyScoreResult]:
        convo = await self._to_convo(post)
        result = await self._sample(convo, post)
        parsed = await self._parse(result)
        Metrics.histogram(
            "ranked_replies_scores",
            explicit_bucket_boundaries_advisory=[0.0, 1.0, 2.0, 3.0],
        ).record(parsed[0].score)

        return parsed

    async def _to_convo(self, post: Post, non_reasoning: bool = False) -> Conversation:
        convo = Conversation(conversation_id=uuid.uuid4().hex)
        system_prompt = ReplyScoringSystem().render(
            params={"large_account_follower_threshold": ""}
        )
        if non_reasoning:
            system_prompt = "" + system_prompt
        convo.messages.append(Message(role=Role.SYSTEM, content=[system_prompt]))
        convo.messages.append(
            ThreadRenderer.render(
                post, role=Role.HUMAN, separator="\n\n", include_signals=True
            )
        )
        if non_reasoning:
            convo.messages.append(
                Message(role=Role.ASSISTANT, content=[""], separator="")
            )
        else:
            convo.messages.append(Message(role=Role.ASSISTANT, content=[]))
        return convo

    async def _sample(self, convo: Conversation, post: Post) -> str:
        output = await self.vlm.sample(
            convo.interleave(),
            conversation_id=convo.conversation_id,
            json_schema=json.dumps(ReplyScoreResult.model_json_schema()),
        )
        match = re.search(r"\{.*\}", output, re.DOTALL)
        if not (match and "score" in match.group(0)):
            fallback_convo = await self._to_convo(post, non_reasoning=True)
            output = await self.vlm_fallback.sample(
                fallback_convo.interleave(),
                conversation_id=fallback_convo.conversation_id,
                json_schema=json.dumps(ReplyScoreResult.model_json_schema()),
            )
        return output

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

    async def _parse(self, output: str) -> list[ReplyScoreResult]:
        score = None
        reason = ""

        match = re.search(r"\{.*\}", output, re.DOTALL)
        if match and "score" in match.group(0):
            raw_result = match.group(0).strip()
        else:
            raw_result = output

        cleaned_result = await self._clean_output(raw_result)

        try:
            result = ReplyScoreResult.model_validate_json(cleaned_result)
            score = result.score
            reason = result.reason
        except (ValidationError, ValueError):
            try:
                repaired = json_repair.repair_json(cleaned_result, return_objects=True)
                if isinstance(repaired, dict) and "score" in repaired:
                    result = ReplyScoreResult.model_validate(repaired)
                    score = result.score
                    reason = result.reason
                    Metrics.counter("task.reply_ranker.json_repaired.count").add(1)
            except Exception:
                pass

            if score is None:
                score_match = re.search(r'"score":\s*([\d.]+)', cleaned_result)
                if score_match:
                    try:
                        score = float(score_match.group(1).strip())
                    except ValueError:
                        score = None
                        Metrics.counter("task.reply_ranker.invalid.count").add(
                            1,
                            attributes={
                                "filter": "reply_ranking",
                                "reason": "invalid_score_format",
                            },
                        )
            if not reason:
                reason_match = re.search(
                    r'"reason":\s*"((?:[^"\\]|\\.)*)"', cleaned_result, re.DOTALL
                )
                if reason_match:
                    reason = reason_match.group(1)

        if not score and score != 0:
            logger.error(f"Invalid output format: {output}")
            Metrics.counter("task.reply_ranker.invalid.count").add(
                1, attributes={"filter": "reply_ranking", "reason": "invalid_format"}
            )
            raise ValueError(f"Invalid output: {output}")

        return [ReplyScoreResult(score=score, reason=reason)]
