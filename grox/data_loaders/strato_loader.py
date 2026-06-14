import asyncio
import logging

from grox.data_loaders.data_types import Post, User
from strato_http.queries.data_types import (
    ReplyRankingScore,
    ReplyRankingScoreKafka,
)
from strato_http.queries.content_understanding_author_metadata import (
    StratoContentUnderstandingAuthorMetadata,
)
from strato_http.queries.content_understanding_post_quote_metadata import (
    StratoContentUnderstandingPostQuoteMetadata,
)
from strato_http.queries.content_understanding_metadata_v2 import (
    StratoContentUnderstandingMetadataV2,
)
from strato_http.queries.reply_ranking_score import StratoReplyRankingScore
from strato_http.queries.reply_spam_annotation import StratoReplySpamAnnotation
from strato_http.queries.reply_ranking_score_kafka_v2 import (
    StratoReplyRankingScoreV2Kafka,
)
from strato_http.queries.safety_label import StratoSafetyLabel
from strato_http.queries.user_recent_posts import StratoUserRecentPosts
from grox.data_loaders.mappers.post_mapper import PostMapper


logger = logging.getLogger(__name__)


class TweetStratoLoader:
    content_understanding_metadata_strato = StratoContentUnderstandingMetadataV2()
    content_understanding_post_quote_metadata_strato = (
        StratoContentUnderstandingPostQuoteMetadata()
    )

    @classmethod
    async def load_post(
        cls, tweet_id: str, include_ancestors: bool = True
    ) -> Post | None:
        if include_ancestors:
            content_understanding_metadata = (
                await cls.content_understanding_metadata_strato.fetch(int(tweet_id))
            )
            if content_understanding_metadata:
                post = PostMapper.from_strato_content_understanding_metadata(
                    content_understanding_metadata
                )
                return post
        else:
            post_with_quote_metadata = (
                await cls.content_understanding_post_quote_metadata_strato.fetch(
                    int(tweet_id)
                )
            )
            if post_with_quote_metadata:
                post = PostMapper.from_strato_post_with_quote_metadata(
                    post_with_quote_metadata
                )
                return post
        return None


class UserStratoLoader:
    strato = StratoContentUnderstandingAuthorMetadata()

    @classmethod
    async def load_user(cls, user_id: int) -> User | None:
        strato_user = await cls.strato.fetch(user_id)
        if not strato_user:
            logger.warning(f"failed to hydrate user with {user_id=}, not found")
            return None
        return PostMapper._from_strato_user_metadata_to_user(strato_user)


class ReplyRankingScoreStratoLoader:
    strato = StratoReplyRankingScore()
    reply_ranking_v2_kafka_strato = StratoReplyRankingScoreV2Kafka()

    @classmethod
    async def save_reply_ranking_score(
        cls, post_id: str, reply_ranking_score: ReplyRankingScore
    ):
        await cls.strato.put(int(post_id), reply_ranking_score)

    @classmethod
    async def save_reply_ranking_kafka_v2(
        cls, post_id: str, reply_ranking_score_kafka: ReplyRankingScoreKafka
    ):
        await cls.reply_ranking_v2_kafka_strato.insert(
            int(post_id), reply_ranking_score_kafka
        )


class ReplySpamStratoLoader:
    strato = StratoReplySpamAnnotation()

    @classmethod
    async def save_spam_reply_annotation(
        cls, post_id: str, score: float, positive: bool, reason: str
    ):
        await cls.strato.put(int(post_id), score, positive, reason)


class UserRecentPostsLoader:
    recent_posts_strato = StratoUserRecentPosts()
    post_hydrator = StratoContentUnderstandingPostQuoteMetadata()
    safety_label = StratoSafetyLabel()

    @classmethod
    async def load(cls, user_id: int, limit: int = 10) -> list[Post]:
        res = await cls.recent_posts_strato.fetch(
            user_id, limit=limit, max_per_type=limit
        )
        if not res or "v" not in res:
            logger.warning(f"No recent posts found for {user_id=}")
            return []

        post_ids: list[int] = []
        for _post_type, posts in res["v"]:
            for post in posts:
                if _post_type == "TypeRetweet":
                    if "inReactionToPostId" in post:
                        post_ids.append(post["inReactionToPostId"])
                else:
                    if "postId" in post:
                        post_ids.append(post["postId"])

        if not post_ids:
            return []

        tasks = [cls.post_hydrator.fetch(post_id) for post_id in post_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        hydrated: list[Post] = []
        for post_id, result in zip(post_ids, results):
            if isinstance(result, Exception):
                logger.warning(
                    f"Failed to hydrate recent post {post_id} for {user_id=}: {result}"
                )
                continue
            if result is None:
                continue
            try:
                hydrated.append(PostMapper.from_strato_post_with_quote_metadata(result))
            except Exception:
                logger.warning(
                    f"Failed to map recent post {post_id} for {user_id=}", exc_info=True
                )

        for post in hydrated:
            post.safety_labels = await cls.safety_label.scan(post.id)

        return hydrated
