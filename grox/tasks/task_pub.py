import time
import logging
import traceback
from functools import cache

import thrifts.gen.twitter.strato.columns.content_understanding.content_understanding.ttypes as t
from thrifts.serdes import Serializer
from grox.tasks.task import Task
from monitor.metrics import Metrics
from grox.config.config import KafkaTopicName, grox_config
from kafka_cli.producer import KafkaProducer
from grox.schedules.types import ReplyScoreResult, TaskContext
from grox.tasks.disable_rules import (
    DisableTaskForDev,
    DisableTaskForLocal,
    DisableTaskForNonProd,
)
from strato_http.queries.unified_post_annotations import (
    StratoUnifiedPostAnnotations,
    StratoUpsertTweetBoolMetadataToUnifiedPostAnnotations,
)
from strato_http.queries.grok_reply_spam_action_with_labels import (
    StratoGrokReplySpamActionWithLabels,
)
from grox.data_loaders.data_types import (
    Post,
    ContentCategoryType,
    ContentCategoryResult,
    ContentCategoryScore,
    Image,
    Video,
)
from strato_http.queries.data_types import (
    EntityWithMetadata,
    FoundMetadata,
    UnifiedPostAnnotations,
    ReplyRankingScore,
    ReplyRankingScoreKafka,
    QualifiedId,
)
from grox.data_loaders.strato_loader import (
    ReplyRankingScoreStratoLoader,
    ReplySpamStratoLoader,
)


logger = logging.getLogger(__name__)


class TaskPublishKafka(Task):
    DISABLE_RULES = [DisableTaskForLocal, DisableTaskForDev]

    @classmethod
    async def _exec(cls, ctx: TaskContext) -> None:
        post = ctx.payload.post
        results = ctx.content_categories
        if not post:
            return

        if not results:
            Metrics.counter("task.publish_kafka.skipped.count").add(
                1, attributes={"reason": "no_results"}
            )
            return
        Metrics.counter("task.publish_kafka.intaken.count").add(1)
        try:
            await cls._publish_to_kafka(
                post.id,
                post.user.id if post.user else None,
                results,
                summary=ctx.summary,
                embedding=ctx.multimodal_post_embedding,
            )
            for result in results:
                Metrics.counter("task.publish_kafka.success.count").add(
                    1, attributes={"category": result.category.value}
                )
            if post.created_at:
                latency = time.time() - post.created_at.timestamp()
                for res in results:
                    Metrics.histogram("task.classification_e2e_latency").record(
                        latency, attributes={"category": res.category.value}
                    )
            else:
                Metrics.counter("task.publish_kafka.post_no_created_at.count").add(1)
        except Exception:
            Metrics.counter("task.publish_kafka.failed.count").add(1)
            logger.error(
                f"Failed to publish classification record: {traceback.format_exc()}"
            )
            raise

    @classmethod
    async def _publish_to_kafka(
        cls,
        post_id: str,
        user_id: int | None,
        results: list[ContentCategoryResult],
        summary: str,
        embedding: list[float] | None,
    ):
        category_results = [
            t.CategoryResult(
                category=r.category.name,
                positive=r.positive,
                score=r.score,
                summary=r.summary,
                taxonomyCategories=[
                    t.TaxonomyCategoryScore(id=tc.id, name=tc.name, score=tc.score)
                    for tc in r.taxonomy_categories
                ]
                if r.taxonomy_categories
                else None,
                keywords=None,
            )
            for r in results
        ]
        grox_content_analysis = t.GroxContentAnalysis(
            postId=int(post_id),
            userId=user_id,
            categoryResults=category_results,
            summary=summary,
            createdAt=int(time.time()),
        )
        serialized_bytes = Serializer.serialize(grox_content_analysis)
        await cls._get_kafka_producer().send(id=post_id, value=serialized_bytes)

    @classmethod
    @cache
    def _get_kafka_producer(cls):
        producer_config = grox_config.get_kafka_producer_topic(
            KafkaTopicName.GROX_CONTENT_ANALYSIS
        )
        logger.info(
            f"Creating kafka producer with config: {producer_config.model_dump()}"
        )
        return KafkaProducer(producer_config)


class TaskPublishUnifiedPostAnnotationsManhattan(Task):
    DISABLE_RULES = [DisableTaskForNonProd]

    @classmethod
    async def _exec(cls, ctx: TaskContext) -> None:
        Metrics.counter("task.publish_unified_post_annotations.count").add(1)
        results = ctx.content_categories
        if not results:
            logger.info("No unified post annotations to publish")
            return

        post = ctx.payload.post
        if not post:
            return

        grok_response = next(
            (
                r
                for r in results
                if r.category == ContentCategoryType.BANGER_INITIAL_SCREEN
            ),
            None,
        )
        if not grok_response:
            return

        if grok_response.slop_score is not None:
            if grok_response.slop_score == 1:
                Metrics.counter(
                    "task.publish_unified_post_annotations.slop_score_1.count"
                ).add(1)
            elif grok_response.slop_score == 2:
                Metrics.counter(
                    "task.publish_unified_post_annotations.slop_score_2.count"
                ).add(1)
            elif grok_response.slop_score == 3:
                Metrics.counter(
                    "task.publish_unified_post_annotations.slop_score_3.count"
                ).add(1)

        if grok_response.tweet_bool_metadata:
            if grok_response.tweet_bool_metadata.isHighQuality:
                Metrics.counter(
                    "task.publish_unified_post_annotations.is_high_quality_true.count"
                ).add(1)
            if grok_response.tweet_bool_metadata.isNsfw:
                Metrics.counter(
                    "task.publish_unified_post_annotations.is_nsfw_true.count"
                ).add(1)
                record_nsfw_detection(post)
            if grok_response.tweet_bool_metadata.isGore:
                Metrics.counter(
                    "task.publish_unified_post_annotations.is_gore_true.count"
                ).add(1)
            if grok_response.tweet_bool_metadata.isViolent:
                Metrics.counter(
                    "task.publish_unified_post_annotations.is_violent_true.count"
                ).add(1)
            if grok_response.tweet_bool_metadata.isSpam:
                Metrics.counter(
                    "task.publish_unified_post_annotations.is_spam_true.count"
                ).add(1)
            if grok_response.tweet_bool_metadata.isSoftNsfw:
                Metrics.counter(
                    "task.publish_unified_post_annotations.is_soft_nsfw_true.count"
                ).add(1)
            if grok_response.tweet_bool_metadata.isAdult:
                Metrics.counter(
                    "task.publish_unified_post_annotations.is_adult_true.count"
                ).add(1)

        if grok_response.tags and len(grok_response.tags) > 0:
            Metrics.counter(
                "task.publish_unified_post_annotations.tags_non_empty.count"
            ).add(1)

        if grok_response.is_image_editable_by_grok:
            Metrics.counter(
                "task.publish_unified_post_annotations.is_image_editable_by_grok_true.count"
            ).add(1)

        if post.media:
            if any(isinstance(m, Video) for m in post.media):
                Metrics.counter(
                    "task.publish_unified_post_annotations.has_video_true.count"
                ).add(1)
            if any(isinstance(m, Image) for m in post.media):
                Metrics.counter(
                    "task.publish_unified_post_annotations.has_image_true.count"
                ).add(1)

        resolved_grok_topics = []
        if grok_response.taxonomy_categories and ctx.available_topics:
            id_to_name = {}
            name_to_category_id = {}
            for category in ctx.available_topics:
                id_to_name[category.categoryEntityId] = category.categoryName
                name_to_category_id[category.categoryName] = category.categoryEntityId
                for sub in category.subtopics:
                    id_to_name[sub.topicEntityId] = sub.topicName
                    name_to_category_id[sub.topicName] = category.categoryEntityId

            topic_id_to_best_score = {}
            for grok_topic in grok_response.taxonomy_categories:
                topic_id = grok_topic.id
                if topic_id in id_to_name:
                    topic_name = id_to_name[topic_id]
                    category_id = name_to_category_id[topic_name]

                    resolved_grok_topic = ContentCategoryScore(
                        id=topic_id,
                        name=topic_name,
                        score=grok_topic.score,
                        category_id=category_id,
                    )
                    logger.info(
                        f"Validated grok_topic: ID {topic_id} -> '{topic_name}' (category_id: {category_id})"
                    )
                else:
                    logger.warning(
                        f"Invalid topic ID from Grok: {topic_id} not found in available topics"
                    )
                    Metrics.counter(
                        "task.publish_unified_post_annotations.invalid_grok_topic.count"
                    ).add(1)
                    continue

                if (
                    topic_id not in topic_id_to_best_score
                    or grok_topic.score > topic_id_to_best_score[topic_id].score
                ):
                    topic_id_to_best_score[topic_id] = resolved_grok_topic

            resolved_grok_topics = list(topic_id_to_best_score.values())
        elif grok_response.taxonomy_categories:
            logger.warning("No available topics to validate grok_topics")
            resolved_grok_topics = []

        for topic in resolved_grok_topics:
            sanitized_topic_name_for_metric = (
                topic.name.lower().replace(" ", "_").replace("&", "and")
            )
            Metrics.counter(
                f"task.publish_unified_post_annotations.topic_{sanitized_topic_name_for_metric}.count"
            ).add(1)

        entities = []
        if resolved_grok_topics and len(resolved_grok_topics) > 0:
            Metrics.counter(
                "task.publish_unified_post_annotations.with_grok_topics.count"
            ).add(1)
            entities = [
                EntityWithMetadata(
                    qualifiedId=QualifiedId(domainId=236, entityId=str(grok_topic.id)),
                    score=grok_topic.score,
                    categoryId=QualifiedId(
                        domainId=236, entityId=str(grok_topic.category_id)
                    )
                    if grok_topic.category_id
                    else None,
                )
                for grok_topic in resolved_grok_topics
            ]
        else:
            Metrics.counter(
                "task.publish_unified_post_annotations.with_empty_grok_topics.count"
            ).add(1)

        annotations = UnifiedPostAnnotations(
            tweetId=post.id,
            entities=entities,
            tags=[{"tag": tag, "score": 0.0} for tag in (grok_response.tags or [])],
            tweetBoolMetadata=grok_response.tweet_bool_metadata.model_dump()
            if grok_response.tweet_bool_metadata
            else None,
            description=grok_response.summary,
            isImageEditableByGrok=grok_response.is_image_editable_by_grok,
            slopScore=grok_response.slop_score,
            originalOcrText="",
            evergreenScore=None,
            hasVideo=post.media and any(isinstance(m, Video) for m in post.media),
            hasImage=post.media and any(isinstance(m, Image) for m in post.media),
            imageDescription=None,
            videoDescription=None,
            qualityScore=grok_response.score,
            hasMinorScore=grok_response.has_minor_score,
            hasCard=post.card is not None,
            foundMetadata=FoundMetadata(
                imageCount=sum(1 for m in post.media if isinstance(m, Image))
                if post.media
                else 0,
                videoCount=sum(1 for m in post.media if isinstance(m, Video))
                if post.media
                else 0,
                cardCount=1 if post.card else 0,
                cardV2Count=len(post.cardsV2) if post.cardsV2 else 0,
            ),
        )

        await StratoUnifiedPostAnnotations().put(int(post.id), annotations)
        Metrics.counter("task.publish_unified_post_annotations.success.count").add(1)


class TaskUpsertTweetBoolMetadataToUnifiedPostAnnotation(Task):
    DISABLE_RULES = [DisableTaskForNonProd]

    @classmethod
    async def _exec(cls, ctx: TaskContext) -> None:
        Metrics.counter(
            "task.upsert_tweet_bool_metadata_to_unified_post_annotations.count"
        ).add(1)
        results = ctx.content_categories
        if not results:
            logger.info("No unified post annotations to publish")
            return

        post = ctx.payload.post
        if not post:
            return

        grok_response = next(
            (
                r
                for r in results
                if r.category == ContentCategoryType.POST_SAFETY_SCREEN
            ),
            None,
        )
        if not grok_response or not grok_response.tweet_bool_metadata:
            return

        if grok_response.tweet_bool_metadata.isHighQuality:
            Metrics.counter(
                "task.upsert_tweet_bool_metadata_to_unified_post_annotations.is_high_quality_true.count"
            ).add(1)
        if grok_response.tweet_bool_metadata.isNsfw:
            Metrics.counter(
                "task.upsert_tweet_bool_metadata_to_unified_post_annotations.is_nsfw_true.count"
            ).add(1)
        if grok_response.tweet_bool_metadata.isGore:
            Metrics.counter(
                "task.upsert_tweet_bool_metadata_to_unified_post_annotations.is_gore_true.count"
            ).add(1)
        if grok_response.tweet_bool_metadata.isViolent:
            Metrics.counter(
                "task.upsert_tweet_bool_metadata_to_unified_post_annotations.is_violent_true.count"
            ).add(1)
        if grok_response.tweet_bool_metadata.isSpam:
            Metrics.counter(
                "task.upsert_tweet_bool_metadata_to_unified_post_annotations.is_spam_true.count"
            ).add(1)
        if grok_response.tweet_bool_metadata.isSoftNsfw:
            Metrics.counter(
                "task.upsert_tweet_bool_metadata_to_unified_post_annotations.is_soft_nsfw_true.count"
            ).add(1)
        if grok_response.tweet_bool_metadata.isAdult:
            Metrics.counter(
                "task.upsert_tweet_bool_metadata_to_unified_post_annotations.is_adult_true.count"
            ).add(1)

        await StratoUpsertTweetBoolMetadataToUnifiedPostAnnotations().put(
            int(post.id), grok_response.tweet_bool_metadata.model_dump()
        )
        Metrics.counter(
            "task.upsert_tweet_bool_metadata_to_unified_post_annotations.success.count"
        ).add(1)


class TaskWriteReplyRankingManhattan(Task):
    DISABLE_RULES = [DisableTaskForNonProd]

    _strato_grok_reply_spam_action_with_labels = StratoGrokReplySpamActionWithLabels()

    @classmethod
    async def _exec(cls, ctx: TaskContext) -> None:
        post = ctx.payload.post
        results = ctx.reply_ranking_results
        if not post:
            return
        if not results:
            Metrics.counter("task.write_reply_ranking_manhattan.skipped.count").add(
                1, attributes={"reason": "no_results"}
            )
            return
        Metrics.counter("task.write_reply_ranking_manhattan.intaken.count").add(1)
        try:
            await cls._publish_to_reply_ranking_manhattan(post, results)
            logger.info(
                f"Published reply ranking post to manhattan: {post.id=} {post.user.id=}"
            )
        except Exception:
            Metrics.counter("task.write_reply_ranking_manhattan.failed.count").add(1)
            logger.error(
                f"Failed to write reply ranking score to manhattan: {traceback.format_exc()}"
            )
            raise

    @classmethod
    async def _publish_to_reply_ranking_manhattan(
        cls, post: Post, results: list[ReplyScoreResult]
    ):
        logger.info(
            f"[_publish_to_reply_ranking_manhattan] checking results: {results}"
        )
        reasoning = ""

        try:
            reply_ranking_result = next(r for r in results)
        except:
            reply_ranking_result = None

        score = reply_ranking_result.score if reply_ranking_result else 3.0
        reasoning = reply_ranking_result.reason if reply_ranking_result else ""

        if post.user:
            logger.info(
                f"[_publish_to_reply_ranking_manhattan] {reasoning=} {post.id=} {post.user.id=} {score=}"
            )
        else:
            logger.info(
                f"Missing user id [_publish_to_reply_ranking_manhattan] {reasoning=} {post.id=} {score=}"
            )

        if score == 0.0:
            action_result = (
                await cls._strato_grok_reply_spam_action_with_labels.execute(
                    int(post.id)
                )
            )
            if action_result and len(action_result.applied_labels) > 0:
                logger.info(
                    f"grokReplySpamActionWithLabels applied labels: debugString='{action_result.debug_string}', appliedLabels={action_result.applied_labels} for post {post.id}"
                )
                Metrics.counter(
                    "task.grok_reply_spam_action_with_labels.applied.count"
                ).add(1)
            elif action_result:
                logger.info(
                    f"grokReplySpamActionWithLabels no labels applied: debugString='{action_result.debug_string}' for post {post.id}"
                )
                Metrics.counter(
                    "task.grok_reply_spam_action_with_labels.empty.count"
                ).add(1)
            else:
                logger.info(f"grokReplySpamActionWithLabels failed for post {post.id}")
                Metrics.counter(
                    "task.grok_reply_spam_action_with_labels.failed.count"
                ).add(1)

        await ReplyRankingScoreStratoLoader.save_reply_ranking_score(
            post_id=post.id,
            reply_ranking_score=ReplyRankingScore(
                score=score, reasoning=reasoning[-500:]
            ),
        )

        await ReplyRankingScoreStratoLoader.save_reply_ranking_kafka_v2(
            post_id=post.id,
            reply_ranking_score_kafka=ReplyRankingScoreKafka(
                postId=int(post.id), score=score, reasoning=reasoning[-500:]
            ),
        )

        Metrics.counter("task.write_reply_ranking_manhattan.success.count").add(
            1, attributes={"column": "reply_ranking"}
        )


class TaskWriteReplySpamManhattan(Task):
    DISABLE_RULES = [DisableTaskForNonProd]

    _strato_grok_reply_spam_action_with_labels = StratoGrokReplySpamActionWithLabels()

    @classmethod
    async def _exec(cls, ctx: TaskContext) -> None:
        post = ctx.payload.post
        if not post:
            return

        results = ctx.content_categories
        for result in results:
            if result.category == ContentCategoryType.SPAM_COMMENT:
                if result.positive:
                    action_result = (
                        await cls._strato_grok_reply_spam_action_with_labels.execute(
                            int(post.id)
                        )
                    )
                    if action_result and len(action_result.applied_labels) > 0:
                        logger.info(
                            f"grokReplySpamActionWithLabels applied labels: debugString='{action_result.debug_string}', appliedLabels={action_result.applied_labels} for post {post.id}"
                        )
                        Metrics.counter(
                            "task.grok_reply_spam_action_with_labels.applied.count"
                        ).add(1)
                    elif action_result:
                        logger.info(
                            f"grokReplySpamActionWithLabels no labels applied: debugString='{action_result.debug_string}' for post {post.id}"
                        )
                        Metrics.counter(
                            "task.grok_reply_spam_action_with_labels.empty.count"
                        ).add(1)
                    else:
                        logger.info(
                            f"grokReplySpamActionWithLabels failed for post {post.id}"
                        )
                        Metrics.counter(
                            "task.grok_reply_spam_action_with_labels.failed.count"
                        ).add(1)

                await ReplySpamStratoLoader.save_spam_reply_annotation(
                    post.id, result.score, result.positive, ""
                )
                logger.info(
                    f"Published reply spam annotation to manhattan: {post.id=} {post.user.id=}"
                )
