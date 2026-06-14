import time
from enum import Enum
from typing import Any

from pydantic import Field, BaseModel
from grox.config.config import TaskGeneratorType
from grox.data_loaders.data_types import (
    Post,
    User,
    UserContext,
    ContentCategoryResult,
    GroxContentAnalysis,
    ReplyScoreResult,
    SafetyPostAnnotations,
)
from grox.classifiers.content.classifier_data_collection import (
    ClassifierDataCollectionResult,
)


class TaskEligibility(str, Enum):
    SPAM_COMMENT = "spam_comment"
    BANGER_INITIAL_SCREEN = "banger_initial_screen"
    POST_EMBEDDING_WITH_SUMMARY = "post_embedding_with_summary"
    POST_EMBEDDING_WITH_SUMMARY_FOR_REPLY = "post_embedding_with_summary_for_reply"
    MM_EMB_V4 = "mm_emb_v4"
    MM_EMB_V5 = "mm_emb_v5"
    MM_EMB_V5_FOR_REPLY = "mm_emb_v5_for_reply"
    REPLY_RANKING = "reply_ranking"
    SAFETY_PTOS = "safety_ptos"
    POST_SAFETY = "post_safety"


class TaskPayload(BaseModel):
    payload_id: str
    post: Post | None = None
    user: User | None = None
    user_context: UserContext | None = None
    attempt: int = 0
    eligibilities: set[TaskEligibility] = Field(default_factory=set)
    deadline_ts_secs: int | None = None
    task_type: TaskGeneratorType | None = None
    grox_content_analysis: GroxContentAnalysis | None = None

class TaskResult(BaseModel):
    task: TaskPayload
    task_started_at: float
    task_finished_at: float = Field(default_factory=time.perf_counter)
    content_categories: list[ContentCategoryResult] = Field(default_factory=list)
    multimodal_post_embedding: list[float] | None = None
    reason: str = ""
    success: bool = Field(default=True)
    error: str | None = Field(default=None)


class TaskContext:
    def __init__(self, task: TaskPayload):
        self.payload = task
        self.eligibilities: set[TaskEligibility] = task.eligibilities.copy()
        self.content_categories: list[ContentCategoryResult] = []
        self.summary: str = ""
        self.multimodal_post_embedding: list[float] | None = None
        self.multimodal_post_embedding_dict: dict[str, list[float]] = {}
        self.reply_ranking_results: list[ReplyScoreResult] = []
        self.reason: str = ""
        self.available_topics: list | None = None
        self.start_time = time.perf_counter()
        self.errors: list[Exception] = []
        self.safety_annotations: SafetyPostAnnotations | None = None


class TaskError(Exception):
    pass
