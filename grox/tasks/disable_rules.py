from abc import ABC

from grox.config.env import is_dev, is_prod, is_local, is_mm_emb_prod, is_ptos_prod
from grox.schedules.types import TaskContext


class DisableTaskRule(ABC):
    DISABLE_REASON: str = ""

    @classmethod
    def should_disable(cls, ctx: TaskContext) -> bool:
        return False

    @classmethod
    def disable_reason(cls) -> str | None:
        return cls.DISABLE_REASON


class DisableTaskForLocal(DisableTaskRule):
    DISABLE_REASON = "Task is disabled for local mode"

    @classmethod
    def should_disable(cls, ctx: TaskContext) -> bool:
        return is_local


class DisableTaskForDev(DisableTaskRule):
    DISABLE_REASON = "Task is disabled for dev mode"

    @classmethod
    def should_disable(cls, ctx: TaskContext) -> bool:
        return is_dev


class DisableTaskForNonProd(DisableTaskRule):
    DISABLE_REASON = "Task is disabled for non-prod mode"

    @classmethod
    def should_disable(cls, ctx: TaskContext) -> bool:
        return not is_prod


class DisableTaskForNonMmEmbProd(DisableTaskRule):
    DISABLE_REASON = "Task is disabled for non-mm-emb-prod mode"

    @classmethod
    def should_disable(cls, ctx: TaskContext) -> bool:
        return not is_mm_emb_prod


class DisableTaskForNonPtosProd(DisableTaskRule):
    DISABLE_REASON = "Task is disabled for non-ptos-prod mode"

    @classmethod
    def should_disable(cls, ctx: TaskContext) -> bool:
        return not is_ptos_prod
