# Copyright 2026 X.AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from grok import (
    TransformerConfig,
    Transformer,
    layer_norm,
    right_anchored_rope_positions,
)

logger = logging.getLogger(__name__)


POST_AGE_MAX_MINUTES = 4800


def compute_post_age_bucket(
    impr_ts_sec: jax.Array,
    post_creation_ts_sec: jax.Array,
    granularity_mins: int = 60,
) -> jax.Array:
    """Compute post age buckets from impression and creation timestamps."""
    num_normal_buckets = POST_AGE_MAX_MINUTES // granularity_mins
    overflow_bucket = num_normal_buckets + 1

    post_age_minutes = (impr_ts_sec - post_creation_ts_sec) // 60

    bucket = (post_age_minutes // granularity_mins) + 1
    bucket = jnp.clip(bucket, 0, overflow_bucket)

    bucket = jnp.where(
        (post_age_minutes < 0) | (impr_ts_sec == 0) | (post_creation_ts_sec == 0),
        0,
        bucket,
    )
    return bucket.astype(jnp.int32)


@dataclass
class NormConfig:
    """Configuration for continuous value normalization."""

    norm_scale: float = 30.0
    use_log: bool = False


@dataclass
class ContinuousActionConfig:
    """Configuration for a single continuous action loss (e.g., dwell time)."""

    loss_weight: float = 0.0
    loss_type: str = "mae"
    tweedie_power: float = 1.5
    norm_config: NormConfig = None  # type: ignore

    def __post_init__(self):
        if self.norm_config is None:
            self.norm_config = NormConfig()


def normalize_continuous_value(
    values: jnp.ndarray,
    config: NormConfig,
) -> jnp.ndarray:
    """Normalize continuous values to 0-1 range."""
    values_clamped = jnp.clip(values, 0.0, config.norm_scale)

    if config.use_log:
        return jnp.log1p(values_clamped) / jnp.log1p(config.norm_scale)
    else:
        return values_clamped / config.norm_scale


@dataclass
class HashConfig:
    """Configuration for hash-based embeddings."""

    num_user_hashes: int = 2
    num_item_hashes: int = 2
    num_author_hashes: int = 2
    num_ip_hashes: int = 0


@dataclass
class RecsysEmbeddings:
    """Container for pre-looked-up embeddings from the embedding tables.

    These embeddings are looked up from hash tables before being passed to the model.
    The block_*_reduce functions will combine multiple hash embeddings into single representations.
    """

    user_embeddings: jax.typing.ArrayLike
    history_post_embeddings: jax.typing.ArrayLike
    candidate_post_embeddings: jax.typing.ArrayLike
    history_author_embeddings: jax.typing.ArrayLike
    candidate_author_embeddings: jax.typing.ArrayLike
    user_ip_embeddings: Optional[jax.typing.ArrayLike] = None


class RecsysModelOutput(NamedTuple):
    """Output of the recommendation model."""

    logits: jax.Array
    continuous_preds: Optional[jax.Array] = None


class RecsysBatch(NamedTuple):
    """Input batch for the recommendation model.

    Contains the feature data (hashes, actions, product surfaces) but NOT the embeddings.
    Embeddings are passed separately via RecsysEmbeddings.
    """

    user_hashes: jax.typing.ArrayLike
    history_post_hashes: jax.typing.ArrayLike
    history_author_hashes: jax.typing.ArrayLike
    history_actions: jax.typing.ArrayLike
    history_product_surface: jax.typing.ArrayLike
    candidate_post_hashes: jax.typing.ArrayLike
    candidate_author_hashes: jax.typing.ArrayLike
    candidate_product_surface: jax.typing.ArrayLike
    history_continuous_actions: Optional[jax.typing.ArrayLike] = None
    candidate_impr_ts: Optional[jax.typing.ArrayLike] = None
    candidate_post_creation_ts: Optional[jax.typing.ArrayLike] = None
    user_ip_hashes: Optional[jax.typing.ArrayLike] = None


def block_user_reduce(
    user_hashes: jnp.ndarray,
    user_embeddings: jnp.ndarray,
    num_user_hashes: int,
    emb_size: int,
    embed_init_scale: float = 1.0,
    *,
    user_ip_embeddings: Optional[jnp.ndarray] = None,
    num_ip_hashes: int = 0,
) -> Tuple[jax.Array, jax.Array]:
    """Combine multiple user hash embeddings into a single user representation.

    Args:
        user_hashes: [B, num_user_hashes] - hash values (0 = invalid/padding)
        user_embeddings: [B, num_user_hashes, D] - looked-up embeddings
        num_user_hashes: number of hash functions used
        emb_size: embedding dimension D
        embed_init_scale: initialization scale for projection
        user_ip_embeddings: optional [B, num_ip_hashes, D] IP address embeddings
        num_ip_hashes: number of IP hash functions (0 = disabled)

    Returns:
        user_embedding: [B, 1, D] - combined user embedding
        user_padding_mask: [B, 1] - True where user is valid
    """
    B = user_embeddings.shape[0]
    D = emb_size

    user_embedding = user_embeddings.reshape((B, 1, num_user_hashes * D))

    embed_init = hk.initializers.VarianceScaling(embed_init_scale, mode="fan_out")
    proj_mat_1 = hk.get_parameter(
        "proj_mat_1",
        [num_user_hashes * D, D],
        dtype=jnp.float32,
        init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
    )

    user_embedding = jnp.dot(user_embedding.astype(proj_mat_1.dtype), proj_mat_1).astype(
        user_embeddings.dtype
    )

    # hash 0 is reserved for padding)
    if user_ip_embeddings is not None and num_ip_hashes > 0:
        ip_emb = user_ip_embeddings.reshape((B, num_ip_hashes, D))
        ip_emb = jnp.sum(ip_emb, axis=1, keepdims=True)  # [B, 1, D]
        user_embedding = user_embedding + ip_emb

    user_padding_mask = (user_hashes[:, 0] != 0).reshape(B, 1).astype(jnp.bool_)

    return user_embedding, user_padding_mask


def block_history_reduce(
    history_post_hashes: jnp.ndarray,
    history_post_embeddings: jnp.ndarray,
    history_author_embeddings: jnp.ndarray,
    history_product_surface_embeddings: jnp.ndarray,
    history_actions_embeddings: jnp.ndarray,
    num_item_hashes: int,
    num_author_hashes: int,
    embed_init_scale: float = 1.0,
    *,
    history_continuous_embeddings: Optional[jnp.ndarray] = None,
    history_post_age_embeddings: Optional[jnp.ndarray] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Combine history embeddings (post, author, actions, product_surface, ...) into sequence.

    Args:
        history_post_hashes: [B, S, num_item_hashes]
        history_post_embeddings: [B, S, num_item_hashes, D]
        history_author_embeddings: [B, S, num_author_hashes, D]
        history_product_surface_embeddings: [B, S, D]
        history_actions_embeddings: [B, S, D]
        num_item_hashes: number of hash functions for items
        num_author_hashes: number of hash functions for authors
        embed_init_scale: initialization scale
        history_continuous_embeddings: optional [B, S, D] continuous action embeddings
        history_post_age_embeddings: optional [B, S, D] post age embeddings

    Returns:
        history_embeddings: [B, S, D]
        history_padding_mask: [B, S]
    """
    B, S, _, D = history_post_embeddings.shape

    history_post_embeddings_reshaped = history_post_embeddings.reshape((B, S, num_item_hashes * D))
    history_author_embeddings_reshaped = history_author_embeddings.reshape(
        (B, S, num_author_hashes * D)
    )

    parts = [
        history_post_embeddings_reshaped,
        history_author_embeddings_reshaped,
        history_actions_embeddings,
        history_product_surface_embeddings,
    ]

    if history_continuous_embeddings is not None:
        parts.append(history_continuous_embeddings)
    if history_post_age_embeddings is not None:
        parts.append(history_post_age_embeddings)

    post_author_embedding = jnp.concatenate(parts, axis=-1)

    embed_init = hk.initializers.VarianceScaling(embed_init_scale, mode="fan_out")
    proj_mat_3 = hk.get_parameter(
        "proj_mat_3",
        [post_author_embedding.shape[-1], D],
        dtype=jnp.float32,
        init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
    )

    history_embedding = jnp.dot(post_author_embedding.astype(proj_mat_3.dtype), proj_mat_3).astype(
        post_author_embedding.dtype
    )

    history_embedding = history_embedding.reshape(B, S, D)

    history_padding_mask = (history_post_hashes[:, :, 0] != 0).reshape(B, S)

    return history_embedding, history_padding_mask


def block_candidate_reduce(
    candidate_post_hashes: jnp.ndarray,
    candidate_post_embeddings: jnp.ndarray,
    candidate_author_embeddings: jnp.ndarray,
    candidate_product_surface_embeddings: jnp.ndarray,
    num_item_hashes: int,
    num_author_hashes: int,
    embed_init_scale: float = 1.0,
    *,
    candidate_post_age_embeddings: Optional[jnp.ndarray] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Combine candidate embeddings (post, author, product_surface, ...) into sequence.

    Args:
        candidate_post_hashes: [B, C, num_item_hashes]
        candidate_post_embeddings: [B, C, num_item_hashes, D]
        candidate_author_embeddings: [B, C, num_author_hashes, D]
        candidate_product_surface_embeddings: [B, C, D]
        num_item_hashes: number of hash functions for items
        num_author_hashes: number of hash functions for authors
        embed_init_scale: initialization scale
        candidate_post_age_embeddings: optional [B, C, D] post age embeddings

    Returns:
        candidate_embeddings: [B, C, D]
        candidate_padding_mask: [B, C]
    """
    B, C, _, D = candidate_post_embeddings.shape

    candidate_post_embeddings_reshaped = candidate_post_embeddings.reshape(
        (B, C, num_item_hashes * D)
    )
    candidate_author_embeddings_reshaped = candidate_author_embeddings.reshape(
        (B, C, num_author_hashes * D)
    )

    parts = [
        candidate_post_embeddings_reshaped,
        candidate_author_embeddings_reshaped,
        candidate_product_surface_embeddings,
    ]

    if candidate_post_age_embeddings is not None:
        parts.append(candidate_post_age_embeddings)

    post_author_embedding = jnp.concatenate(parts, axis=-1)

    embed_init = hk.initializers.VarianceScaling(embed_init_scale, mode="fan_out")
    proj_mat_2 = hk.get_parameter(
        "proj_mat_2",
        [post_author_embedding.shape[-1], D],
        dtype=jnp.float32,
        init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
    )

    candidate_embedding = jnp.dot(
        post_author_embedding.astype(proj_mat_2.dtype), proj_mat_2
    ).astype(post_author_embedding.dtype)

    candidate_padding_mask = (candidate_post_hashes[:, :, 0] != 0).reshape(B, C).astype(jnp.bool_)

    return candidate_embedding, candidate_padding_mask


@dataclass
class PhoenixModelConfig:
    """Configuration for the recommendation system model."""

    model: TransformerConfig
    emb_size: int
    num_actions: int
    history_seq_len: int = 128
    candidate_seq_len: int = 32

    name: Optional[str] = None
    fprop_dtype: Any = jnp.bfloat16

    hash_config: HashConfig = None  # type: ignore

    product_surface_vocab_size: int = 16

    post_age_granularity_mins: int = 60

    num_continuous_actions: int = 8

    continuous_action_hidden_dim: int = 64

    continuous_action_config: ContinuousActionConfig = None  # type: ignore

    use_ip_address: bool = False

    right_anchored_rope: bool = False

    mask_neg_feedback_on_negatives: bool = True

    _initialized = False

    def __post_init__(self):
        if self.hash_config is None:
            self.hash_config = HashConfig()
        if self.continuous_action_config is None:
            self.continuous_action_config = ContinuousActionConfig()

    @property
    def post_age_vocab_size(self) -> int:
        """Derived vocab size for post age buckets: num_normal + overflow + missing."""
        return (POST_AGE_MAX_MINUTES // self.post_age_granularity_mins) + 2

    def initialize(self):
        self._initialized = True
        return self

    def make(self):
        if not self._initialized:
            logger.warning(f"PhoenixModel {self.name} is not initialized. Initializing.")
            self.initialize()

        return PhoenixModel(
            model=self.model.make(),
            config=self,
            fprop_dtype=self.fprop_dtype,
        )


@dataclass
class PhoenixModel(hk.Module):
    """A transformer-based recommendation model for ranking candidates."""

    model: Transformer
    config: PhoenixModelConfig
    fprop_dtype: Any = jnp.bfloat16
    name: Optional[str] = None

    def _get_action_embeddings(
        self,
        actions: jax.Array,
    ) -> jax.Array:
        """Convert multi-hot action vectors to embeddings.

        Uses a learned projection matrix to map the signed action vector
        to the embedding dimension. This works for any number of actions.
        """
        config = self.config
        _, _, num_actions = actions.shape
        D = config.emb_size

        embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")
        action_projection = hk.get_parameter(
            "action_projection",
            [num_actions, D],
            dtype=jnp.float32,
            init=embed_init,
        )

        actions_signed = (2 * actions - 1).astype(jnp.float32)

        action_emb = jnp.dot(actions_signed.astype(action_projection.dtype), action_projection)

        valid_mask = jnp.any(actions, axis=-1, keepdims=True)
        action_emb = action_emb * valid_mask

        return action_emb.astype(self.fprop_dtype)

    def _single_hot_to_embeddings(
        self,
        input: jax.Array,
        vocab_size: int,
        emb_size: int,
        name: str,
    ) -> jax.Array:
        """Convert single-hot indices to embeddings via lookup table.

        Args:
            input: [B, S] tensor of categorical indices
            vocab_size: size of the vocabulary
            emb_size: embedding dimension
            name: name for the embedding table parameter

        Returns:
            embeddings: [B, S, emb_size]
        """
        embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")
        embedding_table = hk.get_parameter(
            name,
            [vocab_size, emb_size],
            dtype=jnp.float32,
            init=embed_init,
        )

        input_one_hot = jax.nn.one_hot(input, vocab_size)
        output = jnp.dot(input_one_hot, embedding_table)
        return output.astype(self.fprop_dtype)

    def _get_unembedding(self) -> jax.Array:
        """Get the unembedding matrix for decoding to discrete action logits."""
        config = self.config
        embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")
        unembed_mat = hk.get_parameter(
            "unembeddings",
            [config.emb_size, config.num_actions],
            dtype=jnp.float32,
            init=embed_init,
        )
        return unembed_mat

    def _get_continuous_head(self) -> jax.Array:
        config = self.config
        embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")
        continuous_mat = hk.get_parameter(
            "continuous_unembeddings",
            [config.emb_size, config.num_continuous_actions],
            dtype=jnp.float32,
            init=embed_init,
        )
        return continuous_mat

    def _project_continuous_value_to_embedding(
        self,
        values: jnp.ndarray,
        D: int,
        param_name: str,
        norm_config: NormConfig,
        hidden_dim: int = 64,
    ) -> jax.Array:
        values_normalized = normalize_continuous_value(values, norm_config)
        values_expanded = values_normalized[..., None]  # [B, seq_len, 1]

        embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")

        proj1 = hk.get_parameter(
            f"{param_name}_proj1",
            [1, hidden_dim],
            dtype=jnp.float32,
            init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
        )
        hidden = jnp.dot(values_expanded.astype(proj1.dtype), proj1)

        hidden = jax.nn.gelu(hidden)

        proj2 = hk.get_parameter(
            f"{param_name}_proj2",
            [hidden_dim, D],
            dtype=jnp.float32,
            init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
        )
        embedding = jnp.dot(hidden, proj2)

        return embedding.astype(self.fprop_dtype)

    def build_inputs(
        self,
        batch: RecsysBatch,
        recsys_embeddings: RecsysEmbeddings,
    ) -> Tuple[jax.Array, jax.Array, int]:
        """Build input embeddings from batch and pre-looked-up embeddings.

        Args:
            batch: RecsysBatch containing hashes, actions, product surfaces
            recsys_embeddings: RecsysEmbeddings containing pre-looked-up embeddings

        Returns:
            embeddings: [B, 1 + history_len + num_candidates, D]
            padding_mask: [B, 1 + history_len + num_candidates]
            candidate_start_offset: int - position where candidates start
        """
        config = self.config
        hash_config = config.hash_config

        history_product_surface_embeddings = self._single_hot_to_embeddings(
            batch.history_product_surface,  # type: ignore
            config.product_surface_vocab_size,
            config.emb_size,
            "product_surface_embedding_table",
        )
        candidate_product_surface_embeddings = self._single_hot_to_embeddings(
            batch.candidate_product_surface,  # type: ignore
            config.product_surface_vocab_size,
            config.emb_size,
            "product_surface_embedding_table",
        )

        history_actions_embeddings = self._get_action_embeddings(batch.history_actions)  # type: ignore

        B_size = batch.history_product_surface.shape[0]  # type: ignore
        S_size = batch.history_product_surface.shape[1]  # type: ignore
        if batch.history_continuous_actions is not None:
            dwell_values = batch.history_continuous_actions[:, :, 1]  # index 1 = dwell_time
        else:
            dwell_values = jnp.zeros((B_size, S_size), dtype=jnp.float32)
        history_continuous_embeddings = self._project_continuous_value_to_embedding(
            dwell_values,
            config.emb_size,
            "history_dwell_time",
            config.continuous_action_config.norm_config,
            config.continuous_action_hidden_dim,
        )

        user_embeddings, user_padding_mask = block_user_reduce(
            batch.user_hashes,  # type: ignore
            recsys_embeddings.user_embeddings,  # type: ignore
            hash_config.num_user_hashes,
            config.emb_size,
            1.0,
            user_ip_embeddings=recsys_embeddings.user_ip_embeddings,
            num_ip_hashes=hash_config.num_ip_hashes,
        )

        history_embeddings, history_padding_mask = block_history_reduce(
            batch.history_post_hashes,  # type: ignore
            recsys_embeddings.history_post_embeddings,  # type: ignore
            recsys_embeddings.history_author_embeddings,  # type: ignore
            history_product_surface_embeddings,
            history_actions_embeddings,
            hash_config.num_item_hashes,
            hash_config.num_author_hashes,
            1.0,
            history_continuous_embeddings=history_continuous_embeddings,
        )

        C_size = batch.candidate_product_surface.shape[1]  # type: ignore
        if batch.candidate_impr_ts is not None and batch.candidate_post_creation_ts is not None:
            post_age_buckets = compute_post_age_bucket(
                batch.candidate_impr_ts,
                batch.candidate_post_creation_ts,
                config.post_age_granularity_mins,
            )
        else:
            post_age_buckets = jnp.zeros((B_size, C_size), dtype=jnp.int32)
        candidate_post_age_embeddings = self._single_hot_to_embeddings(
            post_age_buckets,
            config.post_age_vocab_size,
            config.emb_size,
            "post_age_embedding_table",
        )

        candidate_embeddings, candidate_padding_mask = block_candidate_reduce(
            batch.candidate_post_hashes,  # type: ignore
            recsys_embeddings.candidate_post_embeddings,  # type: ignore
            recsys_embeddings.candidate_author_embeddings,  # type: ignore
            candidate_product_surface_embeddings,
            hash_config.num_item_hashes,
            hash_config.num_author_hashes,
            1.0,
            candidate_post_age_embeddings=candidate_post_age_embeddings,
        )

        embeddings = jnp.concatenate(
            [user_embeddings, history_embeddings, candidate_embeddings], axis=1
        )
        padding_mask = jnp.concatenate(
            [user_padding_mask, history_padding_mask, candidate_padding_mask], axis=1
        )

        candidate_start_offset = user_padding_mask.shape[1] + history_padding_mask.shape[1]

        return embeddings.astype(self.fprop_dtype), padding_mask, candidate_start_offset

    def __call__(
        self,
        batch: RecsysBatch,
        recsys_embeddings: RecsysEmbeddings,
    ) -> RecsysModelOutput:
        """Forward pass for ranking candidates.

        Args:
            batch: RecsysBatch containing hashes, actions, product surfaces
            recsys_embeddings: RecsysEmbeddings containing pre-looked-up embeddings

        Returns:
            RecsysModelOutput containing:
                - logits: [B, num_candidates, num_actions] discrete engagement logits
                - continuous_preds: [B, num_candidates, num_continuous] continuous predictions
        """
        embeddings, padding_mask, candidate_start_offset = self.build_inputs(
            batch, recsys_embeddings
        )

        positions = None
        if self.config.right_anchored_rope:
            positions = right_anchored_rope_positions(
                padding_mask,
                history_seq_len=self.config.history_seq_len,
                num_user_prefix_tokens=1,
            )

        # transformer
        model_output = self.model(
            embeddings,
            padding_mask,
            candidate_start_offset=candidate_start_offset,
            positions=positions,
        )

        out_embeddings = model_output.embeddings

        out_embeddings = layer_norm(out_embeddings)

        candidate_embeddings = out_embeddings[:, candidate_start_offset:, :]

        unembeddings = self._get_unembedding()
        logits = jnp.dot(candidate_embeddings.astype(unembeddings.dtype), unembeddings)
        logits = logits.astype(self.fprop_dtype)

        continuous_mat = self._get_continuous_head()
        continuous_logits = jnp.dot(
            candidate_embeddings.astype(continuous_mat.dtype), continuous_mat
        )
        continuous_preds = jax.nn.sigmoid(continuous_logits).astype(self.fprop_dtype)

        return RecsysModelOutput(logits=logits, continuous_preds=continuous_preds)
