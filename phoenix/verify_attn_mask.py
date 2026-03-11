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

"""
Attention Mask Verification: Candidate Isolation in PhoenixModel

Proves that candidate_A's output logits depend only on (user, history, candidate_A)
and never on what other candidates are present in the same batch.

This holds because make_recsys_attn_mask constructs a mask applied as:
    attn_logits = jnp.where(mask, attn_logits, -1e30)  [grok.py:353]
before softmax — structurally preventing any learned weight from overriding it.

Architecture note: PhoenixModel uses a SINGLE joint forward pass over the full
sequence [user (1 token), history (S tokens), candidates (C tokens)]. This makes
the experiment non-trivial: we are testing a structural property of the mask, not
a trivial separation of passes.

Three experiments, escalating in strength:
  Exp 1 — Different co-candidates:     A stays same, B/C/D change
  Exp 2 — Different sequence lengths:  C=1 vs C=8 (strongest test)
  Exp 3 — Extreme co-candidate values: zeros vs max-float embeddings
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

from grok import TransformerConfig
from recsys_model import PhoenixModelConfig, HashConfig, RecsysBatch, RecsysEmbeddings
from runners import ACTIONS

# ---------------------------------------------------------------------------
# Constants (match run_ranker.py demo config exactly)
# ---------------------------------------------------------------------------
EMB_SIZE = 128
NUM_ACTIONS = len(ACTIONS)          # 19
HISTORY_SEQ_LEN = 32
CANDIDATE_SEQ_LEN = 8               # max; used for param initialization
NUM_USER_HASHES = 2
NUM_ITEM_HASHES = 2
NUM_AUTHOR_HASHES = 2
PRODUCT_SURFACE_VOCAB_SIZE = 16


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def build_config() -> PhoenixModelConfig:
    config = PhoenixModelConfig(
        emb_size=EMB_SIZE,
        num_actions=NUM_ACTIONS,
        history_seq_len=HISTORY_SEQ_LEN,
        candidate_seq_len=CANDIDATE_SEQ_LEN,
        hash_config=HashConfig(
            num_user_hashes=NUM_USER_HASHES,
            num_item_hashes=NUM_ITEM_HASHES,
            num_author_hashes=NUM_AUTHOR_HASHES,
        ),
        product_surface_vocab_size=PRODUCT_SURFACE_VOCAB_SIZE,
        model=TransformerConfig(
            emb_size=EMB_SIZE,
            widening_factor=2,
            key_size=64,
            num_q_heads=2,
            num_kv_heads=2,
            num_layers=2,
            attn_output_multiplier=0.125,
        ),
    )
    config.initialize()
    return config


config = build_config()


def _forward(batch: RecsysBatch, embeddings: RecsysEmbeddings):
    return config.make()(batch, embeddings)


forward_fn = hk.without_apply_rng(hk.transform(_forward))


# ---------------------------------------------------------------------------
# Initialize params with a fixed seed and dummy data (C=CANDIDATE_SEQ_LEN=8)
# Params do NOT depend on C — same params apply to any number of candidates.
# ---------------------------------------------------------------------------

def _zeros_batch(num_candidates: int) -> RecsysBatch:
    return RecsysBatch(
        user_hashes=np.zeros((1, NUM_USER_HASHES), dtype=np.int32),
        history_post_hashes=np.zeros((1, HISTORY_SEQ_LEN, NUM_ITEM_HASHES), dtype=np.int32),
        history_author_hashes=np.zeros((1, HISTORY_SEQ_LEN, NUM_AUTHOR_HASHES), dtype=np.int32),
        history_actions=np.zeros((1, HISTORY_SEQ_LEN, NUM_ACTIONS), dtype=np.float32),
        history_product_surface=np.zeros((1, HISTORY_SEQ_LEN), dtype=np.int32),
        candidate_post_hashes=np.zeros((1, num_candidates, NUM_ITEM_HASHES), dtype=np.int32),
        candidate_author_hashes=np.zeros((1, num_candidates, NUM_AUTHOR_HASHES), dtype=np.int32),
        candidate_product_surface=np.zeros((1, num_candidates), dtype=np.int32),
    )


def _zeros_embeddings(num_candidates: int) -> RecsysEmbeddings:
    return RecsysEmbeddings(
        user_embeddings=np.zeros((1, NUM_USER_HASHES, EMB_SIZE), dtype=np.float32),
        history_post_embeddings=np.zeros((1, HISTORY_SEQ_LEN, NUM_ITEM_HASHES, EMB_SIZE), dtype=np.float32),
        candidate_post_embeddings=np.zeros((1, num_candidates, NUM_ITEM_HASHES, EMB_SIZE), dtype=np.float32),
        history_author_embeddings=np.zeros((1, HISTORY_SEQ_LEN, NUM_AUTHOR_HASHES, EMB_SIZE), dtype=np.float32),
        candidate_author_embeddings=np.zeros((1, num_candidates, NUM_AUTHOR_HASHES, EMB_SIZE), dtype=np.float32),
    )


# Initialize params structure (haiku initializes Linear/RMSNorm with Constant(0) by design).
# We then replace all leaves with random non-trivial values using a fixed seed, so that
# the model produces real non-zero outputs and the experiment is meaningful.
# The isolation claim holds for ANY weight configuration — using random weights is
# strictly more convincing than the degenerate zero case.
rng = jax.random.PRNGKey(42)
rng, init_rng = jax.random.split(rng)
_params_zero = forward_fn.init(init_rng, _zeros_batch(CANDIDATE_SEQ_LEN), _zeros_embeddings(CANDIDATE_SEQ_LEN))

def _perturb_params(params, seed: int = 123, scale: float = 0.1):
    """Replace zero-initialized params with random values, preserving shapes/dtypes."""
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    new_leaves = [
        (jax.random.normal(k, shape=leaf.shape) * scale).astype(leaf.dtype)
        for k, leaf in zip(keys, leaves)
    ]
    return jax.tree_util.tree_unflatten(treedef, new_leaves)

PARAMS = _perturb_params(_params_zero, seed=123, scale=0.1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_forward(batch: RecsysBatch, embeddings: RecsysEmbeddings) -> jax.Array:
    """Return logits [B, C, num_actions]."""
    return forward_fn.apply(PARAMS, batch, embeddings).logits


def assemble(
    user_h, user_e,
    hist_ph, hist_pe, hist_ah, hist_ae, hist_act, hist_ps,
    cand_ph, cand_pe, cand_ah, cand_ae, cand_ps,
):
    batch = RecsysBatch(
        user_hashes=user_h,
        history_post_hashes=hist_ph,
        history_author_hashes=hist_ah,
        history_actions=hist_act,
        history_product_surface=hist_ps,
        candidate_post_hashes=cand_ph,
        candidate_author_hashes=cand_ah,
        candidate_product_surface=cand_ps,
    )
    embeddings = RecsysEmbeddings(
        user_embeddings=user_e,
        history_post_embeddings=hist_pe,
        candidate_post_embeddings=cand_pe,
        history_author_embeddings=hist_ae,
        candidate_author_embeddings=cand_ae,
    )
    return batch, embeddings


def random_candidates(n: int, rng_seed: int):
    """Return (post_hashes, post_embs, author_hashes, author_embs, product_surface) for n candidates."""
    r = np.random.default_rng(rng_seed)
    return (
        r.integers(1, 100_000, size=(1, n, NUM_ITEM_HASHES)).astype(np.int32),
        r.standard_normal((1, n, NUM_ITEM_HASHES, EMB_SIZE)).astype(np.float32),
        r.integers(1, 100_000, size=(1, n, NUM_AUTHOR_HASHES)).astype(np.int32),
        r.standard_normal((1, n, NUM_AUTHOR_HASHES, EMB_SIZE)).astype(np.float32),
        r.integers(0, PRODUCT_SURFACE_VOCAB_SIZE, size=(1, n)).astype(np.int32),
    )


def cat_candidates(*tuples):
    """Concatenate candidate tuples along the candidate axis (axis=1)."""
    return tuple(np.concatenate([t[i] for t in tuples], axis=1) for i in range(5))


# ---------------------------------------------------------------------------
# Fixed user + history (byte-for-byte identical across all experiments)
# ---------------------------------------------------------------------------

_rng = np.random.default_rng(0)

USER_H  = _rng.integers(1, 100_000, size=(1, NUM_USER_HASHES)).astype(np.int32)
USER_E  = _rng.standard_normal((1, NUM_USER_HASHES, EMB_SIZE)).astype(np.float32)

HIST_PH = _rng.integers(1, 100_000, size=(1, HISTORY_SEQ_LEN, NUM_ITEM_HASHES)).astype(np.int32)
HIST_PH[0, 20:, :] = 0   # first 20 history items are valid; rest are padding
HIST_PE = _rng.standard_normal((1, HISTORY_SEQ_LEN, NUM_ITEM_HASHES, EMB_SIZE)).astype(np.float32)
HIST_AH = _rng.integers(1, 100_000, size=(1, HISTORY_SEQ_LEN, NUM_AUTHOR_HASHES)).astype(np.int32)
HIST_AH[0, 20:, :] = 0
HIST_AE = _rng.standard_normal((1, HISTORY_SEQ_LEN, NUM_AUTHOR_HASHES, EMB_SIZE)).astype(np.float32)
HIST_ACT = (_rng.random((1, HISTORY_SEQ_LEN, NUM_ACTIONS)) > 0.7).astype(np.float32)
HIST_PS = _rng.integers(0, PRODUCT_SURFACE_VOCAB_SIZE, size=(1, HISTORY_SEQ_LEN)).astype(np.int32)

# Fixed candidate A
CAND_A = random_candidates(1, rng_seed=42)


def logits_A(cand_tuple):
    """Run forward pass and return candidate A's logits [num_actions]."""
    batch, embs = assemble(
        USER_H, USER_E,
        HIST_PH, HIST_PE, HIST_AH, HIST_AE, HIST_ACT, HIST_PS,
        *cand_tuple,
    )
    return run_forward(batch, embs)[0, 0, :]   # [num_actions]


def report(name: str, la: jax.Array, lb: jax.Array):
    max_diff = float(jnp.max(jnp.abs(la - lb)))
    passed   = bool(jnp.allclose(la, lb, atol=1e-4))
    print(f"\n  Logits A (v1): {np.array(la).round(4)}")
    print(f"  Logits A (v2): {np.array(lb).round(4)}")
    print(f"  max|Δ| = {max_diff:.6f}  |  allclose(atol=1e-4) = {passed}")
    return max_diff, passed


# ===========================================================================
# Experiment 1 — Different co-candidates (core test)
#
# Batch 1: (user_u, history_H, [candidate_A, B, C, D])   — B,C,D from seed 100
# Batch 2: (user_u, history_H, [candidate_A, D, E, F])   — D,E,F from seed 200
# A, user, history are byte-for-byte identical.
# ===========================================================================

print("=" * 70)
print("ATTENTION MASK VERIFICATION — CANDIDATE ISOLATION EXPERIMENTS")
print("=" * 70)

print("\n─── Experiment 1: Different Co-Candidates in Batch ───────────────────")

co_1 = random_candidates(3, rng_seed=100)
co_2 = random_candidates(3, rng_seed=200)

la_1 = logits_A(cat_candidates(CAND_A, co_1))
lb_1 = logits_A(cat_candidates(CAND_A, co_2))
diff_1, pass_1 = report("Exp1", la_1, lb_1)


# ===========================================================================
# Experiment 2 — Different number of candidates (sequence-length test)
#
# Batch 1: (user_u, history_H, [A])            C=1  →  seq_len = 1+32+1  = 34
# Batch 2: (user_u, history_H, [A, B,…,H])     C=8  →  seq_len = 1+32+8  = 41
#
# This is the strongest test: adding candidates changes the total sequence
# length. If any sequence-level operation (e.g. layer norm over the sequence
# axis) leaked cross-candidate information, this experiment would catch it.
# Note: RMSNorm is applied axis=-1 (feature dim), not across tokens — no leak.
# ===========================================================================

print("\n─── Experiment 2: Different Sequence Length (C=1 vs C=8) ─────────────")

co_7 = random_candidates(7, rng_seed=300)

la_2 = logits_A(CAND_A)                               # C=1
lb_2 = logits_A(cat_candidates(CAND_A, co_7))         # C=8
diff_2, pass_2 = report("Exp2", la_2, lb_2)


# ===========================================================================
# Experiment 3 — Extreme co-candidate values (stress test)
#
# Batch 1: (user_u, history_H, [A, candidate_zeros])
#           → co-candidate has hash=0 everywhere (padding/invalid) + zero embeddings
# Batch 2: (user_u, history_H, [A, candidate_maxfloat])
#           → co-candidate has large non-zero hashes + embedding vectors = +1e4
#
# The most extreme possible embedding table lookups. If there is any floating-
# point contamination path, this would surface it.
#
# Note: hash=0 marks a candidate as padding (candidate_padding_mask = False),
# so the zeros co-candidate is already masked out by the padding mask alone.
# The maxfloat co-candidate is a valid token with extreme activations — it
# tests whether the attention mask physically blocks contamination of A.
# ===========================================================================

print("\n─── Experiment 3: Extreme Co-Candidate Values (Stress Test) ──────────")

# Zeros co-candidate: hash=0 (invalid/padding), embeddings=0
cand_zeros = (
    np.zeros((1, 1, NUM_ITEM_HASHES), dtype=np.int32),             # post hashes = 0 (padding)
    np.zeros((1, 1, NUM_ITEM_HASHES, EMB_SIZE), dtype=np.float32), # post embeddings = 0
    np.zeros((1, 1, NUM_AUTHOR_HASHES), dtype=np.int32),           # author hashes = 0 (padding)
    np.zeros((1, 1, NUM_AUTHOR_HASHES, EMB_SIZE), dtype=np.float32),
    np.zeros((1, 1), dtype=np.int32),                              # product surface = 0
)

# Maxfloat co-candidate: non-zero hashes (valid), embeddings = +1e4 everywhere
_MAX_HASH = np.iinfo(np.int32).max
cand_max = (
    np.full((1, 1, NUM_ITEM_HASHES), _MAX_HASH, dtype=np.int32),
    np.full((1, 1, NUM_ITEM_HASHES, EMB_SIZE), 1e4, dtype=np.float32),
    np.full((1, 1, NUM_AUTHOR_HASHES), _MAX_HASH, dtype=np.int32),
    np.full((1, 1, NUM_AUTHOR_HASHES, EMB_SIZE), 1e4, dtype=np.float32),
    np.full((1, 1), PRODUCT_SURFACE_VOCAB_SIZE - 1, dtype=np.int32),  # max valid surface
)

la_3 = logits_A(cat_candidates(CAND_A, cand_zeros))
lb_3 = logits_A(cat_candidates(CAND_A, cand_max))
diff_3, pass_3 = report("Exp3", la_3, lb_3)


# ===========================================================================
# Summary
# ===========================================================================

def status(p: bool) -> str:
    return "PASS ✓" if p else "FAIL ✗"


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Experiment 1 (different co-candidates):   max_diff = {diff_1:.4f},  {status(pass_1)}")
print(f"Experiment 2 (different batch size):      max_diff = {diff_2:.4f},  {status(pass_2)}")
print(f"Experiment 3 (extreme co-candidate vals): max_diff = {diff_3:.4f},  {status(pass_3)}")
print("=" * 70)

overall = pass_1 and pass_2 and pass_3
print(f"\nOverall: {'ALL EXPERIMENTS PASSED ✓' if overall else 'ONE OR MORE EXPERIMENTS FAILED ✗'}")
print()
print("What this proves:")
print("  The attention mask (grok.py:353, jnp.where(mask, attn_logits, -1e30))")
print("  is a structural guarantee. No learned weight can override the -1e30")
print("  clamp before softmax. candidate_A's logits depend solely on")
print("  (user, history, candidate_A) regardless of co-candidate identity,")
print("  count, or embedding magnitude.")
print()
print("What this does NOT prove:")
print("  - That trained weights won't route shared context through the user")
print("    token (all candidates CAN attend to user+history — that's by design).")
print("  - Anything about the retrieval model (separate architecture).")
print("=" * 70)
