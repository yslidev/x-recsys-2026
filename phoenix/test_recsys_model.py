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

import jax.numpy as jnp
import numpy as np
import pytest

from grok import make_recsys_attn_mask, right_anchored_rope_positions
from recsys_model import compute_post_age_bucket, normalize_continuous_value, NormConfig


class TestMakeRecsysAttnMask:
    """Tests for the make_recsys_attn_mask function."""

    def test_output_shape(self):
        """Test that the output has the correct shape [1, 1, seq_len, seq_len]."""
        seq_len = 10
        candidate_start_offset = 5

        mask = make_recsys_attn_mask(seq_len, candidate_start_offset)

        assert mask.shape == (1, 1, seq_len, seq_len)

    def test_user_history_has_causal_attention(self):
        """Test that user+history positions (before candidate_start_offset) have causal attention."""
        seq_len = 8
        candidate_start_offset = 5

        mask = make_recsys_attn_mask(seq_len, candidate_start_offset)
        mask_2d = mask[0, 0]

        for i in range(candidate_start_offset):
            for j in range(candidate_start_offset):
                if j <= i:
                    assert mask_2d[i, j] == 1, f"Position {i} should attend to position {j}"
                else:
                    assert (
                        mask_2d[i, j] == 0
                    ), f"Position {i} should NOT attend to future position {j}"

    def test_candidates_attend_to_user_history(self):
        """Test that candidates can attend to all user+history positions."""
        seq_len = 8
        candidate_start_offset = 5

        mask = make_recsys_attn_mask(seq_len, candidate_start_offset)
        mask_2d = mask[0, 0]

        for candidate_pos in range(candidate_start_offset, seq_len):
            for history_pos in range(candidate_start_offset):
                assert (
                    mask_2d[candidate_pos, history_pos] == 1
                ), f"Candidate at {candidate_pos} should attend to user+history at {history_pos}"

    def test_candidates_attend_to_themselves(self):
        """Test that candidates can attend to themselves (self-attention)."""
        seq_len = 8
        candidate_start_offset = 5

        mask = make_recsys_attn_mask(seq_len, candidate_start_offset)
        mask_2d = mask[0, 0]

        for candidate_pos in range(candidate_start_offset, seq_len):
            assert (
                mask_2d[candidate_pos, candidate_pos] == 1
            ), f"Candidate at {candidate_pos} should attend to itself"

    def test_candidates_do_not_attend_to_other_candidates(self):
        """Test that candidates cannot attend to other candidates."""
        seq_len = 8
        candidate_start_offset = 5

        mask = make_recsys_attn_mask(seq_len, candidate_start_offset)
        mask_2d = mask[0, 0]

        for query_pos in range(candidate_start_offset, seq_len):
            for key_pos in range(candidate_start_offset, seq_len):
                if query_pos != key_pos:
                    assert (
                        mask_2d[query_pos, key_pos] == 0
                    ), f"Candidate at {query_pos} should NOT attend to candidate at {key_pos}"

    def test_full_mask_structure(self):
        """Test the complete mask structure with a small example."""
        # Sequence: [user, h1, h2, c1, c2, c3]
        # Positions:  0     1   2   3   4   5

        seq_len = 6
        candidate_start_offset = 3

        mask = make_recsys_attn_mask(seq_len, candidate_start_offset)
        mask_2d = mask[0, 0]

        expected = np.array(
            [
                [1, 0, 0, 0, 0, 0],  # user
                [1, 1, 0, 0, 0, 0],  # h1
                [1, 1, 1, 0, 0, 0],  # h2
                [1, 1, 1, 1, 0, 0],  # c1: user+history + self
                [1, 1, 1, 0, 1, 0],  # c2: user+history + self
                [1, 1, 1, 0, 0, 1],  # c3: user+history + self
            ],
            dtype=np.float32,
        )

        np.testing.assert_array_equal(
            np.array(mask_2d),
            expected,
            err_msg="Full mask structure does not match expected pattern",
        )

    def test_dtype_preserved(self):
        """Test that the specified dtype is used."""
        seq_len = 5
        candidate_start_offset = 3

        mask_f32 = make_recsys_attn_mask(seq_len, candidate_start_offset, dtype=jnp.float32)
        mask_f16 = make_recsys_attn_mask(seq_len, candidate_start_offset, dtype=jnp.float16)

        assert mask_f32.dtype == jnp.float32
        assert mask_f16.dtype == jnp.float16

    def test_single_candidate(self):
        """Test edge case with a single candidate."""
        seq_len = 4
        candidate_start_offset = 3

        mask = make_recsys_attn_mask(seq_len, candidate_start_offset)
        mask_2d = mask[0, 0]

        expected = np.array(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
            ],
            dtype=np.float32,
        )

        np.testing.assert_array_equal(np.array(mask_2d), expected)

    def test_all_candidates(self):
        """Test edge case where all positions except first are candidates."""
        seq_len = 4
        candidate_start_offset = 1

        mask = make_recsys_attn_mask(seq_len, candidate_start_offset)
        mask_2d = mask[0, 0]

        expected = np.array(
            [
                [1, 0, 0, 0],  # user
                [1, 1, 0, 0],  # c1: user + self
                [1, 0, 1, 0],  # c2: user + self
                [1, 0, 0, 1],  # c3: user + self
            ],
            dtype=np.float32,
        )

        np.testing.assert_array_equal(np.array(mask_2d), expected)


class TestRightAnchoredRopePositions:
    """Tests for the right_anchored_rope_positions function."""

    def test_output_shape(self):
        """Test that the output has the correct shape [B, T]."""
        B, T = 2, 10
        padding_mask = jnp.ones((B, T), dtype=jnp.bool_)
        positions = right_anchored_rope_positions(
            padding_mask, history_seq_len=6, num_user_prefix_tokens=1
        )
        assert positions.shape == (B, T)

    def test_prefix_positions_preserved(self):
        """Test that prefix token positions are 0..num_prefix-1."""
        B, T = 1, 10
        padding_mask = jnp.ones((B, T), dtype=jnp.bool_)
        positions = right_anchored_rope_positions(
            padding_mask, history_seq_len=6, num_user_prefix_tokens=2
        )
        assert float(positions[0, 0]) == 0.0
        assert float(positions[0, 1]) == 1.0

    def test_candidates_share_position(self):
        """Test that all candidate positions are the same (history_end)."""
        B = 1
        num_prefix = 1
        history_len = 4
        num_candidates = 3
        T = num_prefix + history_len + num_candidates

        padding_mask = jnp.ones((B, T), dtype=jnp.bool_)
        positions = right_anchored_rope_positions(
            padding_mask, history_seq_len=history_len, num_user_prefix_tokens=num_prefix
        )

        history_end = num_prefix + history_len
        for c in range(num_candidates):
            assert float(positions[0, history_end + c]) == float(history_end)

    def test_padding_gets_zero(self):
        """Test that padded positions get position 0."""
        B, T = 1, 8
        padding_mask = jnp.array([[True, True, True, True, False, False, False, False]])
        positions = right_anchored_rope_positions(
            padding_mask, history_seq_len=4, num_user_prefix_tokens=1
        )
        for i in range(4, 8):
            assert float(positions[0, i]) == 0.0


class TestComputePostAgeBucket:
    """Tests for the compute_post_age_bucket function."""

    def test_basic_bucketing(self):
        """Test basic bucketing with 60-minute granularity."""
        # Post that is 30 minutes old -> bucket 1 (0-59 minutes)
        impr_ts = jnp.array([[1000000]])
        post_ts = jnp.array([[1000000 - 30 * 60]])
        bucket = compute_post_age_bucket(impr_ts, post_ts, granularity_mins=60)
        assert int(bucket[0, 0]) == 1

    def test_two_hour_post(self):
        """Test a 2-hour-old post -> bucket 3 (120-179 minutes)."""
        impr_ts = jnp.array([[1000000]])
        post_ts = jnp.array([[1000000 - 120 * 60]])
        bucket = compute_post_age_bucket(impr_ts, post_ts, granularity_mins=60)
        assert int(bucket[0, 0]) == 3

    def test_missing_timestamp_zero(self):
        """Test that missing timestamps (0) map to bucket 0."""
        impr_ts = jnp.array([[0]])
        post_ts = jnp.array([[1000000]])
        bucket = compute_post_age_bucket(impr_ts, post_ts, granularity_mins=60)
        assert int(bucket[0, 0]) == 0

        impr_ts = jnp.array([[1000000]])
        post_ts = jnp.array([[0]])
        bucket = compute_post_age_bucket(impr_ts, post_ts, granularity_mins=60)
        assert int(bucket[0, 0]) == 0

    def test_negative_age_maps_to_zero(self):
        """Test that negative age (clock skew) maps to bucket 0."""
        impr_ts = jnp.array([[1000000]])
        post_ts = jnp.array([[1000000 + 60 * 60]])  # post created AFTER impression
        bucket = compute_post_age_bucket(impr_ts, post_ts, granularity_mins=60)
        assert int(bucket[0, 0]) == 0

    def test_overflow_bucket(self):
        """Test that very old posts go to the overflow bucket."""
        # POST_AGE_MAX_MINUTES = 4800 (80 hours)
        impr_ts = jnp.array([[1000000]])
        post_ts = jnp.array([[1000000 - 5000 * 60]])  # 5000 minutes old
        bucket = compute_post_age_bucket(impr_ts, post_ts, granularity_mins=60)
        # overflow bucket = 4800 // 60 + 1 = 81
        assert int(bucket[0, 0]) == 81

    def test_batch_processing(self):
        """Test that bucketing works on batches."""
        impr_ts = jnp.array([[1000000, 1000000, 1000000]])
        post_ts = jnp.array([[1000000 - 30 * 60, 1000000 - 120 * 60, 0]])
        buckets = compute_post_age_bucket(impr_ts, post_ts, granularity_mins=60)
        assert buckets.shape == (1, 3)
        assert int(buckets[0, 0]) == 1  # 30 min -> bucket 1
        assert int(buckets[0, 1]) == 3  # 120 min -> bucket 3
        assert int(buckets[0, 2]) == 0  # missing -> bucket 0


class TestNormalizeContinuousValue:
    """Tests for continuous value normalization."""

    def test_linear_normalization(self):
        """Test linear normalization: x / norm_scale."""
        config = NormConfig(norm_scale=30.0, use_log=False)
        values = jnp.array([0.0, 15.0, 30.0, 60.0])
        result = normalize_continuous_value(values, config)
        np.testing.assert_allclose(np.array(result), [0.0, 0.5, 1.0, 1.0], atol=1e-6)

    def test_log_normalization(self):
        """Test log normalization: log1p(x) / log1p(norm_scale)."""
        config = NormConfig(norm_scale=30.0, use_log=True)
        values = jnp.array([0.0, 30.0])
        result = normalize_continuous_value(values, config)
        assert float(result[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(result[1]) == pytest.approx(1.0, abs=1e-6)

    def test_clamping(self):
        """Test that values are clamped to [0, norm_scale]."""
        config = NormConfig(norm_scale=10.0, use_log=False)
        values = jnp.array([-5.0, 0.0, 5.0, 15.0])
        result = normalize_continuous_value(values, config)
        np.testing.assert_allclose(np.array(result), [0.0, 0.0, 0.5, 1.0], atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
