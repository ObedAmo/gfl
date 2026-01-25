"""Test for fusion index validation and canonicalization"""
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from gfl.utils.validation import check_fusion_pairs

# =============================================================================
# Basic functionality tests
# =============================================================================

def test_basic_canonicalization():
    """Test basic case: remove self-loops, order pairs, remove duplicates."""
    pairs = np.array([[1, 0], [0, 1], [1, 1], [2, 0]])
    weights = np.array([1.0, 2.0, 0.5, 1.5])
    
    pairs_norm, weights_norm, n_params = check_fusion_pairs(pairs, weights)
    
    # Expected: (0,1) with weight 1.0 (first), (0,2) with weight 1.5
    assert_array_equal(pairs_norm, [[0, 1], [0, 2]])
    assert_array_equal(weights_norm, [1.0, 1.5])
    assert n_params == 3


def test_already_canonical():
    """Test pairs that are already in canonical form."""
    pairs = np.array([[0, 1], [1, 2], [2, 3]])
    weights = np.array([1.0, 2.0, 3.0])
    
    pairs_norm, weights_norm, n_params = check_fusion_pairs(pairs, weights)
    
    assert_array_equal(pairs_norm, pairs)
    assert_array_equal(weights_norm, weights)
    assert n_params == 4


def test_transpose_input():
    """Test (2, n_pairs) input format."""
    pairs = np.array([[0, 1, 2], [1, 2, 3]])  # Shape (2, 3)
    weights = np.array([1.0, 2.0, 3.0])
    
    pairs_norm, weights_norm, n_params = check_fusion_pairs(pairs, weights)
    
    assert_array_equal(pairs_norm, [[0, 1], [1, 2], [2, 3]])
    assert_array_equal(weights_norm, weights)
    assert n_params == 4


def test_empty_after_self_loop_removal():
    """Test when all pairs are self-loops."""
    pairs = np.array([[0, 0], [1, 1], [2, 2]])
    weights = np.array([1.0, 2.0, 3.0])
    
    pairs_norm, weights_norm, n_params = check_fusion_pairs(pairs, weights)
    
    assert pairs_norm.shape == (0, 2)
    assert weights_norm.shape == (0,)
    assert n_params == 0


def test_explicit_n_params():
    """Test that explicit n_params is respected."""
    pairs = np.array([[0, 1], [1, 2]])
    weights = np.array([1.0, 2.0])
    
    pairs_norm, weights_norm, n_params = check_fusion_pairs(
        pairs, weights, n_params=10
    )
    
    assert n_params == 10  # User-provided value


# =============================================================================
# Duplicate aggregation strategies
# =============================================================================

def test_duplicate_strategy_first():
    """Test 'first' strategy keeps first occurrence."""
    pairs = np.array([[0, 1], [1, 0], [0, 1]])
    weights = np.array([1.0, 2.0, 3.0])
    
    _, weights_norm, _ = check_fusion_pairs(
        pairs, weights, duplicate_strategy='first'
    )
    
    assert_array_equal(weights_norm, [1.0])


def test_duplicate_strategy_sum():
    """Test 'sum' strategy sums duplicate weights."""
    pairs = np.array([[0, 1], [1, 0], [0, 1]])
    weights = np.array([1.0, 2.0, 3.0])
    
    _, weights_norm, _ = check_fusion_pairs(
        pairs, weights, duplicate_strategy='sum'
    )
    
    assert_array_equal(weights_norm, [6.0])


def test_duplicate_strategy_mean():
    """Test 'mean' strategy averages duplicate weights."""
    pairs = np.array([[0, 1], [1, 0], [0, 1]])
    weights = np.array([1.0, 2.0, 3.0])
    
    _, weights_norm, _ = check_fusion_pairs(
        pairs, weights, duplicate_strategy='mean'
    )
    
    assert_array_equal(weights_norm, [2.0])


def test_duplicate_strategy_max():
    """Test 'max' strategy takes maximum weight."""
    pairs = np.array([[0, 1], [1, 0], [0, 1]])
    weights = np.array([1.0, 2.0, 3.0])
    
    _, weights_norm, _ = check_fusion_pairs(
        pairs, weights, duplicate_strategy='max'
    )
    
    assert_array_equal(weights_norm, [3.0])


def test_duplicate_strategy_min():
    """Test 'min' strategy takes minimum weight."""
    pairs = np.array([[0, 1], [1, 0], [0, 1]])
    weights = np.array([1.0, 2.0, 3.0])
    
    _, weights_norm, _ = check_fusion_pairs(
        pairs, weights, duplicate_strategy='min'
    )
    
    assert_array_equal(weights_norm, [1.0])


def test_multiple_duplicate_groups():
    """Test handling multiple groups of duplicates."""
    pairs = np.array([[0, 1], [1, 0], [1, 2], [2, 1], [0, 2]])
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    pairs_norm, weights_norm, _ = check_fusion_pairs(
        pairs, weights, duplicate_strategy='sum'
    )
    
    assert_array_equal(pairs_norm, [[0, 1], [0, 2], [1, 2]])
    assert_array_equal(weights_norm, [3.0, 5.0, 7.0])


# =============================================================================
# Edge cases
# =============================================================================

def test_single_pair():
    """Test single fusion pair."""
    pairs = np.array([[0, 1]])
    weights = np.array([1.0])
    
    pairs_norm, weights_norm, n_params = check_fusion_pairs(pairs, weights)
    
    assert_array_equal(pairs_norm, [[0, 1]])
    assert_array_equal(weights_norm, [1.0])
    assert n_params == 2


def test_no_duplicates():
    """Test that no-duplicate fast path works correctly."""
    pairs = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    
    pairs_norm, weights_norm, _ = check_fusion_pairs(pairs, weights)
    
    assert_array_equal(pairs_norm, pairs)
    assert_array_equal(weights_norm, weights)


def test_reverse_order_pairs():
    """Test that pairs are properly ordered i < j."""
    pairs = np.array([[3, 0], [2, 1], [5, 4]])
    weights = np.array([1.0, 2.0, 3.0])
    
    pairs_norm, weights_norm, _ = check_fusion_pairs(pairs, weights)
    
    assert_array_equal(pairs_norm, [[0, 3], [1, 2], [4, 5]])
    assert_array_equal(weights_norm, [1.0, 2.0, 3.0])


def test_large_index_values():
    """Test with large index values."""
    pairs = np.array([[0, 1000], [999, 1000]])
    weights = np.array([1.0, 2.0])
    
    pairs_norm, _, n_params = check_fusion_pairs(pairs, weights)
    
    assert_array_equal(pairs_norm, [[0, 1000], [999, 1000]])
    assert n_params == 1001


def test_large_index_values():
    """Test with large index values."""
    pairs = np.array([[0, 1000], [999, 1000]])
    weights = np.array([1.0, 2.0])
    
    pairs_norm, weights_norm, n_params = check_fusion_pairs(pairs, weights)
    
    assert_array_equal(pairs_norm, [[0, 1000], [999, 1000]])
    assert n_params == 1001


# =============================================================================
# Input validation tests
# =============================================================================

def test_invalid_fusion_pairs_ndim():
    """Test error on wrong number of dimensions."""
    pairs = np.array([0, 1, 2])  # 1D array
    weights = np.array([1.0])
    
    with pytest.raises(ValueError, match="fusion_pairs must be 2D, got 1D array"):
        check_fusion_pairs(pairs, weights)
        

def test_invalid_fusion_pairs_shape():
    """Test error when pairs shape doesn't allow valid pairing with weights."""
    pairs = np.array([[0, 1, 2], [3, 4, 5]])
    weights = np.array([1.0])  
    
    with pytest.raises(ValueError, match="weights has length 1, expected 3"):
        check_fusion_pairs(pairs, weights)


def test_invalid_weights_shape():
    """Test error on mismatched weights shape."""
    pairs = np.array([[0, 1], [1, 2]])
    weights = np.array([1.0])  # Wrong length
    
    with pytest.raises(ValueError, match="weights has length 1, expected 2"):
        check_fusion_pairs(pairs, weights)


def test_negative_weights():
    """Test error on negative weights."""
    pairs = np.array([[0, 1]])
    weights = np.array([-1.0])
    
    with pytest.raises(ValueError, match="weights must contain only positive values"):
        check_fusion_pairs(pairs, weights)


def test_zero_weights():
    """Test error on zero weights."""
    pairs = np.array([[0, 1]])
    weights = np.array([0.0])
    
    with pytest.raises(ValueError, match="weights must contain only positive values"):
        check_fusion_pairs(pairs, weights)


def test_negative_indices():
    """Test error on negative indices."""
    pairs = np.array([[-1, 1]])
    weights = np.array([1.0])
    
    with pytest.raises(ValueError, match="fusion_pairs must contain only non-negative values"):
        check_fusion_pairs(pairs, weights)


def test_indices_exceed_n_params():
    """Test error when indices exceed n_params."""
    pairs = np.array([[0, 5]])
    weights = np.array([1.0])
    
    with pytest.raises(ValueError, match="must be <= 2"):
        check_fusion_pairs(pairs, weights, n_params=3)


def test_sorted_output():
    """Test that output is lexicographically sorted."""
    pairs = np.array([[2, 3], [0, 1], [1, 2]])
    weights = np.array([3.0, 1.0, 2.0])
    
    pairs_norm, weights_norm, _ = check_fusion_pairs(pairs, weights)
    
    # Should be sorted by (i, j)
    assert_array_equal(pairs_norm, [[0, 1], [1, 2], [2, 3]])
    assert_array_equal(weights_norm, [1.0, 2.0, 3.0])


def test_preserves_dtype():
    """Test that output dtypes are correct."""
    pairs = np.array([[0, 1]], dtype=np.int32)
    weights = np.array([1.0], dtype=np.float32)
    
    pairs_norm, weights_norm, _ = check_fusion_pairs(pairs, weights)
    
    assert pairs_norm.dtype == np.int64
    assert weights_norm.dtype == np.float64


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
