"""Tests for combined groups and fusion pairs validation."""
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from gfl.utils.validation import check_groups_and_pairs


# =============================================================================
# Valid inputs - consistency checks
# =============================================================================

def test_valid_groups_and_pairs():
    """Test valid groups and pairs that reference same parameter space."""
    groups = np.array([0, 1, 2, 0, 1])
    pairs = np.array([[0, 1], [1, 2]])
    weights = np.array([1.0, 2.0])
    
    g, n, p, w = check_groups_and_pairs(groups, 5, pairs, weights)
    
    assert_array_equal(g, groups)
    assert n == 3
    assert_array_equal(p, pairs)
    assert_array_equal(w, weights)


def test_pairs_reference_empty_groups():
    """Test pairs can reference groups with no observations."""
    groups = np.array([0, 2, 4])  # Groups 1 and 3 are empty
    pairs = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    
    g, n, p, w = check_groups_and_pairs(groups, 3, pairs, weights, n_groups=5)
    
    assert n == 5
    assert_array_equal(g, groups)
    # Pairs should be canonicalized
    assert p.shape[0] == 4


def test_single_group_no_pairs():
    """Test single group with no fusion pairs."""
    groups = np.array([0, 0, 0])
    pairs = np.array([]).reshape(0, 2)
    weights = np.array([])
    
    g, n, p, w = check_groups_and_pairs(groups, 3, pairs, weights)
    
    assert n == 1
    assert p.shape == (0, 2)


def test_all_groups_connected():
    """Test fully connected groups."""
    groups = np.array([0, 1, 2])
    pairs = np.array([[0, 1], [0, 2], [1, 2]])
    weights = np.array([1.0, 1.0, 1.0])
    
    g, n, p, w = check_groups_and_pairs(groups, 3, pairs, weights)
    
    assert n == 3
    assert p.shape[0] == 3


def test_chain_structure():
    """Test chain/path graph structure."""
    groups = np.array([0, 0, 1, 1, 2, 2])
    pairs = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    weights = np.ones(4)
    
    g, n, p, w = check_groups_and_pairs(groups, 6, pairs, weights, n_groups=5)
    
    assert n == 5


def test_explicit_n_groups_larger():
    """Test with explicit n_groups larger than observed."""
    groups = np.array([0, 1])
    pairs = np.array([[0, 1], [1, 2], [2, 3]])  # References groups 0-3
    weights = np.ones(3)
    
    g, n, p, w = check_groups_and_pairs(groups, 2, pairs, weights, n_groups=4)
    
    assert n == 4


# =============================================================================
# Pairs validation still works (canonicalization, deduplication)
# =============================================================================

def test_pairs_are_canonicalized():
    """Test that pairs are still canonicalized."""
    groups = np.array([0, 1, 2])
    pairs = np.array([[1, 0], [2, 1], [0, 0]])  # Reversed, self-loop
    weights = np.array([1.0, 2.0, 3.0])
    
    g, n, p, w = check_groups_and_pairs(groups, 3, pairs, weights)
    
    # Self-loop removed, pairs ordered i < j
    assert_array_equal(p, [[0, 1], [1, 2]])
    assert len(w) == 2


def test_duplicate_pairs_aggregated():
    """Test duplicate pairs are aggregated."""
    groups = np.array([0, 1, 2])
    pairs = np.array([[0, 1], [1, 0], [0, 1]])  # Duplicates
    weights = np.array([1.0, 2.0, 3.0])
    
    g, n, p, w = check_groups_and_pairs(
        groups, 3, pairs, weights, duplicate_strategy='sum'
    )
    
    assert_array_equal(p, [[0, 1]])
    assert_array_equal(w, [6.0])


# =============================================================================
# Error cases - inconsistency between groups and pairs
# =============================================================================

def test_pairs_exceed_n_groups():
    """Test error when pairs reference indices >= n_groups."""
    groups = np.array([0, 1, 2])  # n_groups = 3
    pairs = np.array([[0, 1], [1, 5]])  # Index 5 exceeds n_groups
    weights = np.array([1.0, 2.0])
    
    with pytest.raises(ValueError, match="fusion_pairs must be <= 2"):
        check_groups_and_pairs(groups, 3, pairs, weights)


def test_pairs_exceed_explicit_n_groups():
    """Test error when pairs exceed explicit n_groups."""
    groups = np.array([0, 1])
    pairs = np.array([[0, 1], [1, 10]])  # Index 10 exceeds n_groups=5
    weights = np.array([1.0, 2.0])
    
    with pytest.raises(ValueError, match="fusion_pairs must be <= 4"):
        check_groups_and_pairs(groups, 2, pairs, weights, n_groups=5)


def test_groups_not_contiguous():
    """Test error when groups are not contiguous."""
    groups = np.array([0, 2, 5])  # Not contiguous
    pairs = np.array([[0, 2]])
    weights = np.array([1.0])
    
    with pytest.raises(ValueError, match="groups must be in contiguous format"):
        check_groups_and_pairs(groups, 3, pairs, weights)


def test_negative_groups():
    """Test error on negative group IDs."""
    groups = np.array([0, -1, 2])
    pairs = np.array([[0, 1]])
    weights = np.array([1.0])
    
    with pytest.raises(ValueError, match="groups must contain only non-negative"):
        check_groups_and_pairs(groups, 3, pairs, weights)


def test_negative_pair_indices():
    """Test error on negative pair indices."""
    groups = np.array([0, 1, 2])
    pairs = np.array([[0, -1]])
    weights = np.array([1.0])
    
    with pytest.raises(ValueError, match="fusion_pairs must contain only non-negative"):
        check_groups_and_pairs(groups, 3, pairs, weights)


def test_groups_length_mismatch():
    """Test error when groups length doesn't match n_samples."""
    groups = np.array([0, 1])  # Only 2 elements
    pairs = np.array([[0, 1]])
    weights = np.array([1.0])
    
    with pytest.raises(ValueError, match="groups has length 2, expected 5"):
        check_groups_and_pairs(groups, 5, pairs, weights)


def test_weights_length_mismatch():
    """Test error when weights length doesn't match pairs."""
    groups = np.array([0, 1, 2])
    pairs = np.array([[0, 1], [1, 2]])
    weights = np.array([1.0])  # Only 1 weight
    
    with pytest.raises(ValueError, match="weights has length 1, expected 2"):
        check_groups_and_pairs(groups, 3, pairs, weights)


def test_invalid_n_groups():
    """Test error on invalid n_groups value."""
    groups = np.array([0, 1])
    pairs = np.array([[0, 1]])
    weights = np.array([1.0])
    
    with pytest.raises(ValueError, match="n_groups must be a positive integer"):
        check_groups_and_pairs(groups, 2, pairs, weights, n_groups=-1)


# =============================================================================
# Edge cases
# =============================================================================

def test_empty_groups_with_pairs():
    """Test empty groups but pairs exist (should work with explicit n_groups)."""
    groups = np.array([], dtype=np.int64)
    pairs = np.array([[0, 1], [1, 2]])
    weights = np.array([1.0, 2.0])
    
    g, n, p, w = check_groups_and_pairs(groups, 0, pairs, weights, n_groups=3)
    
    assert len(g) == 0
    assert n == 3
    assert len(p) == 2


def test_groups_exist_no_pairs():
    """Test groups without any fusion pairs."""
    groups = np.array([0, 1, 2, 0, 1])
    pairs = np.array([]).reshape(0, 2)
    weights = np.array([])
    
    g, n, p, w = check_groups_and_pairs(groups, 5, pairs, weights)
    
    assert n == 3
    assert p.shape == (0, 2)


def test_single_observation_single_pair():
    """Test minimal case."""
    groups = np.array([0])
    pairs = np.array([[0, 1]])
    weights = np.array([1.0])
    
    g, n, p, w = check_groups_and_pairs(groups, 1, pairs, weights, n_groups=2)
    
    assert n == 2


# =============================================================================
# Realistic scenarios
# =============================================================================

def test_spatial_grid_scenario():
    """Test realistic spatial grid with neighbors."""
    # 3x3 grid: groups 0-8, observations only in corners and center
    groups = np.array([0, 0, 2, 2, 4, 4, 6, 6, 8, 8])
    # Grid edges
    pairs = np.array([
        [0, 1], [1, 2],
        [3, 4], [4, 5],
        [6, 7], [7, 8],
        [0, 3], [1, 4], [2, 5],
        [3, 6], [4, 7], [5, 8]
    ])
    weights = np.ones(12)
    
    g, n, p, w = check_groups_and_pairs(groups, 10, pairs, weights, n_groups=9)
    
    assert n == 9
    assert len(p) == 12


def test_time_series_scenario():
    """Test time series with temporal connections."""
    # 5 time points, multiple observations per time
    groups = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 4, 4])
    # Sequential connections
    pairs = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    weights = np.ones(4)
    
    g, n, p, w = check_groups_and_pairs(groups, 11, pairs, weights)
    
    assert n == 5
    assert_array_equal(p, pairs)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
