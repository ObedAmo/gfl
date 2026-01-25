"""Tests for group operations utilities."""
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from gfl.core._group_stats import compute_group_data


# =============================================================================
# Basic Functionality Tests
# =============================================================================

def test_basic_grouping():
    """Test basic group statistics computation."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    groups = np.array([0, 0, 1, 1])
    
    gd = compute_group_data(data, groups)
    
    assert_array_equal(gd.sizes, [2, 2])
    assert_array_almost_equal(gd.sums, [3.0, 7.0])
    assert_array_almost_equal(gd.means, [1.5, 3.5])
    assert gd.is_grouped_data


def test_single_group():
    """Test when all data belongs to one group."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    groups = np.array([0, 0, 0, 0])
    
    gd = compute_group_data(data, groups)
    
    assert_array_equal(gd.sizes, [4])
    assert_array_almost_equal(gd.sums, [10.0])
    assert_array_almost_equal(gd.means, [2.5])
    assert gd.is_grouped_data


def test_many_groups():
    """Test with many groups."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    groups = np.array([0, 1, 0, 1, 2, 2])
    
    gd = compute_group_data(data, groups)
    
    assert_array_equal(gd.sizes, [2, 2, 2])
    assert_array_almost_equal(gd.sums, [4.0, 6.0, 11.0])
    assert_array_almost_equal(gd.means, [2.0, 3.0, 5.5])
    assert gd.is_grouped_data


# =============================================================================
# Empty and Missing Groups Tests
# =============================================================================

def test_empty_input():
    """Test with empty arrays."""
    gd = compute_group_data(
        np.array([], dtype=np.float64),
        np.array([], dtype=np.int64)
    )
    
    assert gd.sizes.size == 0
    assert gd.sums.size == 0
    assert gd.means.size == 0
    assert not gd.is_grouped_data


def test_missing_group_zero_padded():
    """Test that missing groups are zero-padded."""
    data = np.array([10.0, 20.0])
    groups = np.array([0, 2])
    
    gd = compute_group_data(data, groups)
    
    assert_array_equal(gd.sizes, [1, 0, 1])
    assert_array_almost_equal(gd.sums, [10.0, 0.0, 20.0])
    assert_array_almost_equal(gd.means, [10.0, 0.0, 20.0])
    assert gd.is_grouped_data


def test_multiple_missing_groups():
    """Test with multiple missing groups."""
    data = np.array([5.0, 15.0])
    groups = np.array([1, 4])
    
    gd = compute_group_data(data, groups)
    
    assert_array_equal(gd.sizes, [0, 1, 0, 0, 1])
    assert_array_almost_equal(gd.sums, [0.0, 5.0, 0.0, 0.0, 15.0])
    assert_array_almost_equal(gd.means, [0.0, 5.0, 0.0, 0.0, 15.0])


def test_explicit_n_groups_creates_padding():
    """Test explicit n_groups parameter creates zero-padded groups."""
    data = np.array([1.0, 2.0])
    groups = np.array([0, 1])
    
    gd = compute_group_data(data, groups, n_groups=5)
    
    assert_array_equal(gd.sizes, [1, 1, 0, 0, 0])
    assert_array_almost_equal(gd.sums, [1.0, 2.0, 0.0, 0.0, 0.0])
    assert_array_almost_equal(gd.means, [1.0, 2.0, 0.0, 0.0, 0.0])


# =============================================================================
# Fast Path Tests (Identity Groups)
# =============================================================================

def test_fast_path_identity_groups():
    """Test fast path when each observation is its own group."""
    data = np.array([5.0, 6.0, 7.0])
    groups = np.array([0, 1, 2])
    
    gd = compute_group_data(data, groups)
    
    assert_array_equal(gd.sizes, [1, 1, 1])
    assert_array_almost_equal(gd.sums, data)
    assert_array_almost_equal(gd.means, data)
    assert not gd.is_grouped_data  # Fast path flag


def test_fast_path_single_element():
    """Test fast path with single element."""
    data = np.array([42.0])
    groups = np.array([0])
    
    gd = compute_group_data(data, groups)
    
    assert_array_equal(gd.sizes, [1])
    assert_array_almost_equal(gd.sums, [42.0])
    assert_array_almost_equal(gd.means, [42.0])
    assert not gd.is_grouped_data


def test_no_fast_path_when_not_identity():
    """Test that fast path is not taken for non-identity grouping."""
    data = np.array([1.0, 2.0, 3.0])
    groups = np.array([0, 0, 2])  # Not identity
    
    gd = compute_group_data(data, groups)
    
    assert gd.is_grouped_data  # Should not use fast path


# =============================================================================
# Input Validation Tests
# =============================================================================

def test_length_mismatch_raises():
    """Test that mismatched data and groups raise ValueError."""
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_group_data(
            np.array([1.0, 2.0]),
            np.array([0])
        )


def test_negative_group_id_raises():
    """Test that negative group IDs raise ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        compute_group_data(
            np.array([1.0]),
            np.array([-1])
        )


def test_multidimensional_data_raises():
    """Test that multidimensional arrays raise ValueError."""
    with pytest.raises(ValueError, match="1D arrays"):
        compute_group_data(
            np.array([[1.0], [2.0]]),
            np.array([0, 1])
        )


def test_multidimensional_groups_raises():
    """Test that multidimensional group arrays raise ValueError."""
    with pytest.raises(ValueError, match="1D arrays"):
        compute_group_data(
            np.array([1.0, 2.0]),
            np.array([[0], [1]])
        )


def test_explicit_n_groups_too_small_raises():
    """Test that n_groups smaller than max group ID raises ValueError."""
    with pytest.raises(ValueError, match="n_groups must exceed"):
        compute_group_data(
            np.array([1.0, 2.0]),
            np.array([0, 2]),
            n_groups=2  # max(groups) = 2, needs n_groups >= 3
        )


# =============================================================================
# Edge Cases and Special Values
# =============================================================================

def test_negative_values():
    """Test with negative values in data."""
    data = np.array([-1.0, -2.0, 3.0])
    groups = np.array([0, 0, 1])
    
    gd = compute_group_data(data, groups)
    
    assert_array_almost_equal(gd.sums, [-3.0, 3.0])
    assert_array_almost_equal(gd.means, [-1.5, 3.0])


def test_unordered_groups():
    """Test that group order doesn't matter."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    groups = np.array([1, 0, 1, 0])
    
    gd = compute_group_data(data, groups)
    
    assert_array_equal(gd.sizes, [2, 2])
    assert_array_almost_equal(gd.sums, [6.0, 4.0])
    assert_array_almost_equal(gd.means, [3.0, 2.0])


# =============================================================================
# Large Scale Tests
# =============================================================================

def test_large_number_of_groups():
    """Test with large number of groups."""
    n = 10000
    data = np.random.randn(n)
    groups = np.arange(n)
    
    gd = compute_group_data(data, groups)
    
    assert gd.sizes.shape == (n,)
    assert_array_equal(gd.sizes, 1)
    assert_array_almost_equal(gd.sums, data)
    assert_array_almost_equal(gd.means, data)
    assert not gd.is_grouped_data  # Fast path


def test_many_samples_per_group():
    """Test with many samples in few groups."""
    n_samples = 10000
    n_groups = 10
    data = np.random.randn(n_samples)
    groups = np.random.randint(0, n_groups, size=n_samples)
    
    gd = compute_group_data(data, groups)
    
    assert gd.sizes.shape == (n_groups,)
    assert gd.sizes.sum() == n_samples
    assert gd.is_grouped_data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
