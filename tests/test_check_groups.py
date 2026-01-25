"""Tests for group validation."""
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from gfl.utils.validation import check_groups


# =============================================================================
# Valid inputs - should pass
# =============================================================================

def test_basic_contiguous_groups():
    """Test basic contiguous groups [0, 1, 2]."""
    groups_in = np.array([0, 1, 2, 1, 0])
    groups_out, n_groups = check_groups(groups_in, n_samples=5)
    
    assert_array_equal(groups_out, groups_in)
    assert n_groups == 3


def test_single_group():
    """Test all observations in one group."""
    groups_in = np.array([0, 0, 0, 0])
    groups_out, n_groups = check_groups(groups_in, n_samples=4)
    
    assert_array_equal(groups_out, groups_in)
    assert n_groups == 1


def test_each_observation_own_group():
    """Test each observation in its own group."""
    groups_in = np.array([0, 1, 2, 3, 4])
    groups_out, n_groups = check_groups(groups_in, n_samples=5)
    
    assert_array_equal(groups_out, groups_in)
    assert n_groups == 5


def test_with_explicit_n_groups():
    """Test with explicit n_groups (some groups empty)."""
    groups_in = np.array([0, 2, 4])
    groups_out, n_groups = check_groups(groups_in, n_samples=3, n_groups=5)
    
    assert_array_equal(groups_out, groups_in)
    assert n_groups == 5


def test_large_n_groups_sparse_observations():
    """Test many empty groups."""
    groups_in = np.array([0, 10, 20])
    groups_out, n_groups = check_groups(groups_in, n_samples=3, n_groups=21)
    
    assert_array_equal(groups_out, groups_in)
    assert n_groups == 21


def test_repeated_groups():
    """Test groups with multiple observations."""
    groups_in = np.array([0, 0, 1, 1, 2, 2, 2])
    groups_out, n_groups = check_groups(groups_in, n_samples=7)
    
    assert_array_equal(groups_out, groups_in)
    assert n_groups == 3


def test_unordered_but_valid():
    """Test groups in random order but valid range."""
    groups_in = np.array([2, 0, 1, 2, 0, 1])
    groups_out, n_groups = check_groups(groups_in, n_samples=6)
    
    assert_array_equal(groups_out, groups_in)
    assert n_groups == 3


# =============================================================================
# Edge cases
# =============================================================================

def test_empty_groups_with_n_groups():
    """Test empty groups array when n_groups is specified."""
    groups_in = np.array([], dtype=np.int64)
    groups_out, n_groups = check_groups(groups_in, n_samples=0, n_groups=5)
    
    assert len(groups_out) == 0
    assert n_groups == 5


def test_empty_groups_without_n_groups():
    """Test empty groups array when n_groups is not specified."""
    groups_in = np.array([], dtype=np.int64)
    groups_out, n_groups = check_groups(groups_in, n_samples=0)
    
    assert len(groups_out) == 0
    assert n_groups == 0


def test_single_observation():
    """Test single observation."""
    groups_in = np.array([0])
    groups_out, n_groups = check_groups(groups_in, n_samples=1)
    
    assert_array_equal(groups_out, groups_in)
    assert n_groups == 1


# =============================================================================
# Invalid inputs - should fail
# =============================================================================

def test_non_contiguous_groups_no_n_groups():
    """Test error when groups are not contiguous and n_groups not specified."""
    groups = np.array([0, 2, 5])  # Missing 1, 3, 4
    
    with pytest.raises(ValueError, match="groups must be in contiguous format"):
        check_groups(groups, n_samples=3)


def test_groups_not_starting_from_zero():
    """Test error when groups don't start from 0."""
    groups = np.array([1, 2, 3])
    
    with pytest.raises(ValueError, match="groups must be in contiguous format"):
        check_groups(groups, n_samples=3)


def test_groups_with_gaps():
    """Test error when there are gaps in group IDs."""
    groups = np.array([0, 1, 3, 4])  # Missing 2
    
    with pytest.raises(ValueError, match="groups must be in contiguous format"):
        check_groups(groups, n_samples=4)


def test_negative_groups():
    """Test error on negative group IDs."""
    groups = np.array([0, -1, 2])
    
    with pytest.raises(ValueError, match="groups must contain only non-negative values"):
        check_groups(groups, n_samples=3)


def test_length_mismatch():
    """Test error when groups length doesn't match n_samples."""
    groups = np.array([0, 1, 2])
    
    with pytest.raises(ValueError, match="groups has length 3, expected 5"):
        check_groups(groups, n_samples=5)


def test_max_group_exceeds_n_groups():
    """Test error when max group ID >= n_groups."""
    groups = np.array([0, 1, 5])
    
    with pytest.raises(ValueError, match="groups contains values >= n_groups"):
        check_groups(groups, n_samples=3, n_groups=3)


def test_more_unique_groups_than_n_groups():
    """Test error when unique groups > n_groups."""
    groups = np.array([0, 1, 2, 3])
    
    with pytest.raises(ValueError, match="Cannot have more unique groups than n_groups"):
        check_groups(groups, n_samples=4, n_groups=3)


def test_groups_not_starting_from_zero_with_n_groups():
    """Test error when groups don't start from 0 with n_groups specified."""
    groups = np.array([1, 2, 3])
    
    with pytest.raises(ValueError, match="groups must start from 0"):
        check_groups(groups, n_samples=3, n_groups=4)


def test_invalid_n_groups_negative():
    """Test error on negative n_groups."""
    groups = np.array([0, 1])
    
    with pytest.raises(ValueError, match="n_groups must be a positive integer"):
        check_groups(groups, n_samples=2, n_groups=-1)


def test_invalid_n_groups_zero():
    """Test error on zero n_groups."""
    groups = np.array([0, 1])
    
    with pytest.raises(ValueError, match="n_groups must be a positive integer"):
        check_groups(groups, n_samples=2, n_groups=0)


def test_invalid_n_groups_float():
    """Test error on float n_groups."""
    groups = np.array([0, 1])
    
    with pytest.raises(ValueError, match="n_groups must be a positive integer"):
        check_groups(groups, n_samples=2, n_groups=3.5)


def test_nan_in_groups():
    """Test error on NaN values in groups (converted to 0, causes non-contiguous)."""
    import warnings
    groups = np.array([0, np.nan, 2])
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError, match="groups must be in contiguous format"):
            check_groups(groups, n_samples=3)


def test_inf_in_groups():
    """Test error on infinite values in groups (converted to large int, causes non-contiguous)."""
    import warnings
    groups = np.array([0, np.inf, 2])
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError, match="groups must be in contiguous format"):
            check_groups(groups, n_samples=3)


# =============================================================================
# Type conversion tests
# =============================================================================

def test_converts_list_to_array():
    """Test that list input is converted to array."""
    groups_in = [0, 1, 2, 1, 0]
    groups_out, n_groups = check_groups(groups_in, n_samples=5)
    
    assert isinstance(groups_out, np.ndarray)
    assert groups_out.dtype == np.int64
    assert n_groups == 3


def test_converts_int32_to_int64():
    """Test dtype conversion to int64."""
    groups_in = np.array([0, 1, 2], dtype=np.int32)
    groups_out, _ = check_groups(groups_in, n_samples=3)
    
    assert groups_out.dtype == np.int64


def test_converts_float_to_int():
    """Test that float groups are converted to int."""
    groups_in = np.array([0.0, 1.0, 2.0])
    groups_out, _ = check_groups(groups_in, n_samples=3)
    
    assert groups_out.dtype == np.int64
    assert_array_equal(groups_out, [0, 1, 2])


# =============================================================================
# Error message clarity tests
# =============================================================================

def test_error_message_includes_range():
    """Test that error message shows actual range found."""
    groups = np.array([5, 10, 15])
    
    with pytest.raises(ValueError, match=r"Found groups in range \[5, 15\]"):
        check_groups(groups, n_samples=3)


def test_error_message_suggests_encoder():
    """Test that error message suggests GroupEncoder."""
    groups = np.array([5, 10, 15])
    
    with pytest.raises(ValueError, match="GroupEncoder"):
        check_groups(groups, n_samples=3)


def test_error_message_shows_max_when_exceeds():
    """Test error message when max group exceeds n_groups."""
    groups = np.array([0, 1, 5])
    
    with pytest.raises(ValueError, match="Max group ID is 5, but n_groups=3"):
        check_groups(groups, n_samples=3, n_groups=3)


# =============================================================================
# Multidimensional input tests
# =============================================================================

def test_2d_array_fails():
    """Test error on 2D array input."""
    groups = np.array([[0, 1], [2, 3]])
    
    with pytest.raises(ValueError, match="groups must be 1D"):
        check_groups(groups, n_samples=4)


def test_3d_array_fails():
    """Test error on 3D array input."""
    groups = np.array([[[0, 1]]])
    
    with pytest.raises(ValueError, match="groups must be 1D"):
        check_groups(groups, n_samples=2)


# =============================================================================
# Integration scenarios
# =============================================================================

def test_sparse_groups_scenario():
    """Test realistic scenario with sparse group observations."""
    # Simulating spatial data where only some regions have observations
    groups = np.array([0, 0, 5, 5, 10, 10, 15])
    groups_out, n_groups = check_groups(groups, n_samples=7, n_groups=20)
    
    assert_array_equal(groups_out, groups)
    assert n_groups == 20


def test_dense_groups_scenario():
    """Test realistic scenario with dense observations."""
    # All groups have multiple observations
    groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    groups_out, n_groups = check_groups(groups, n_samples=9)
    
    assert_array_equal(groups_out, groups)
    assert n_groups == 3


def test_mixed_group_sizes():
    """Test groups with varying numbers of observations."""
    groups = np.array([0, 1, 1, 1, 2, 2, 3])
    groups_out, n_groups = check_groups(groups, n_samples=7)
    
    assert_array_equal(groups_out, groups)
    assert n_groups == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
