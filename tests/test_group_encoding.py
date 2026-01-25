"""Tests for GroupEncoder."""
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from gfl.utils import GroupEncoder


# =============================================================================
# Basic functionality - numeric groups
# =============================================================================

def test_basic_numeric_encoding():
    """Test basic encoding with numeric groups."""
    encoder = GroupEncoder()
    groups = np.array([10, 5, 10, 15, 5])
    
    groups_enc = encoder.fit_transform(groups)
    
    assert_array_equal(groups_enc, [1, 0, 1, 2, 0])
    assert encoder.n_groups_ == 3
    assert_array_equal(encoder.classes_, [5, 10, 15])


def test_inverse_transform_numeric():
    """Test inverse transform returns original labels."""
    encoder = GroupEncoder()
    groups = np.array([10, 5, 10, 15, 5])
    
    groups_enc = encoder.fit_transform(groups)
    groups_orig = encoder.inverse_transform(groups_enc)
    
    assert_array_equal(groups_orig, groups)


def test_contiguous_groups_unchanged():
    """Test already contiguous groups [0, 1, 2] remain [0, 1, 2]."""
    encoder = GroupEncoder()
    groups = np.array([0, 1, 2, 1, 0])
    
    groups_enc = encoder.fit_transform(groups)
    
    assert_array_equal(groups_enc, groups)


def test_sort_true_vs_false():
    """Test sort parameter affects encoding order."""
    groups = np.array([15, 5, 10])
    
    # With sort=True (default)
    enc1 = GroupEncoder(sort=True)
    result1 = enc1.fit_transform(groups)
    assert_array_equal(enc1.classes_, [5, 10, 15])
    assert_array_equal(result1, [2, 0, 1])
    
    # With sort=False (first appearance order)
    enc2 = GroupEncoder(sort=False)
    result2 = enc2.fit_transform(groups)
    assert_array_equal(enc2.classes_, [15, 5, 10])
    assert_array_equal(result2, [0, 1, 2])


# =============================================================================
# String groups
# =============================================================================

def test_basic_string_encoding():
    """Test encoding with string groups."""
    encoder = GroupEncoder()
    groups = np.array(['Norfolk', 'Hampton', 'Norfolk', 'Virginia Beach'])
    
    groups_enc = encoder.fit_transform(groups)
    
    assert_array_equal(groups_enc, [1, 0, 1, 2])
    assert encoder.n_groups_ == 3
    assert_array_equal(encoder.classes_, ['Hampton', 'Norfolk', 'Virginia Beach'])


def test_inverse_transform_string():
    """Test inverse transform with strings."""
    encoder = GroupEncoder()
    groups = np.array(['Norfolk', 'Hampton', 'Norfolk'])
    
    groups_enc = encoder.fit_transform(groups)
    groups_orig = encoder.inverse_transform(groups_enc)
    
    assert_array_equal(groups_orig, groups)


# =============================================================================
# Fusion pairs
# =============================================================================

def test_transform_pairs_numeric():
    """Test transforming fusion pairs with numeric groups."""
    encoder = GroupEncoder()
    groups = np.array([5, 10, 15])
    encoder.fit(groups)
    
    pairs = np.array([[5, 10], [10, 15]])
    pairs_enc = encoder.transform_pairs(pairs)
    
    assert_array_equal(pairs_enc, [[0, 1], [1, 2]])


def test_transform_pairs_string():
    """Test transforming fusion pairs with string groups."""
    encoder = GroupEncoder()
    groups = np.array(['A', 'B', 'C'])
    encoder.fit(groups)
    
    pairs = np.array([['A', 'B'], ['B', 'C']])
    pairs_enc = encoder.transform_pairs(pairs)
    
    assert_array_equal(pairs_enc, [[0, 1], [1, 2]])


def test_transform_pairs_transpose():
    """Test pairs in (2, n_pairs) format."""
    encoder = GroupEncoder()
    encoder.fit([5, 10, 15])
    
    pairs = np.array([[5, 10], [10, 15]]).T  # Shape (2, 2)
    pairs_enc = encoder.transform_pairs(pairs)
    
    assert pairs_enc.shape == (2, 2)
    assert_array_equal(pairs_enc, [[0, 1], [1, 2]])


def test_inverse_transform_pairs():
    """Test inverse transform for pairs."""
    encoder = GroupEncoder()
    encoder.fit([5, 10, 15])
    
    pairs_enc = np.array([[0, 1], [1, 2]])
    pairs_orig = encoder.inverse_transform_pairs(pairs_enc)
    
    assert_array_equal(pairs_orig, [[5, 10], [10, 15]])


# =============================================================================
# Error cases
# =============================================================================

def test_unfitted_transform_error():
    """Test error when transforming before fitting."""
    encoder = GroupEncoder()
    
    with pytest.raises(ValueError, match="not fitted"):
        encoder.transform([1, 2, 3])


def test_unfitted_inverse_transform_error():
    """Test error when inverse transforming before fitting."""
    encoder = GroupEncoder()
    
    with pytest.raises(ValueError, match="not fitted"):
        encoder.inverse_transform([0, 1, 2])


def test_unfitted_transform_pairs_error():
    """Test error when transforming pairs before fitting."""
    encoder = GroupEncoder()
    
    with pytest.raises(ValueError, match="not fitted"):
        encoder.transform_pairs([[1, 2]])


def test_unknown_labels_error():
    """Test error on unknown labels in transform."""
    encoder = GroupEncoder()
    encoder.fit([5, 10, 15])
    
    with pytest.raises(ValueError, match="Unknown group labels"):
        encoder.transform([5, 10, 20])  # 20 is unknown


def test_unknown_labels_in_pairs_error():
    """Test error on unknown labels in pairs."""
    encoder = GroupEncoder()
    encoder.fit([5, 10, 15])
    
    with pytest.raises(ValueError, match="Unknown labels in fusion_pairs"):
        encoder.transform_pairs([[5, 20]])  # 20 is unknown


def test_out_of_range_inverse_transform():
    """Test error on out of range encoded labels."""
    encoder = GroupEncoder()
    encoder.fit([5, 10, 15])  # n_groups=3, valid range [0, 2]
    
    with pytest.raises(ValueError, match="must be in"):
        encoder.inverse_transform([0, 1, 5])  # 5 is out of range


def test_negative_inverse_transform():
    """Test error on negative encoded labels."""
    encoder = GroupEncoder()
    encoder.fit([5, 10, 15])
    
    with pytest.raises(ValueError, match="must be in"):
        encoder.inverse_transform([0, -1, 2])


def test_2d_groups_error():
    """Test error on 2D groups array."""
    encoder = GroupEncoder()
    
    with pytest.raises(ValueError, match="must be 1D"):
        encoder.fit([[1, 2], [3, 4]])


def test_nan_in_groups_error():
    """Test error on NaN in groups."""
    encoder = GroupEncoder()
    
    with pytest.raises(ValueError, match="cannot contain NaN"):
        encoder.fit([1.0, np.nan, 3.0])


def test_none_in_groups_error():
    """Test error on None in object array."""
    encoder = GroupEncoder()
    
    with pytest.raises(ValueError, match="cannot contain None"):
        encoder.fit(['A', None, 'B'])


def test_pairs_wrong_shape_error():
    """Test error on wrong pairs shape."""
    encoder = GroupEncoder()
    encoder.fit([1, 2, 3])
    
    with pytest.raises(ValueError, match="must be 2D"):
        encoder.transform_pairs([1, 2, 3])  # 1D instead of 2D


# =============================================================================
# Edge cases
# =============================================================================

def test_single_group():
    """Test single unique group."""
    encoder = GroupEncoder()
    groups = np.array([5, 5, 5])
    
    groups_enc = encoder.fit_transform(groups)
    
    assert_array_equal(groups_enc, [0, 0, 0])
    assert encoder.n_groups_ == 1


def test_all_unique_groups():
    """Test all observations in different groups."""
    encoder = GroupEncoder()
    groups = np.array([1, 2, 3, 4, 5])
    
    groups_enc = encoder.fit_transform(groups)
    
    assert_array_equal(groups_enc, [0, 1, 2, 3, 4])
    assert encoder.n_groups_ == 5


def test_mixed_int_types():
    """Test encoder handles different integer types."""
    encoder = GroupEncoder()
    groups_int32 = np.array([1, 2, 3], dtype=np.int32)
    
    groups_enc = encoder.fit_transform(groups_int32)
    
    assert groups_enc.dtype == np.int_
    assert_array_equal(groups_enc, [0, 1, 2])


# =============================================================================
# Integration scenarios
# =============================================================================

def test_census_tract_scenario():
    """Test realistic census tract encoding scenario."""
    # Census tract IDs are often non-contiguous
    tracts = np.array([101, 205, 307, 101, 205, 450])
    
    encoder = GroupEncoder()
    encoded = encoder.fit_transform(tracts)
    
    # Should be [0, 1, 2, 0, 1, 3] in sorted order
    assert_array_equal(encoded, [0, 1, 2, 0, 1, 3])
    assert encoder.n_groups_ == 4
    
    # Decode back
    decoded = encoder.inverse_transform(encoded)
    assert_array_equal(decoded, tracts)


def test_region_names_scenario():
    """Test realistic region names scenario."""
    regions = np.array(['Norfolk', 'Hampton', 'Virginia Beach', 
                       'Norfolk', 'Chesapeake', 'Hampton'])
    
    encoder = GroupEncoder()
    encoded = encoder.fit_transform(regions)
    decoded = encoder.inverse_transform(encoded)
    
    assert_array_equal(decoded, regions)
    assert encoder.n_groups_ == 4


# =============================================================================
# is_fitted and repr
# =============================================================================

def test_is_fitted():
    """Test is_fitted method."""
    encoder = GroupEncoder()
    
    assert not encoder.is_fitted()
    
    encoder.fit([1, 2, 3])
    
    assert encoder.is_fitted()


def test_repr_unfitted():
    """Test repr for unfitted encoder."""
    encoder = GroupEncoder()
    
    assert repr(encoder) == "GroupEncoder(not fitted)"


def test_repr_fitted():
    """Test repr for fitted encoder."""
    encoder = GroupEncoder()
    encoder.fit([1, 2, 3])
    
    repr_str = repr(encoder)
    
    assert "n_groups=3" in repr_str
    assert "classes" in repr_str


def test_repr_many_classes():
    """Test repr truncates long class list."""
    encoder = GroupEncoder()
    encoder.fit(range(100))
    
    repr_str = repr(encoder)
    
    assert "n_groups=100" in repr_str
    assert "..." in repr_str  # Should truncate


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
