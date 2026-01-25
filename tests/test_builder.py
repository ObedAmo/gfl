import numpy as np
import pytest
from gfl import build_gfl_structure


def test_build_adaptive_structure():
    """Test basic adaptive structure building."""
    y = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    groups = np.array([0, 0, 0, 1, 1, 1])
    pairs = np.array([[0, 1]])
    
    structure = build_gfl_structure(y, groups, pairs, adaptive=True)
    
    assert structure.n_params == 2
    assert structure.n_pairs == 1
    assert structure.weights[0] < 1.0  # Small weight (dissimilar groups)


def test_build_uniform_structure():
    """Test uniform weight structure building."""
    y = np.array([1.0, 2.0, 10.0, 11.0])
    groups = np.array([0, 0, 1, 1])
    pairs = np.array([[0, 1]])
    
    structure = build_gfl_structure(y, groups, pairs, adaptive=False)
    
    assert structure.weights[0] == 1.0


def test_build_with_robust_ols():
    """Test building with robust initial estimates."""
    y = np.array([1.0, 2.0, 100.0, 10.0, 11.0, 12.0])  # Outlier in group 0
    groups = np.array([0, 0, 0, 1, 1, 1])
    pairs = np.array([[0, 1]])
    
    # Huber should be more robust than mean
    structure_huber = build_gfl_structure(
        y, groups, pairs, 
        adaptive=True, 
        ols_method='huber'
    )
    structure_mean = build_gfl_structure(
        y, groups, pairs,
        adaptive=True,
        ols_method='mean'
    )
    
    # Weights should differ due to outlier influence
    assert structure_huber.weights[0] != structure_mean.weights[0]


def test_build_infers_n_groups():
    """Test that n_groups is correctly inferred."""
    y = np.array([1.0, 2.0, 3.0])
    groups = np.array([0, 1, 2])
    pairs = np.array([[0, 1], [1, 2]])
    
    structure = build_gfl_structure(y, groups, pairs)
    
    assert structure.n_params == 3


def test_build_respects_n_groups_override():
    """Test that explicit n_groups is respected."""
    y = np.array([1.0, 2.0])
    groups = np.array([0, 1])
    pairs = np.array([[0, 1]])
    
    structure = build_gfl_structure(y, groups, pairs, n_groups=5)
    
    assert structure.n_params == 5


def test_build_no_validation():
    """Test fast path with check_input=False."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    groups = np.array([0, 0, 1, 1])
    pairs = np.array([[0, 1]])
    
    # Should run without validation
    structure = build_gfl_structure(
        y, groups, pairs,
        check_input=False
    )
    
    assert structure.n_params == 2


if __name__ == '__main__':
    pytest.main([__file__, "-v", "--tb=short"])
