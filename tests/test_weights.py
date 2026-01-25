"""Tests for adaptive weights and other weight functions"""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from gfl import (
    compute_adaptive_weights,
    compute_uniform_weights,
    compute_distance_weights,
)


# =============================================================================
# Tests for compute_adaptive_weights
# =============================================================================

def test_adaptive_weights_basic():
    """Basic adaptive weight computation"""
    fusion_pairs = np.array([[0, 1], [1, 2]])
    theta_init = np.array([1.0, 5.0, 10.0])
    
    weights = compute_adaptive_weights(fusion_pairs, theta_init, gamma=1.0)
    
    # w_01 = 1/|1-5| = 1/4 = 0.25
    # w_12 = 1/|5-10| = 1/5 = 0.2
    assert_allclose(weights, [0.25, 0.2])


def test_adaptive_weights_gamma_effect():
    """Different gamma values produce different weights"""
    fusion_pairs = np.array([[0, 1]])
    theta_init = np.array([0.0, 2.0])
    
    w_gamma_05 = compute_adaptive_weights(fusion_pairs, theta_init, gamma=0.5)
    w_gamma_10 = compute_adaptive_weights(fusion_pairs, theta_init, gamma=1.0)
    w_gamma_20 = compute_adaptive_weights(fusion_pairs, theta_init, gamma=2.0)
    
    # Higher gamma -> more aggressive downweighting
    assert w_gamma_05[0] > w_gamma_10[0] > w_gamma_20[0]


def test_adaptive_weights_identical_groups():
    """Identical initial estimates should give maximum weight (w_max)"""
    fusion_pairs = np.array([[0, 1]])
    theta_init = np.array([5.0, 5.0])
    
    weights = compute_adaptive_weights(fusion_pairs, theta_init, w_max=1e10)
    
    assert_allclose(weights[0], 1e10)


def test_adaptive_weights_w_max_capping():
    """w_max should cap weights when diff is very small"""
    fusion_pairs = np.array([[0, 1], [1, 2]])
    theta_init = np.array([1.0, 1.0001, 10.0])
    
    weights = compute_adaptive_weights(fusion_pairs, theta_init, w_max=100.0)
    
    # First weight should be capped at 100
    assert weights[0] <= 100.0
    # Second weight should be normal (not capped)
    assert weights[1] < 100.0


def test_adaptive_weights_with_nan():
    """NaN in theta_init (empty groups) should give neutral weight of 1.0"""
    fusion_pairs = np.array([[0, 1], [1, 2]])
    theta_init = np.array([5.0, np.nan, 10.0])
    
    weights = compute_adaptive_weights(fusion_pairs, theta_init)
    
    # Pair with NaN should have weight 1.0
    assert_allclose(weights[0], 1.0)


def test_adaptive_weights_input_formats():
    """Should accept both (n_pairs, 2) and (2, n_pairs) formats"""
    theta_init = np.array([1.0, 5.0, 10.0])
    
    pairs_row = np.array([[0, 1], [1, 2]])  # (2, 2)
    pairs_col = np.array([[0, 1], [1, 2]]).T  # (2, 2) transposed
    
    w1 = compute_adaptive_weights(pairs_row, theta_init)
    w2 = compute_adaptive_weights(pairs_col, theta_init)
    
    assert_allclose(w1, w2)


def test_adaptive_weights_invalid_gamma():
    """gamma must be positive"""
    fusion_pairs = np.array([[0, 1]])
    theta_init = np.array([1.0, 2.0])
    
    with pytest.raises(ValueError, match="gamma must be positive"):
        compute_adaptive_weights(fusion_pairs, theta_init, gamma=0.0)
    
    with pytest.raises(ValueError, match="gamma must be positive"):
        compute_adaptive_weights(fusion_pairs, theta_init, gamma=-1.0)


def test_adaptive_weights_invalid_w_max():
    """w_max must be positive if provided"""
    fusion_pairs = np.array([[0, 1]])
    theta_init = np.array([1.0, 2.0])
    
    with pytest.raises(ValueError, match="w_max must be positive"):
        compute_adaptive_weights(fusion_pairs, theta_init, w_max=0.0)
    
    with pytest.raises(ValueError, match="w_max must be positive"):
        compute_adaptive_weights(fusion_pairs, theta_init, w_max=-10.0)


# =============================================================================
# Tests for compute_uniform_weights
# =============================================================================

def test_uniform_weights_default():
    """Default uniform weights should all be 1.0"""
    fusion_pairs = np.array([[0, 1], [1, 2], [2, 3]])
    
    weights = compute_uniform_weights(fusion_pairs)
    
    assert weights.shape == (3,)
    assert_allclose(weights, [1.0, 1.0, 1.0])


def test_uniform_weights_custom_value():
    """Custom value should be applied to all weights"""
    fusion_pairs = np.array([[0, 1], [1, 2]])
    
    weights = compute_uniform_weights(fusion_pairs, value=2.5)
    
    assert_allclose(weights, [2.5, 2.5])


def test_uniform_weights_single_pair():
    """Should work with single fusion pair"""
    fusion_pairs = np.array([[0, 1]])
    
    weights = compute_uniform_weights(fusion_pairs, value=3.0)
    
    assert weights.shape == (1,)
    assert_allclose(weights[0], 3.0)


def test_uniform_weights_invalid_value():
    """value must be positive and finite"""
    fusion_pairs = np.array([[0, 1]])
    
    with pytest.raises(ValueError, match="value must be a positive finite number"):
        compute_uniform_weights(fusion_pairs, value=0.0)
    
    with pytest.raises(ValueError, match="value must be a positive finite number"):
        compute_uniform_weights(fusion_pairs, value=-1.0)
    
    with pytest.raises(ValueError, match="value must be a positive finite number"):
        compute_uniform_weights(fusion_pairs, value=np.inf)


# =============================================================================
# Tests for compute_distance_weights
# =============================================================================

def test_distance_weights_basic():
    """Basic distance weight computation"""
    fusion_pairs = np.array([[0, 1], [1, 2]])
    theta_init = np.array([1.0, 5.0, 10.0])
    
    weights = compute_distance_weights(fusion_pairs, theta_init)
    
    # w_01 = |1-5| = 4
    # w_12 = |5-10| = 5
    assert_allclose(weights, [4.0, 5.0])


def test_distance_weights_inverse_of_adaptive():
    """Distance weights should be inverse relationship to adaptive weights"""
    fusion_pairs = np.array([[0, 1], [1, 2]])
    theta_init = np.array([1.0, 5.0, 10.0])
    
    dist_weights = compute_distance_weights(fusion_pairs, theta_init)
    adaptive_weights = compute_adaptive_weights(fusion_pairs, theta_init, gamma=1.0, w_max=1e10)
    
    # dist_weights * adaptive_weights should be approximately constant (1.0)
    # (ignoring w_max effects)
    products = dist_weights * adaptive_weights
    assert_allclose(products, [1.0, 1.0], rtol=1e-6)


def test_distance_weights_identical_groups():
    """Identical groups should give zero distance"""
    fusion_pairs = np.array([[0, 1]])
    theta_init = np.array([5.0, 5.0])
    
    weights = compute_distance_weights(fusion_pairs, theta_init)
    
    assert_allclose(weights[0], 0.0)


def test_distance_weights_with_nan():
    """NaN in theta_init should give zero weight"""
    fusion_pairs = np.array([[0, 1], [1, 2]])
    theta_init = np.array([5.0, np.nan, 10.0])
    
    weights = compute_distance_weights(fusion_pairs, theta_init)
    
    # Pair with NaN should have weight 0.0
    assert_allclose(weights[0], 0.0)


def test_distance_weights_symmetry():
    """Distance should be symmetric: d(i,j) = d(j,i)"""
    theta_init = np.array([1.0, 5.0])
    
    pairs_ij = np.array([[0, 1]])
    pairs_ji = np.array([[1, 0]])
    
    w_ij = compute_distance_weights(pairs_ij, theta_init)
    w_ji = compute_distance_weights(pairs_ji, theta_init)
    
    assert_allclose(w_ij, w_ji)


# =============================================================================
# Integration Tests
# =============================================================================

def test_weight_functions_consistency():
    """All weight functions should handle same fusion_pairs"""
    fusion_pairs = np.array([[0, 1], [1, 2], [2, 3]])
    theta_init = np.array([1.0, 3.0, 7.0, 15.0])
    
    w_uniform = compute_uniform_weights(fusion_pairs)
    w_adaptive = compute_adaptive_weights(fusion_pairs, theta_init)
    w_distance = compute_distance_weights(fusion_pairs, theta_init)
    
    # All should return same shape
    assert w_uniform.shape == w_adaptive.shape == w_distance.shape == (3,)
    
    # All should be positive
    assert np.all(w_uniform > 0)
    assert np.all(w_adaptive > 0)
    assert np.all(w_distance > 0)


def test_check_inputs_flag():
    """check_inputs=False should skip validation"""
    fusion_pairs = np.array([[0, 1]])
    theta_init = np.array([1.0, 2.0])
    
    # Should work with check_inputs=False
    w1 = compute_uniform_weights(fusion_pairs, check_inputs=False)
    w2 = compute_adaptive_weights(fusion_pairs, theta_init, check_inputs=False)
    w3 = compute_distance_weights(fusion_pairs, theta_init, check_inputs=False)
    
    assert w1.shape == w2.shape == w3.shape == (1,)


if __name__ == '__main__':
    pytest.main([__file__, "-v", "--tb=short"])
