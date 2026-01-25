"""Tests for GFLStructure"""
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.sparse import csr_array

from gfl import GFLStructure


# =============================================================================
# Basic Construction Tests
# =============================================================================

def test_basic_construction():
    """Test basic GFLStructure construction."""
    pairs = np.array([[0, 1], [1, 2], [0, 2]])
    weights = np.array([1.0, 2.0, 1.5])
    
    gfl = GFLStructure(pairs, weights)
    
    assert gfl.n_params == 3
    assert gfl.n_pairs == 3
    assert_array_equal(gfl.fusion_pairs, [[0, 1], [0, 2], [1, 2]])
    assert_array_equal(gfl.weights, [1.0, 1.5, 2.0])


def test_construction_with_explicit_n_params():
    """Test construction with explicit n_params."""
    pairs = np.array([[0, 1], [1, 2]])
    weights = np.array([1.0, 2.0])
    
    gfl = GFLStructure(pairs, weights, n_params=10)
    
    assert gfl.n_params == 10
    assert gfl.n_pairs == 2


def test_construction_transpose_input():
    """Test construction with transposed input."""
    pairs = np.array([[0, 1, 2], [1, 2, 3]])  # Shape (2, 3)
    weights = np.array([1.0, 2.0, 3.0])
    
    gfl = GFLStructure(pairs, weights)
    
    assert gfl.n_params == 4
    assert gfl.n_pairs == 3


def test_construction_no_check():
    """Test construction with check_input=False."""
    pairs = np.array([[0, 1], [1, 2]])
    weights = np.array([1.0, 2.0])
    
    gfl = GFLStructure(pairs, weights, check_input=False)
    
    assert gfl.n_params == 3
    assert gfl.n_pairs == 2


def test_empty_structure():
    """Test empty GFLStructure."""
    pairs = np.array([]).reshape(0, 2)
    weights = np.array([])
    
    gfl = GFLStructure(pairs, weights)
    
    assert gfl.n_params == 0
    assert gfl.n_pairs == 0
    assert len(gfl) == 0


# =============================================================================
# Neighbor Query Tests
# =============================================================================

def test_get_neighbors():
    """Test neighbor queries."""
    pairs = np.array([[0, 1], [1, 2], [0, 2]])
    weights = np.array([1.0, 2.0, 1.5])
    
    gfl = GFLStructure(pairs, weights)
    
    assert_array_equal(gfl.get_neighbors(0), [1, 2])
    assert_array_equal(gfl.get_neighbors(1), [0, 2])
    assert_array_equal(gfl.get_neighbors(2), [0, 1])


def test_get_neighbor_weights():
    """Test neighbor weight queries."""
    pairs = np.array([[0, 1], [1, 2], [0, 2]])
    weights = np.array([1.0, 2.0, 1.5])
    
    gfl = GFLStructure(pairs, weights)
    
    assert_array_almost_equal(gfl.get_neighbor_weights(0), [1.0, 1.5])
    assert_array_almost_equal(gfl.get_neighbor_weights(1), [1.0, 2.0])
    assert_array_almost_equal(gfl.get_neighbor_weights(2), [1.5, 2.0])


def test_get_neighbor_data():
    """Test combined neighbor/weight queries."""
    pairs = np.array([[0, 1], [1, 2]])
    weights = np.array([1.0, 2.0])
    
    gfl = GFLStructure(pairs, weights)
    
    neighbors, ws = gfl.get_neighbor_data(1)
    assert_array_equal(neighbors, [0, 2])
    assert_array_almost_equal(ws, [1.0, 2.0])


def test_neighbors_isolated_node():
    """Test queries on isolated node."""
    pairs = np.array([[0, 1], [2, 3]])
    weights = np.array([1.0, 2.0])
    
    gfl = GFLStructure(pairs, weights, n_params=5)
    
    # Node 4 is isolated
    assert_array_equal(gfl.get_neighbors(4), [])
    assert_array_equal(gfl.get_neighbor_weights(4), [])
    neighbors, ws = gfl.get_neighbor_data(4)
    assert len(neighbors) == 0
    assert len(ws) == 0


def test_neighbor_query_invalid_index():
    """Test that invalid indices raise IndexError."""
    pairs = np.array([[0, 1]])
    weights = np.array([1.0])
    
    gfl = GFLStructure(pairs, weights)
    
    with pytest.raises(IndexError, match="out of range"):
        gfl.get_neighbors(5)
    
    with pytest.raises(IndexError, match="out of range"):
        gfl.get_neighbors(-1)


# =============================================================================
# Degree Tests
# =============================================================================

def test_degree():
    """Test degree queries."""
    pairs = np.array([[0, 1], [1, 2], [0, 2]])
    weights = np.array([1.0, 2.0, 1.5])
    
    gfl = GFLStructure(pairs, weights)
    
    assert gfl.degree(0) == 2
    assert gfl.degree(1) == 2
    assert gfl.degree(2) == 2


def test_degrees():
    """Test degrees for all nodes."""
    pairs = np.array([[0, 1], [1, 2], [0, 2]])
    weights = np.array([1.0, 2.0, 1.5])
    
    gfl = GFLStructure(pairs, weights, n_params=5)
    
    degrees = gfl.degrees()
    assert_array_equal(degrees, [2, 2, 2, 0, 0])


def test_degree_invalid_index():
    """Test that degree queries validate indices."""
    pairs = np.array([[0, 1]])
    weights = np.array([1.0])
    
    gfl = GFLStructure(pairs, weights)
    
    with pytest.raises(IndexError):
        gfl.degree(10)


# =============================================================================
# Matrix Conversion Tests
# =============================================================================

def test_to_adjacency_matrix():
    """Test adjacency matrix conversion."""
    pairs = np.array([[0, 1], [1, 2]])
    weights = np.array([1.0, 2.0])
    
    gfl = GFLStructure(pairs, weights)
    
    adj = gfl.to_adjacency_matrix()
    
    assert isinstance(adj, csr_array)
    assert adj.shape == (3, 3)
    
    # Check symmetry
    assert adj[0, 1] == 1.0
    assert adj[1, 0] == 1.0
    assert adj[1, 2] == 2.0
    assert adj[2, 1] == 2.0
    
    # Check zeros
    assert adj[0, 2] == 0.0
    assert adj[2, 0] == 0.0


def test_to_incidence_matrix_weighted():
    """Test weighted incidence matrix."""
    pairs = np.array([[0, 1], [1, 2]])
    weights = np.array([1.0, 2.0])
    
    gfl = GFLStructure(pairs, weights)
    
    B = gfl.to_incidence_matrix(weighted=True)
    
    assert B.shape == (3, 2)
    
    # First pair (0, 1) with weight 1.0
    assert B[0, 0] == 1.0
    assert B[1, 0] == -1.0
    
    # Second pair (1, 2) with weight 2.0
    assert B[1, 1] == 2.0
    assert B[2, 1] == -2.0


def test_to_incidence_matrix_unweighted():
    """Test unweighted incidence matrix."""
    pairs = np.array([[0, 1], [1, 2]])
    weights = np.array([1.0, 2.0])
    
    gfl = GFLStructure(pairs, weights)
    
    B = gfl.to_incidence_matrix(weighted=False)
    
    assert B.shape == (3, 2)
    
    # All weights should be ±1
    assert B[0, 0] == 1.0
    assert B[1, 0] == -1.0
    assert B[1, 1] == 1.0
    assert B[2, 1] == -1.0


def test_incidence_matrix_empty():
    """Test incidence matrix for empty structure."""
    pairs = np.array([]).reshape(0, 2)
    weights = np.array([])
    
    gfl = GFLStructure(pairs, weights, n_params=3)
    
    B = gfl.to_incidence_matrix()
    
    assert B.shape == (3, 0)


def test_to_adjacency_list():
    """Test adjacency list conversion."""
    pairs = np.array([[0, 1], [1, 2]])
    weights = np.array([1.0, 2.0])
    
    gfl = GFLStructure(pairs, weights)
    
    adj_list = gfl.to_adjacency_list()
    
    assert len(adj_list) == 3
    
    # Node 0
    neighbors, ws = adj_list[0]
    assert_array_equal(neighbors, [1])
    assert_array_almost_equal(ws, [1.0])
    
    # Node 1
    neighbors, ws = adj_list[1]
    assert_array_equal(neighbors, [0, 2])
    assert_array_almost_equal(ws, [1.0, 2.0])
    
    # Node 2
    neighbors, ws = adj_list[2]
    assert_array_equal(neighbors, [1])
    assert_array_almost_equal(ws, [2.0])


def test_to_adjacency_dict():
    """Test adjacency dict conversion."""
    pairs = np.array([[0, 1], [1, 2]])
    weights = np.array([1.0, 2.0])
    
    gfl = GFLStructure(pairs, weights)
    
    adj_dict = gfl.to_adjacency_dict()
    
    assert len(adj_dict) == 3
    assert adj_dict[0] == {1: 1.0}
    assert adj_dict[1] == {0: 1.0, 2: 2.0}
    assert adj_dict[2] == {1: 2.0}


# =============================================================================
# Duplicate Strategy Tests
# =============================================================================

def test_duplicate_strategy_sum():
    """Test sum strategy for duplicates."""
    pairs = np.array([[0, 1], [1, 0]])
    weights = np.array([1.0, 2.0])
    
    gfl = GFLStructure(pairs, weights, duplicate_strategy='sum')
    
    assert gfl.n_pairs == 1
    assert gfl.weights[0] == 3.0


def test_duplicate_strategy_max():
    """Test max strategy for duplicates."""
    pairs = np.array([[0, 1], [1, 0]])
    weights = np.array([1.0, 2.0])
    
    gfl = GFLStructure(pairs, weights, duplicate_strategy='max')
    
    assert gfl.n_pairs == 1
    assert gfl.weights[0] == 2.0


def test_self_loops_removed():
    """Test that self-loops are removed during validation."""
    pairs = np.array([[0, 1], [1, 1], [2, 2]])
    weights = np.array([1.0, 2.0, 3.0])
    
    gfl = GFLStructure(pairs, weights)
    
    # Only (0, 1) should remain
    assert gfl.n_pairs == 1
    assert_array_equal(gfl.fusion_pairs, [[0, 1]])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
