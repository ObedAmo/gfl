"""Tests for UnionFind data structure."""
import pytest
from gfl.utils import UnionFind


# =============================================================================
# Basic Initialization Tests
# =============================================================================

def test_initialization():
    """Test initialization of UnionFind."""
    uf = UnionFind(10)
    assert uf.n_elements == 10
    assert uf.n_components == 10
    assert len(uf) == 10


def test_initial_state():
    """Test that each element is its own parent initially."""
    uf = UnionFind(5)
    for i in range(5):
        assert uf.find(i) == i
    assert uf.n_components == 5


def test_single_element():
    """Test UnionFind with a single element."""
    uf = UnionFind(1)
    assert uf.n_elements == 1
    assert uf.find(0) == 0
    assert uf.n_components == 1


# =============================================================================
# Core Operations Tests
# =============================================================================

def test_simple_union():
    """Test basic union operation."""
    uf = UnionFind(5)
    assert uf.union(0, 1)
    assert uf.find(0) == uf.find(1)
    assert uf.n_components == 4


def test_union_already_connected():
    """Test union on already connected elements."""
    uf = UnionFind(5)
    uf.union(0, 1)
    assert not uf.union(0, 1)
    assert uf.n_components == 4


def test_transitive_union():
    """Test transitive connectivity through unions."""
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(2)
    assert uf.n_components == 3


def test_find_after_union():
    """Test find operation after union."""
    uf = UnionFind(5)
    uf.union(0, 1)
    root = uf.find(0)
    assert uf.find(1) == root


def test_path_compression():
    """Test that path compression flattens the tree."""
    uf = UnionFind(6)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(2, 3)
    root = uf.find(3)
    for i in range(4):
        assert uf.find(i) == root


# =============================================================================
# Connectivity Tests
# =============================================================================

def test_connected_same_element():
    """Test that an element is connected to itself."""
    uf = UnionFind(5)
    assert uf.connected(0, 0)
    assert uf.connected(3, 3)


def test_connected_after_union():
    """Test connectivity after union operation."""
    uf = UnionFind(5)
    uf.union(0, 1)
    assert uf.connected(0, 1)
    assert uf.connected(1, 0)


def test_not_connected():
    """Test that unconnected elements return False."""
    uf = UnionFind(5)
    uf.union(0, 1)
    assert not uf.connected(0, 2)
    assert not uf.connected(3, 4)


# =============================================================================
# Component Analysis Tests
# =============================================================================

def test_get_components_after_unions():
    """Test retrieving components after multiple unions."""
    uf = UnionFind(10)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(5, 6)

    components = uf.get_components()
    root_0 = uf.find(0)
    root_5 = uf.find(5)

    assert set(components[root_0]) == {0, 1, 2}
    assert set(components[root_5]) == {5, 6}


def test_get_component_sizes():
    """Test component size calculation."""
    uf = UnionFind(10)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(5, 6)

    sizes = uf.get_component_sizes()
    assert sizes[uf.find(0)] == 3
    assert sizes[uf.find(5)] == 2


def test_get_largest_component():
    """Test retrieval of largest component."""
    uf = UnionFind(8)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(2, 3)
    uf.union(5, 6)

    largest = uf.get_largest_component()
    assert set(largest) == {0, 1, 2, 3}


def test_get_multi_element_components():
    """Test retrieval of multi-element components only."""
    uf = UnionFind(8)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(5, 6)

    multi = uf.get_multi_element_components()
    sizes = sorted(len(v) for v in multi.values())
    assert sizes == [2, 3]


# =============================================================================
# Edge Cases
# =============================================================================

def test_all_elements_in_one_component():
    """Test when all elements are union into one component."""
    uf = UnionFind(5)
    for i in range(4):
        uf.union(i, i + 1)

    assert uf.n_components == 1
    comps = uf.get_components()
    root = uf.find(0)
    assert set(comps[root]) == {0, 1, 2, 3, 4}


def test_repeated_unions_same_pair():
    """Test repeated union operations on the same pair."""
    uf = UnionFind(4)
    assert uf.union(0, 1)
    assert not uf.union(0, 1)
    assert not uf.union(1, 0)
    assert uf.n_components == 3


# =============================================================================
# Real-World Scenarios
# =============================================================================

def test_graph_connected_components():
    """Test simulating graph connected components."""
    # Simulated graph: 0–1–2, 3–4, 5
    uf = UnionFind(6)
    for u, v in [(0, 1), (1, 2), (3, 4)]:
        uf.union(u, v)

    comps = [set(v) for v in uf.get_components().values()]
    assert {0, 1, 2} in comps
    assert {3, 4} in comps
    assert {5} in comps


def test_large_union_find():
    """Test performance with large number of elements."""
    n = 5000
    uf = UnionFind(n)
    for i in range(0, n - 1, 2):
        uf.union(i, i + 1)

    assert uf.n_components == n // 2
    assert uf.connected(0, 1)
    assert not uf.connected(0, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
