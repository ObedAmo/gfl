"""Tests for scalar_fused_lasso module."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from gfl.core.scalar_fused_lasso import (
    solve_scalar_fused_lasso,
    scalar_fused_lasso_objective
)


class TestInputValidation:
    """Test input validation and error handling."""
    
    def test_negative_lambda_raises_error(self):
        """Test that negative regularization parameter raises ValueError."""
        with pytest.raises(ValueError, match="reg_lambda must be non-negative"):
            solve_scalar_fused_lasso(10.0, 50.0, [1.0], [1.0], -0.5)
    
    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched adj_values and weights raise ValueError."""
        with pytest.raises(ValueError, match="must have same size"):
            solve_scalar_fused_lasso(10.0, 50.0, [1.0, 2.0], [1.0], 1.0)
    
    def test_accepts_lists(self):
        """Test that function accepts Python lists."""
        result = solve_scalar_fused_lasso(10.0, 50.0, [3.0, 7.0], [1.0, 1.0], 1.0)
        assert isinstance(result, float)
    
    def test_accepts_tuples(self):
        """Test that function accepts tuples."""
        result = solve_scalar_fused_lasso(10.0, 50.0, (3.0, 7.0), (1.0, 1.0), 1.0)
        assert isinstance(result, float)
    
    def test_accepts_numpy_arrays(self):
        """Test that function accepts numpy arrays."""
        result = solve_scalar_fused_lasso(
            10.0, 50.0, 
            np.array([3.0, 7.0]), 
            np.array([1.0, 1.0]), 
            1.0
        )
        assert isinstance(result, float)


class TestEmptyGroupCase:
    """Test cases when group_size is zero."""
    
    def test_empty_group_no_neighbors(self):
        """When group_size=0 and no neighbors, should return 0."""
        result = solve_scalar_fused_lasso(0.0, 0.0, [], [], 1.0)
        assert result == 0.0
    
    def test_empty_group_with_neighbors(self):
        """When group_size=0 with neighbors, should return weighted median."""
        # Weighted median of [1, 5, 9] with equal weights should be 5
        result = solve_scalar_fused_lasso(0.0, 0.0, [1.0, 5.0, 9.0], [1.0, 1.0, 1.0], 1.0)
        assert_allclose(result, 5.0, rtol=1e-6)
    
    def test_empty_group_weighted_median(self):
        """Test weighted median calculation for empty group."""
        # Weighted median with weights [2, 1, 1] should be closer to 1
        result = solve_scalar_fused_lasso(0.0, 0.0, [1.0, 5.0, 9.0], [2.0, 1.0, 1.0], 1.0)
        assert result <= 5.0  # Should be pulled toward the heavily weighted value


class TestNoNeighborsCase:
    """Test cases with no neighbors."""
    
    def test_no_neighbors_returns_mean(self):
        """With no neighbors, should return group mean."""
        result = solve_scalar_fused_lasso(10.0, 50.0, [], [], 1.0)
        assert_allclose(result, 5.0, rtol=1e-6)
    
    def test_no_neighbors_different_values(self):
        """Test group mean with different values."""
        result = solve_scalar_fused_lasso(20.0, 100.0, [], [], 5.0)
        assert_allclose(result, 5.0, rtol=1e-6)


class TestTwoNeighborsCase:
    """Test closed-form solution with two neighbors."""
    
    def test_two_neighbors_symmetric(self):
        """Test with symmetric neighbors around the mean."""
        # Mean is 5.0, neighbors at 3.0 and 7.0
        result = solve_scalar_fused_lasso(10.0, 50.0, [3.0, 7.0], [1.0, 1.0], 1.0)
        assert_allclose(result, 5.0, rtol=1e-6)
    
    def test_two_neighbors_asymmetric_weights(self):
        """Test with different weights on neighbors."""
        result = solve_scalar_fused_lasso(10.0, 50.0, [3.0, 7.0], [2.0, 1.0], 1.0)
        # Should be pulled more toward the higher-weighted neighbor (3.0)
        assert result < 5.0
    
    def test_two_neighbors_boundary_fusion(self):
        """Test fusion to one of the boundary neighbors."""
        # Strong regularization with unequal weights
        result = solve_scalar_fused_lasso(20.0, 100.0, [1.0, 9.0], [10.0, 1.0], 5.0)
        # Should be pulled strongly toward heavily weighted neighbor
        assert result < 7.0


class TestGeneralCase:
    """Test general algorithm with 3+ neighbors."""
    
    def test_three_neighbors_symmetric(self):
        """Test with three symmetric neighbors."""
        result = solve_scalar_fused_lasso(
            10.0, 50.0, 
            [3.0, 5.0, 7.0], 
            [1.0, 1.0, 1.0], 
            1.0
        )
        # Should be close to median neighbor
        assert 3.0 <= result <= 7.0
    
    def test_many_neighbors_equal_weights(self):
        """Test with many neighbors and equal weights."""
        neighbors = np.linspace(1.0, 9.0, 9)
        weights = np.ones(9)
        result = solve_scalar_fused_lasso(20.0, 100.0, neighbors, weights, 1.0)
        # Should be in reasonable range
        assert 1.0 <= result <= 9.0
    
    def test_general_case_weighted(self):
        """Test general case with non-uniform weights."""
        result = solve_scalar_fused_lasso(
            20.0, 100.0, 
            [1.0, 5.0, 9.0], 
            [2.0, 1.0, 1.0], 
            1.5
        )
        # With these parameters, optimal is at the middle value
        assert_allclose(result, 5.0, rtol=1e-6)


class TestObjectiveFunction:
    """Test the objective function evaluation."""
    
    def test_objective_at_solution(self):
        """Objective at solution should be minimal."""
        group_size = 10.0
        group_sum = 50.0
        adj_vals = np.array([3.0, 7.0])
        weights = np.array([1.0, 1.0])
        reg_lambda = 1.0
        
        x_opt = solve_scalar_fused_lasso(group_size, group_sum, adj_vals, weights, reg_lambda)
        obj_opt = scalar_fused_lasso_objective(x_opt, group_size, group_sum, adj_vals, weights, reg_lambda)
        
        # Check that nearby points have higher objective
        for dx in [-0.1, 0.1]:
            obj_nearby = scalar_fused_lasso_objective(
                x_opt + dx, group_size, group_sum, adj_vals, weights, reg_lambda
            )
            assert obj_nearby >= obj_opt - 1e-6  # Allow small numerical tolerance
    
    def test_objective_components(self):
        """Test individual components of objective function."""
        x = 5.0
        group_size = 10.0
        group_sum = 50.0
        adj_vals = np.array([3.0, 7.0])
        weights = np.array([1.0, 1.0])
        reg_lambda = 1.0
        
        # Compute manually
        quad_term = 0.5 * group_size * x**2 - group_sum * x
        fusion_term = reg_lambda * (weights[0] * abs(x - adj_vals[0]) + 
                                     weights[1] * abs(x - adj_vals[1]))
        expected = quad_term + fusion_term
        
        result = scalar_fused_lasso_objective(x, group_size, group_sum, adj_vals, weights, reg_lambda)
        assert_allclose(result, expected, rtol=1e-10)


class TestEdgeCases:
    """Test edge cases and corner scenarios."""
    
    def test_all_neighbors_equal(self):
        """When all neighbors have same value."""
        result = solve_scalar_fused_lasso(
            10.0, 50.0, 
            [5.0, 5.0, 5.0], 
            [1.0, 1.0, 1.0], 
            1.0
        )
        # Should be exactly the neighbor value
        assert_allclose(result, 5.0, rtol=1e-6)
    
    def test_very_large_lambda(self):
        """Very large regularization should force fusion."""
        result = solve_scalar_fused_lasso(
            10.0, 100.0, 
            [3.0], 
            [1.0], 
            1000.0
        )
        # Should fuse to neighbor despite mean being 10.0
        assert_allclose(result, 3.0, rtol=1e-6)
    
    def test_very_small_lambda(self):
        """Very small regularization should approach group mean."""
        result = solve_scalar_fused_lasso(
            10.0, 50.0, 
            [3.0, 7.0], 
            [1.0, 1.0], 
            1e-6
        )
        # Should be very close to group mean
        assert_allclose(result, 5.0, rtol=1e-4)
    
    def test_zero_weights(self):
        """Test behavior with zero weights (effectively no neighbors)."""
        result = solve_scalar_fused_lasso(
            10.0, 50.0, 
            [3.0, 7.0], 
            [0.0, 0.0], 
            1.0
        )
        # Zero weights means no fusion penalty
        assert_allclose(result, 5.0, rtol=1e-6)


class TestNumericalStability:
    """Test numerical stability and precision."""
    
    def test_large_values(self):
        """Test with large values."""
        result = solve_scalar_fused_lasso(
            1000.0, 5000.0, 
            [3.0, 7.0], 
            [1.0, 1.0], 
            1.0
        )
        assert np.isfinite(result)
    
    def test_small_values(self):
        """Test with very small values."""
        result = solve_scalar_fused_lasso(
            1e-3, 5e-3, 
            [3e-3, 7e-3], 
            [1e-3, 1e-3], 
            1e-3
        )
        assert np.isfinite(result)
    
    def test_mixed_scale_neighbors(self):
        """Test with neighbors at very different scales."""
        result = solve_scalar_fused_lasso(
            10.0, 50.0, 
            [1e-6, 1e6], 
            [1.0, 1.0], 
            1.0
        )
        assert np.isfinite(result)


class TestDocstringExamples:
    """Test examples from the docstring."""
    
    def test_no_neighbors_example(self):
        """Test docstring example: no neighbors."""
        result = solve_scalar_fused_lasso(10.0, 50.0, [], [], 1.0)
        assert_allclose(result, 5.0, rtol=1e-6)
    
    def test_single_neighbor_example(self):
        """Test docstring example: single neighbor."""
        result = solve_scalar_fused_lasso(10.0, 50.0, [3.0], [1.0], 1.0)
        assert_allclose(result, 4.9, rtol=1e-6)
    
    def test_two_neighbors_example(self):
        """Test docstring example: two neighbors."""
        result = solve_scalar_fused_lasso(10.0, 50.0, [3.0, 7.0], [1.0, 1.0], 1.0)
        assert_allclose(result, 5.0, rtol=1e-6)
    
    def test_multiple_neighbors_example(self):
        """Test docstring example: multiple neighbors with different weights."""
        adj_vals = [1.0, 5.0, 9.0]
        ws = [2.0, 1.0, 100.0]
        result = solve_scalar_fused_lasso(20.0, 100.0, adj_vals, ws, 1.5)
        assert_allclose(result, 9.0, rtol=1e-6)


class TestMonotonicity:
    """Test monotonicity properties."""
    
    def test_increasing_lambda_increases_fusion(self):
        """Increasing lambda should increase fusion toward neighbors."""
        group_size = 10.0
        group_sum = 50.0
        adj_vals = [3.0]
        weights = [1.0]
        
        results = []
        for lam in [0.1, 1.0, 10.0]:
            result = solve_scalar_fused_lasso(group_size, group_sum, adj_vals, weights, lam)
            results.append(result)
        
        # Should move monotonically toward neighbor (3.0) as lambda increases
        diffs = [abs(r - 3.0) for r in results]
        assert all(diffs[i] >= diffs[i+1] for i in range(len(diffs)-1))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
