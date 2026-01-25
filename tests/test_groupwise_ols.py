"""Tests for group-wise location estimators"""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from gfl import compute_groupwise_ols
from gfl.core._ols import _trimmed_mean_sorted, _huber_irls, _lts


# =============================================================================
# Tests for _trimmed_mean_sorted
# =============================================================================

def test_trimmed_mean_no_trimming():
    """When trim=0, should return regular mean"""
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = _trimmed_mean_sorted(vals, trim=0.0)
    assert_allclose(result, 3.0)


def test_trimmed_mean_symmetric_trim():
    """Trim 20% from each tail (40% total)"""
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # n=5
    # trim=0.2 -> k=floor(0.2*5)=1, removes 1 from each tail
    result = _trimmed_mean_sorted(vals, trim=0.2)
    expected = np.mean([2.0, 3.0, 4.0])
    assert_allclose(result, expected)


def test_trimmed_mean_small_sample():
    """n <= 2 should return mean"""
    assert_allclose(_trimmed_mean_sorted(np.array([5.0]), trim=0.1), 5.0)
    assert_allclose(_trimmed_mean_sorted(np.array([3.0, 7.0]), trim=0.1), 5.0)


def test_trimmed_mean_fallback_to_median():
    """When 2*k >= n, should return median"""
    vals = np.array([1.0, 2.0, 3.0])  # n=3
    # trim=0.4 -> k=floor(0.4*3)=1, but 2*1 >= 3, so fallback
    result = _trimmed_mean_sorted(vals, trim=0.4)
    assert_allclose(result, 2.0)  # median


def test_trimmed_mean_with_outliers():
    """Should be robust to outliers"""
    vals = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    # trim=0.2 -> removes 100.0 and 1.0
    result = _trimmed_mean_sorted(vals, trim=0.2)
    expected = np.mean([2.0, 3.0, 4.0])
    assert_allclose(result, expected)


def test_trimmed_mean_invalid_trim():
    """trim must be in [0, 0.5)"""
    vals = np.array([1.0, 2.0, 3.0])
    
    with pytest.raises(ValueError, match="trim must be in"):
        _trimmed_mean_sorted(vals, trim=-0.1)
    
    with pytest.raises(ValueError, match="trim must be in"):
        _trimmed_mean_sorted(vals, trim=0.7)


# =============================================================================
# Tests for _huber_irls
# =============================================================================

def test_huber_clean_data():
    """On clean normal data, should be close to mean"""
    np.random.seed(42)
    vals = np.random.normal(10.0, 1.0, size=100)
    result = _huber_irls(vals, delta=1.345, max_iter=100, tol=1e-6)
    assert_allclose(result, vals.mean(), rtol=0.05)


def test_huber_with_outliers():
    """Should downweight outliers"""
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    result = _huber_irls(vals, delta=1.345, max_iter=100, tol=1e-6)
    
    # Should be much closer to median (3.5) than mean (19.17)
    assert result < 10.0
    assert_allclose(result, np.median(vals), atol=2.0)


def test_huber_small_sample():
    """n <= 2 should return mean"""
    assert_allclose(_huber_irls(np.array([5.0]), delta=1.345, max_iter=100, tol=1e-6), 5.0)
    assert_allclose(_huber_irls(np.array([3.0, 7.0]), delta=1.345, max_iter=100, tol=1e-6), 5.0)


def test_huber_convergence():
    """Should converge within max_iter"""
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 50.0])
    result = _huber_irls(vals, delta=1.345, max_iter=50, tol=1e-8)
    assert np.isfinite(result)


# =============================================================================
# Tests for _lts
# =============================================================================

def test_lts_no_trimming():
    """trim=0 should use all data"""
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = _lts(vals, trim=0.0, patience=10, tol=1e-12)
    assert_allclose(result, 3.0)


def test_lts_with_outliers():
    """Should find best-fitting window excluding outliers"""
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    # trim=0.2 -> h=ceil(0.8*6)=5, should find window [1,2,3,4,5]
    result = _lts(vals, trim=0.2, patience=10, tol=1e-12)
    expected = np.mean([1.0, 2.0, 3.0, 4.0, 5.0])
    assert_allclose(result, expected, atol=0.1)


def test_lts_small_sample():
    """n <= 2 should return mean"""
    assert_allclose(_lts(np.array([5.0]), trim=0.1, patience=5, tol=1e-12), 5.0)
    assert_allclose(_lts(np.array([3.0, 7.0]), trim=0.1, patience=5, tol=1e-12), 5.0)


def test_lts_extreme_contamination():
    """Should handle heavy contamination"""
    vals = np.array([10.0, 11.0, 12.0, 100.0, 200.0, 300.0])
    # trim=0.5 -> h=ceil(0.5*6)=3, should find [10, 11, 12]
    result = _lts(vals, trim=0.5, patience=10, tol=1e-12)
    assert_allclose(result, 11.0, atol=1.0)


def test_lts_early_stopping():
    """Early stopping should still give reasonable result"""
    np.random.seed(42)
    vals = np.concatenate([
        np.random.normal(5.0, 0.5, size=50),
        np.random.uniform(50, 100, size=10)
    ])
    
    result_patient = _lts(vals, trim=0.2, patience=100, tol=1e-12)
    result_impatient = _lts(vals, trim=0.2, patience=3, tol=1e-12)
    
    # Should be close even with early stopping
    assert_allclose(result_patient, result_impatient, atol=0.5)


def test_lts_invalid_trim():
    """trim must be in [0, 0.5)"""
    vals = np.array([1.0, 2.0, 3.0])
    
    with pytest.raises(ValueError, match="trim must be in"):
        _lts(vals, trim=-0.1, patience=5, tol=1e-12)
    
    with pytest.raises(ValueError, match="trim must be in"):
        _lts(vals, trim=0.7, patience=5, tol=1e-12)


def test_lts_numerical_stability():
    """Test with large values to ensure numerical stability"""
    vals = np.array([1e6, 1e6 + 1, 1e6 + 2, 1e6 + 3, 1e6 + 100])
    result = _lts(vals, trim=0.2, patience=10, tol=1e-12)
    # Should find window [1e6, 1e6+1, 1e6+2, 1e6+3]
    expected = 1e6 + 1.5
    assert_allclose(result, expected, rtol=1e-10)


# =============================================================================
# Tests for compute_groupwise_ols
# =============================================================================


def test_compute_groupwise_ols_mean():
    """Basic test with mean method"""
    y = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    groups = np.array([0, 0, 0, 1, 1, 1])
    
    result = compute_groupwise_ols(y, groups, method="mean")
    
    expected = np.array([2.0, 11.0])
    assert_allclose(result, expected)


def test_compute_groupwise_ols_median():
    """Test with median method"""
    y = np.array([1.0, 2.0, 100.0, 10.0, 11.0, 12.0])
    groups = np.array([0, 0, 0, 1, 1, 1])
    
    result = compute_groupwise_ols(y, groups, method="median")
    
    expected = np.array([2.0, 11.0])
    assert_allclose(result, expected)


def test_compute_groupwise_ols_all_methods():
    """Test all methods produce reasonable results"""
    np.random.seed(42)
    y = np.concatenate([
        np.random.normal(5.0, 1.0, size=30),
        np.random.normal(10.0, 1.0, size=30)
    ])
    # Add outliers
    y[5] = 50.0
    y[35] = -50.0
    
    groups = np.concatenate([np.zeros(30, dtype=int), np.ones(30, dtype=int)])
    
    methods = ["mean", "median", "trimmed_mean", "huber", "lts"]
    results = {}
    
    for method in methods:
        result = compute_groupwise_ols(y, groups, method=method, trim=0.1)
        results[method] = result
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))
    
    # Robust methods should be closer to each other than to mean
    robust_group0 = [results[m][0] for m in ["median", "trimmed_mean", "huber", "lts"]]
    assert np.std(robust_group0) < 1.0  # Robust methods agree
    
    # Mean should be pulled by outliers
    assert abs(results["mean"][0] - np.median(robust_group0)) > 1.0


def test_compute_groupwise_ols_empty_groups():
    """Empty groups should return NaN"""
    y = np.array([1.0, 2.0, 3.0])
    groups = np.array([0, 0, 2])  # Group 1 is empty
    
    result = compute_groupwise_ols(y, groups, n_groups=3, method="mean")
    
    assert_allclose(result[0], 1.5)
    assert np.isnan(result[1])
    assert_allclose(result[2], 3.0)


def test_compute_groupwise_ols_single_observation_groups():
    """Groups with single observation"""
    y = np.array([1.0, 5.0, 10.0])
    groups = np.array([0, 1, 2])
    
    result = compute_groupwise_ols(y, groups, method="trimmed_mean", trim=0.1)
    
    assert_allclose(result, y)  # Should just return the values


def test_compute_groupwise_ols_large_groups():
    """Test with realistic group sizes"""
    np.random.seed(42)
    n_groups = 10
    group_size = 50
    
    y = np.concatenate([
        np.random.normal(g * 10, 2.0, size=group_size)
        for g in range(n_groups)
    ])
    groups = np.repeat(np.arange(n_groups), group_size)
    
    result = compute_groupwise_ols(y, groups, method="lts", trim=0.1)
    
    assert result.shape == (n_groups,)
    assert np.all(np.isfinite(result))
    # Check monotonicity (groups have increasing means)
    assert np.all(np.diff(result) > 0)


def test_compute_groupwise_ols_check_input_flag():
    """Test that check_input=False skips validation"""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    groups = np.array([0, 0, 1, 1])
    
    result_checked = compute_groupwise_ols(y, groups, method="mean", check_input=True)
    result_unchecked = compute_groupwise_ols(y, groups, method="mean", 
                                              n_groups=2, check_input=False)
    
    assert_allclose(result_checked, result_unchecked)


def test_compute_groupwise_ols_unknown_method():
    """Unknown method should raise ValueError"""
    y = np.array([1.0, 2.0, 3.0])
    groups = np.array([0, 0, 0])
    
    with pytest.raises(ValueError, match="Unknown method"):
        compute_groupwise_ols(y, groups, method="invalid_method")

    
def test_compute_groupwise_ols_reproducibility():
    """Results should be deterministic"""
    np.random.seed(42)
    y = np.random.normal(5.0, 2.0, size=100)
    groups = np.random.randint(0, 5, size=100)
    
    result1 = compute_groupwise_ols(y, groups, method="lts", trim=0.1)
    result2 = compute_groupwise_ols(y, groups, method="lts", trim=0.1)
    
    assert_allclose(result1, result2)


# =============================================================================
# Edge Cases
# =============================================================================

def test_edge_case_all_identical_values():
    """All values in a group are identical"""
    y = np.array([5.0, 5.0, 5.0, 5.0])
    groups = np.zeros(4, dtype=int)
    
    for method in ["mean", "median", "trimmed_mean", "huber", "lts"]:
        result = compute_groupwise_ols(y, groups, method=method)
        assert_allclose(result[0], 5.0)


def test_edge_case_two_distinct_values():
    """Group with only two distinct values"""
    y = np.array([3.0, 3.0, 3.0, 7.0, 7.0, 7.0])
    groups = np.zeros(6, dtype=int)
    
    for method in ["mean", "median", "trimmed_mean", "huber", "lts"]:
        result = compute_groupwise_ols(y, groups, method=method)
        assert_allclose(result[0], 5.0)


def test_edge_case_negative_values():
    """Test with negative values"""
    y = np.array([-10.0, -5.0, -3.0, -2.0, -1.0])
    groups = np.zeros(5, dtype=int)
    
    result = compute_groupwise_ols(y, groups, method="lts", trim=0.2)
    assert result[0] < 0
    assert_allclose(result[0], np.mean([-5.0, -3.0, -2.0, -1.0]), atol=0.5)


# =============================================================================
# Integration/Stress Tests
# =============================================================================

def test_robustness_comparison():
    """Compare robustness of different methods with contamination"""
    np.random.seed(42)
    
    # Clean data: N(10, 1)
    clean = np.random.normal(10.0, 1.0, size=80)
    # Contamination: uniform outliers
    outliers = np.random.uniform(50, 100, size=20)
    y = np.concatenate([clean, outliers])
    groups = np.zeros(100, dtype=int)
    
    results = {}
    for method in ["mean", "median", "trimmed_mean", "huber", "lts"]:
        results[method] = compute_groupwise_ols(y, groups, method=method, trim=0.2)[0]
    
    # Mean should be heavily biased
    assert results["mean"] > 15.0
    
    # Robust methods should be close to true mean (10)
    for method in ["median", "trimmed_mean", "huber", "lts"]:
        assert 9.0 < results[method] < 11.0, f"{method} failed: {results[method]}"


def test_multiple_groups_with_varying_contamination():
    """Different groups with different contamination levels"""
    np.random.seed(42)
    
    # Group 0: clean
    g0 = np.random.normal(5.0, 0.5, size=50)
    # Group 1: 10% contamination
    g1 = np.concatenate([
        np.random.normal(10.0, 0.5, size=45),
        np.random.uniform(50, 100, size=5)
    ])
    # Group 2: 30% contamination
    g2 = np.concatenate([
        np.random.normal(15.0, 0.5, size=35),
        np.random.uniform(50, 100, size=15)
    ])
    
    y = np.concatenate([g0, g1, g2])
    groups = np.concatenate([
        np.zeros(50, dtype=int),
        np.ones(50, dtype=int),
        np.full(50, 2, dtype=int)
    ])
    
    result_mean = compute_groupwise_ols(y, groups, method="mean")
    result_lts = compute_groupwise_ols(y, groups, method="lts", trim=0.3)
    
    # LTS should be closer to true means for contaminated groups
    assert abs(result_lts[0] - 5.0) < 0.5
    assert abs(result_lts[1] - 10.0) < 1.0
    assert abs(result_lts[2] - 15.0) < 1.0
    
    # Mean should be biased for contaminated groups
    assert result_mean[1] > result_lts[1] + 1.0
    assert result_mean[2] > result_lts[2] + 2.0


if __name__ == '__main__':
    pytest.main([__file__, "v", "--tb=short"])
