"""Scalar fused lasso solver for coordinate descent optimization.

This module provides an efficient solver for the scalar subproblem that
arises in coordinate descent algorithms for generalized fused lasso regression.
"""
import numpy as np
from gfl.typing import FloatArray, FloatArrayLike

__all__ = ['solve_scalar_fused_lasso', 'scalar_fused_lasso_objective']

EPS = 1e-8


def solve_scalar_fused_lasso(
        group_size: float,
        group_sum: float,
        adj_values: FloatArrayLike,
        weights: FloatArrayLike,
        reg_lambda: float
) -> float:
    """
    Solve the scalar fused lasso optimization problem.

    This function solves the optimization problem that arises in coordinate
    descent algorithms for generalized fused lasso regression:

        min_x (1/2) n_s x^2 - s x + λ sum_{i}^{r} w_i |x - z_i|

    where n_s is the group size, s is the group sum, z_i are neighbor values,
    and w_i are edge weights.

    The generalized fused lasso extends the original fused lasso to arbitrary 
    structures by penalizing differences across edges defined by any 
    adjacency relationship (spatial neighbors, temporal sequences, network 
    connections, etc.).

    Parameters
    ----------
    group_size : float
        Number of observations in the group (n_s >= 0). Must be non-negative.
    group_sum : float
        Sum of observations in the group (s). Can be any real number.
    adj_values : array_like of shape (n_neighbors,)
        Values of adjacent nodes/groups (z_i).
    weights : array_like of shape (n_neighbors,)
        Edge weights for each neighbor (w_i ≥ 0). Must be non-negative 
        and same length as adj_values.
    reg_lambda : float
        Regularization parameter (λ >= 0). Controls fusion penalty strength.

    Returns
    -------
    float
        Optimal value x* that minimizes the objective function.

    Raises
    ------
    ValueError
        If reg_lambda is negative, or if adj_values and weights have
        different lengths.

    Examples
    --------
    >>> import numpy as np
    >>> from gfl.core.scalar_fused_lasso import solve_scalar_fused_lasso

    No neighbors case (returns group mean):
    
    >>> solve_scalar_fused_lasso(10.0, 50.0, [], [], 1.0)
    5.0
    
    Single neighbor (soft-thresholding):
    
    >>> solve_scalar_fused_lasso(10.0, 50.0, [3.0], [1.0], 1.0)
    4.9
    
    Two neighbors:
    
    >>> solve_scalar_fused_lasso(10.0, 50.0, [3.0, 7.0], [1.0, 1.0], 1.0)
    5.0
    
    Multiple neighbors with different weights:
    
    >>> adj_vals = [1.0, 5.0, 9.0]
    >>> ws = [2.0, 1.0, 100.0]
    >>> solve_scalar_fused_lasso(20.0, 100.0, adj_vals, ws, 1.5)
    9.0

    References
    ----------
    .. [1] Ohishi, M., Fukui, K., Okamura, K., Itoh, Y., & Yanagihara, H. 
           (2021). "Coordinate optimization for generalized fused Lasso." 
           Communications in Statistics-Theory and Methods, 50(24), 5955-5973.
    
    See Also
    --------
    scalar_fused_lasso_objective : Evaluate the objective function at a point
    """
    if reg_lambda < 0:
        raise ValueError(f"reg_lambda must be non-negative, got {reg_lambda}")

    adj_values = np.asarray(adj_values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    n_adj = len(adj_values)

    if len(adj_values) != len(weights):
        raise ValueError(
            f"adj_vals and weights must have same size. "
            f"Got {len(adj_values)} and {len(weights)}"
        )

    # Handle empty group
    if group_size == 0:
        if n_adj == 0:
            return 0.0
        return _weighted_median(adj_values, weights)

    # No neighbor or no regularization: return group mean
    if n_adj == 0 or reg_lambda == 0:
        return group_sum / group_size

    # Single neighbor: soft-thresholding
    if n_adj == 1:
        return _solve_single_neighbor(
            group_size, group_sum,
            adj_values[0], weights[0],
            reg_lambda
        )

    if n_adj == 2:
        return _solve_two_neighbors(
            group_size, group_sum,
            adj_values, weights,
            reg_lambda
        )

    return _solve_general_case(
        group_size, group_sum,
        adj_values, weights,
        reg_lambda
    )


def scalar_fused_lasso_objective(
        x: float,
        group_size: float,
        group_sum: float,
        adj_values: FloatArrayLike,
        weights: FloatArrayLike,
        reg_lambda: float
) -> float:
    """
    Compute the objective value of the fused lasso at a given x.

    Parameters
    ----------
    x : float
        Point at which to evaluate the objective.
    group_size : float
        Number of observations in the group (n_s >= 0). Must be non-negative.
    group_sum : float
        Sum of observations in the group (s). Can be any real number.
    adj_values : array_like of shape (n_neighbors,)
        Values of adjacent nodes/groups (z_i).
    weights : array_like of shape (n_neighbors,)
        Edge weights for each neighbor (w_i ≥ 0). Must be non-negative 
        and same length as adj_values.
    reg_lambda : float
        Regularization parameter (λ >= 0). Controls fusion penalty strength.

    Returns
    -------
    float
        The objective value at x.
    """
    quad_term = 0.5 * group_size * x ** 2 - group_sum * x
    fusion_term = np.sum(weights * np.abs(x - adj_values))
    return quad_term + reg_lambda * fusion_term


# Private helpers

def _solve_single_neighbor(
        group_size: float,
        group_sum: float,
        z1: float,
        w1: float,
        lam: float
) -> float:
    """Solve for single neighbor case (soft-thresholding)."""
    grp_size_inv = 1.0 / group_size
    mean = group_sum * grp_size_inv
    threshold = lam * w1 * grp_size_inv
    diff = mean - z1

    if abs(diff) <= threshold:
        return z1

    return float(mean - threshold * np.sign(diff))


def _solve_two_neighbors(
        group_size: float,
        group_sum: float,
        adj_values: FloatArray,
        weights: FloatArray,
        lam: float
) -> float:
    """Solve for two neighbors case (closed-form three-region solution)."""
    grp_size_inv = 1.0 / group_size

    # Extract and sort neighbors
    z1, z2 = adj_values[0], adj_values[1]
    w1, w2 = weights[0], weights[1]

    if z2 < z1:
        z1, z2 = z2, z1
        w1, w2 = w2, w1

    # Compute weighted lambda terms
    lw1 = lam * w1
    lw2 = lam * w2
    lw_sum = lw1 + lw2
    lw_diff = lw1 - lw2

    # Compute stationary points in each region
    x_left = (group_sum + lw_sum) * grp_size_inv  # x < z1
    x_mid = (group_sum - lw_diff) * grp_size_inv  # z1 <= x <= z2
    x_right = (group_sum - lw_sum) * grp_size_inv  # # x > z2

    # Check interior region
    if z1 < x_mid < z2:
        return x_mid

    # Check left region
    if x_mid <= z1:
        return x_left if x_left < z1 else z1

    # Check right region
    return x_right if x_right > z2 else z2


def _solve_general_case(
        group_size: float,
        group_sum: float,
        adj_vals: FloatArray,
        weights: FloatArray,
        reg_lambda: float
) -> float:
    """Solve for three or more neighbors (general algorithm)."""
    grp_size_inv = 1.0 / group_size

    # Sort neighbors by value
    order = np.argsort(adj_vals)
    z = adj_vals[order]
    ws = weights[order]

    # Compute cumulative weight sums
    wc = np.r_[0.0, np.cumsum(ws)]
    w_total = wc[-1]
    delta = 2 * wc - w_total

    # Compute all stationary points (critical points)
    x_c = (group_sum - reg_lambda * delta) * grp_size_inv

    # Check interior intervals: x* in (z[k-1], z[k])
    interval_bds = np.r_[-np.inf, z, np.inf]
    interior_mask = (
            (x_c > interval_bds[:-1] + EPS) &
            (x_c < interval_bds[1:] - EPS)
    )

    if np.any(interior_mask):
        k = np.where(interior_mask)[0][0]
        return float(x_c[k])

    # Check boundary points: x* = z[k]
    # Optimality condition: x*[k+1] <= z[k] <= x*[k]
    boundary_mask = (
            (x_c[1:] <= z + EPS) &
            (z <= x_c[:-1] + EPS)
    )
    indices = np.where(boundary_mask)[0]

    if len(indices) > 0:
        return float(z[indices[0]])

    # Fallback: evaluate objective at all boundary points
    return _fallback_boundary_search(
        group_size, group_sum, z, ws,
        reg_lambda
    )


def _fallback_boundary_search(
        group_size: float,
        group_sum: float,
        z_sorted: FloatArray,
        ws_sorted: FloatArray,
        lam: float,
) -> float:
    """
    Fallback boundary search when optimality conditions fail.

    This should be very rare and indicates numerical precision issues.
    """
    # Evaluate objective at all boundary points
    obj_vals = np.array([
        scalar_fused_lasso_objective(
            x_k, group_size, group_sum,
            z_sorted, ws_sorted,
            lam
        )
        for x_k in z_sorted
    ])

    k_star = np.argmin(obj_vals)
    return float(z_sorted[k_star])


def _weighted_median(
        values: FloatArray,
        weights: FloatArray
) -> float:
    """
    Compute the weighted median of values with given non-negative weights.

    This solves:
        minimize_x sum_i w[i] * |x - values[i]|

    Parameters
    ----------
    values : ndarray of shape (n,)
        Input values.
    weights : ndarray of shape (n,)
        Non-negative weights associated with values.

    Returns
    -------
    float
        A weighted median of the input values.

    Notes
    -----
    - If all weights sum to zero, falls back to the unweighted median.
    - Assumes values and weights have the same length.
    """
    # Sort by value
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]

    # Cumulative weights
    wc = np.cumsum(w_sorted)
    w_total = wc[-1]

    # Degenerate case: zero total weight
    if w_total <= 0.0:
        return float(v_sorted[len(v_sorted) // 2])

    # First index where cumulative weight exceeds half
    median_idx = np.searchsorted(wc, 0.5 * w_total)
    median_idx = min(int(median_idx), len(v_sorted) - 1)

    return float(v_sorted[median_idx])
