"""Adaptive weights and other weights for GFL"""
import numpy as np
import numpy.typing as npt
from typing import Optional

from gfl.utils.validation import check_array, _ensure_pairs_format


def compute_adaptive_weights(
        fusion_pairs: npt.ArrayLike,
        theta_init: npt.ArrayLike,
        *,
        gamma: float = 1.0,
        w_max: Optional[float] = None,
        check_inputs: bool = True
) -> npt.NDArray[np.float64]:
    """
    Compute adaptive fusion weights using a Zou-Hastie style scheme.

    Adaptive weights down-weight edges between dissimilar groups, encouraging
    fusion only between groups with similar initial estimates. This implements:

        w_ij = 1 / |θ_i - θ_j|^γ

    where θ_i and θ_j are initial unpenalized estimates (typically from
    `compute_groupwise_ols`). Larger initial differences yield smaller weights,
    resulting in less fusion penalty and allowing those groups to remain separate.
    
    When θ_i = θ_j (identical estimates), the weight is capped at `w_max` if
    provided, or set to a large finite value otherwise.

    Parameters
    ----------
    fusion_pairs : array-like of shape (n_pairs, 2) or (2, n_pairs)
        Pairs of group indices to fuse. Each row [i, j] represents a potential
        fusion between groups i and j.
    theta_init : array-like of shape (n_groups,)
        Initial unpenalized estimates for each group, typically obtained via
        `compute_groupwise_ols`. May contain NaN for empty groups.
    gamma : float, default=1.0
        Power parameter controlling weight decay. Higher gamma -> more aggressive
        down-weighting of dissimilar pairs. Common values: 0.5, 1.0, 2.0.
    w_max : float, optional
        Maximum allowed weight (upper cap). Useful to prevent extreme weights
        when θ_i ≈ θ_j. If None, defaults to 1e10 to handle numerical edge cases.
    check_inputs : bool, default=True
        Whether to validate inputs. Set to False for performance when inputs
        are guaranteed valid (e.g., internal solver iterations).

    Returns
    -------
    weights : ndarray of shape (n_pairs,)
        Adaptive weights for each fusion pair. Higher weights -> stronger fusion
        penalty -> groups more likely to be fused.

    Notes
    -----
    - For pairs involving empty groups (NaN in theta_init), a neutral weight
      of 1.0 is assigned.
    - When θ_i = θ_j exactly, the weight is capped at `w_max` (default 1e10)
      to avoid division by zero while maintaining strong fusion preference.
    - The choice of gamma affects the oracle properties of the fused lasso:
      gamma=1.0 is standard; gamma=0.5 can improve finite-sample performance.

    References
    ----------
    Zou, H. (2006). The adaptive lasso and its oracle properties.
    Journal of the American Statistical Association, 101(476), 1418-1429.

    Examples
    --------
    >>> from gfl import compute_groupwise_ols, compute_adaptive_weights
    >>> import numpy as np
    >>> y = np.array([1, 2, 3, 10, 11, 12])
    >>> groups = np.array([0, 0, 0, 1, 1, 1])
    >>> fusion_pairs = np.array([[0, 1]])
    >>> 
    >>> # Step 1: Get initial estimates
    >>> theta_init = compute_groupwise_ols(y, groups, method="mean")
    >>> # theta_init ≈ [2.0, 11.0]
    >>> 
    >>> # Step 2: Compute adaptive weights
    >>> weights = compute_adaptive_weights(fusion_pairs, theta_init, gamma=1.0)
    >>> # weights ≈ 1/9 ≈ 0.111 (small weight, won't fuse easily)
    """
    if check_inputs:
        fusion_pairs = check_array(
            fusion_pairs, 
            name='fusion_pairs',
            dtype=np.int64,
            ndim=2,
            check_non_negative=True
        )
        fusion_pairs, _ = _ensure_pairs_format(fusion_pairs, name='fusion_pairs')
        theta_init = check_array(
            theta_init, 
            name='theta_init',
            dtype=np.float64,
            ndim=1,
            allow_nan=True
        )
        
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        if w_max is not None and w_max <= 0:
            raise ValueError(f"w_max must be positive, got {w_max}")

    # Default w_max to handle division by zero cases
    if w_max is None:
        w_max = 1e10

    u, v = fusion_pairs[:, 0], fusion_pairs[:, 1]
    diff = np.abs(theta_init[u] - theta_init[v])
    
    # Handle NaN (empty groups) - assign neutral weight
    nan_mask = np.isnan(diff)
    if np.any(nan_mask):
        diff[nan_mask] = 1.0  # Will result in weight = 1.0

    # Compute weights with optimized paths for common gamma values
    if gamma == 1.0:
        weights = np.divide(1.0, diff, out=np.full_like(diff, w_max), where=diff > 0)
    elif gamma == 2.0:
        diff_sq = diff * diff
        weights = np.divide(1.0, diff_sq, out=np.full_like(diff, w_max), where=diff_sq > 0)
    elif gamma == 0.5:
        diff_sqrt = np.sqrt(diff)
        weights = np.divide(1.0, diff_sqrt, out=np.full_like(diff, w_max), where=diff_sqrt > 0)
    else:
        diff_pow = np.power(diff, gamma)
        weights = np.divide(1.0, diff_pow, out=np.full_like(diff, w_max), where=diff_pow > 0)

    # Cap weights at w_max (handles near-zero differences)
    weights = np.minimum(weights, w_max)

    return weights


def compute_uniform_weights(
        fusion_pairs: npt.ArrayLike,
        *,
        value: float = 1.0,
        check_inputs: bool = True
) -> npt.NDArray[np.float64]:
    """
    Compute uniform weights for all fusion pairs.

    All edges receive the same weight, treating all potential fusions equally.
    This is the standard (non-adaptive) generalized fused lasso formulation.

    Parameters
    ----------
    fusion_pairs : array-like of shape (n_pairs, 2) or (2, n_pairs)
        Pairs of group indices to fuse. Each row [i, j] represents a potential
        fusion between groups i and j.
    value : float, default=1.0
        Constant weight value to assign to all fusion pairs.
    check_inputs : bool, default=True
        Whether to validate inputs. Set to False for performance when inputs
        are guaranteed valid.

    Returns
    -------
    weights : ndarray of shape (n_pairs,)
        Uniform weights for each fusion pair, all equal to `value`.

    Examples
    --------
    >>> import numpy as np
    >>> from gfl import compute_uniform_weights
    >>> fusion_pairs = np.array([[0, 1], [1, 2], [2, 3]])
    >>> weights = compute_uniform_weights(fusion_pairs, value=1.0)
    >>> weights
    array([1., 1., 1.])
    """
    if check_inputs:
        fusion_pairs = check_array(
            fusion_pairs, 
            name='fusion_pairs',
            dtype=np.int64,
            ndim=2,
            check_non_negative=True
        )
        fusion_pairs, n_pairs = _ensure_pairs_format(fusion_pairs, name='fusion_pairs')
        
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"value must be a positive finite number, got {value}")
    else:
        n_pairs = fusion_pairs.shape[0]

    return np.full(n_pairs, value, dtype=np.float64)


def compute_distance_weights(
        fusion_pairs: npt.ArrayLike,
        theta_init: npt.ArrayLike,
        *,
        check_inputs: bool = True
) -> npt.NDArray[np.float64]:
    """
    Compute L2-based distance weights measuring dissimilarity between groups.

    The weight for each edge (i, j) is the absolute difference |θ_i - θ_j|,
    where θ_i and θ_j are initial estimates. Larger differences -> larger weights
    -> stronger edge weights for constructing minimum spanning trees or measuring
    dissimilarity in the fused penalty.

    This is the inverse of adaptive weights: dissimilar groups get HIGH weights,
    making them suitable for MST construction where edge weights represent costs
    or distances.

    Parameters
    ----------
    fusion_pairs : array-like of shape (n_pairs, 2) or (2, n_pairs)
        Pairs of group indices to fuse. Each row [i, j] represents a potential
        fusion between groups i and j.
    theta_init : array-like of shape (n_groups,)
        Initial unpenalized estimates for each group, typically obtained via
        `compute_groupwise_ols`. May contain NaN for empty groups.
    check_inputs : bool, default=True
        Whether to validate inputs. Set to False for performance when inputs
        are guaranteed valid.

    Returns
    -------
    weights : ndarray of shape (n_pairs,)
        Distance-based weights for each fusion pair. Higher weights indicate
        greater dissimilarity between groups.

    Notes
    -----
    - For pairs involving empty groups (NaN in theta_init), weight is set to 0.
    - These weights are suitable for:
      * Minimum spanning tree (MST) construction
      * Measuring dissimilarity in spatial graphs
      * Edge pruning (remove high-weight edges = dissimilar groups)

    Examples
    --------
    >>> import numpy as np
    >>> from gfl import compute_groupwise_ols, compute_distance_weights
    >>> y = np.array([1, 2, 3, 10, 11, 12])
    >>> groups = np.array([0, 0, 0, 1, 1, 1])
    >>> fusion_pairs = np.array([[0, 1]])
    >>> 
    >>> # Get initial estimates
    >>> theta_init = compute_groupwise_ols(y, groups, method="mean")
    >>> # theta_init ≈ [2.0, 11.0]
    >>> 
    >>> # Compute distance weights
    >>> weights = compute_distance_weights(fusion_pairs, theta_init)
    >>> # weights ≈ [9.0] (large distance = high dissimilarity)
    """
    if check_inputs:
        fusion_pairs = check_array(
            fusion_pairs, 
            name='fusion_pairs',
            dtype=np.int64,
            ndim=2,
            check_non_negative=True
        )
        fusion_pairs, _ = _ensure_pairs_format(fusion_pairs, name='fusion_pairs')
        theta_init = check_array(
            theta_init, 
            name='theta_init',
            dtype=np.float64,
            ndim=1,
            allow_nan=True
        )

    u, v = fusion_pairs[:, 0], fusion_pairs[:, 1]
    weights = np.abs(theta_init[u] - theta_init[v])
    
    # Handle NaN (empty groups) - assign zero weight
    nan_mask = np.isnan(weights)
    if np.any(nan_mask):
        weights[nan_mask] = 0.0

    return weights
