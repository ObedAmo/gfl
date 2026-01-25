"""Builder utilities for GFL structures."""

import numpy as np
import numpy.typing as npt
from typing import Optional, Literal

from gfl.core._structure import GFLStructure
from gfl.core._ols import compute_groupwise_ols
from gfl.core._weights import (
    compute_adaptive_weights,
    compute_uniform_weights
)


def build_gfl_structure(
    y: npt.ArrayLike,
    groups: npt.ArrayLike,
    fusion_pairs: npt.ArrayLike,
    *,
    n_groups: Optional[int] = None,
    adaptive: bool = True,
    ols_method: Literal["mean", "median", "trimmed_mean", "huber", "lts"] = "mean",
    gamma: float = 1.0,
    w_max: Optional[float] = None,
    check_input: bool = True,
) -> GFLStructure:
    """
    Build GFLStructure from data with automatic weight computation.
    
    This convenience function combines initial estimation, weight computation,
    and structure creation into a single call. It chains:
    
    1. compute_groupwise_ols (if adaptive=True) 
    2. compute_adaptive_weights or compute_uniform_weights
    3. GFLStructure construction
    
    For fine-grained control over estimation or weights, call the components
    directly instead of using this builder.
    
    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Response variable (observations).
    groups : array-like of shape (n_samples,)
        Group labels in {0, 1, ..., n_groups-1}.
        Must be contiguous integers. For non-contiguous or string labels,
        use GroupEncoder first.
    fusion_pairs : array-like of shape (n_pairs, 2) or (2, n_pairs)
        Pairs of group indices to fuse. Each row [i, j] represents a potential
        fusion between groups i and j. Must use same encoding as `groups`.
    n_groups : int, optional
        Total number of groups. If None, inferred as max(groups) + 1.
    adaptive : bool, default=True
        If True, compute adaptive weights based on initial OLS estimates.
        If False, use uniform weights (standard fused lasso).
    ols_method : {"mean", "median", "trimmed_mean", "huber", "lts"}, default="mean"
        Method for computing initial group estimates. Only used if adaptive=True.
        - "mean": Group-wise mean (standard OLS)
        - "median": Group-wise median (robust to outliers)
        - "trimmed_mean": Trimmed mean within groups
        - "huber": Huber M-estimator (robust)
        - "lts": Least Trimmed Squares (high breakdown point)
    gamma : float, default=1.0
        Power parameter for adaptive weights: w_ij = 1 / |θ_i - θ_j|^gamma.
        Only used if adaptive=True. Common values: 0.5, 1.0, 2.0.
    w_max : float, optional
        Maximum adaptive weight (caps weights when θ_i ≈ θ_j).
        Only used if adaptive=True. Default is 1e10.
    check_input : bool, default=True
        Whether to validate inputs. Set to False for performance when inputs
        are guaranteed valid (e.g., inside solver iterations).
    
    Returns
    -------
    structure : GFLStructure
        Complete GFL structure ready for solver, containing:
        - fusion_pairs: validated and canonicalized pairs
        - weights: adaptive or uniform weights
        - n_params: total number of groups
    
    Examples
    --------
    >>> import numpy as np
    >>> from gfl import build_gfl_structure
    
    Basic usage with adaptive weights:
    
    >>> y = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    >>> groups = np.array([0, 0, 0, 1, 1, 1])
    >>> pairs = np.array([[0, 1]])
    >>> structure = build_gfl_structure(y, groups, pairs)
    >>> structure.weights  # Small weight (groups are dissimilar)
    array([0.111...])
    
    Uniform weights (standard fused lasso):
    
    >>> structure = build_gfl_structure(y, groups, pairs, adaptive=False)
    >>> structure.weights
    array([1.])
    
    Robust adaptive weights:
    
    >>> structure = build_gfl_structure(
    ...     y, groups, pairs,
    ...     adaptive=True,
    ...     ols_method='huber'
    ... )
    
    For non-contiguous groups, encode first:
    
    >>> from gfl.utils import GroupEncoder
    >>> groups_raw = np.array(['Norfolk', 'Norfolk', 'Hampton', 'Hampton'])
    >>> pairs_raw = np.array([['Norfolk', 'Hampton']])
    >>> encoder = GroupEncoder()
    >>> groups_enc = encoder.fit_transform(groups_raw)
    >>> pairs_enc = encoder.transform_pairs(pairs_raw)
    >>> structure = build_gfl_structure(y, groups_enc, pairs_enc)
    
    See Also
    --------
    GFLStructure : Low-level structure container
    compute_groupwise_ols : Initial estimation
    compute_adaptive_weights : Adaptive weight computation
    compute_uniform_weights : Uniform weight computation
    GroupEncoder : Label encoding for non-contiguous groups
    
    Notes
    -----
    This builder is optimized for common workflows. For advanced use cases:
    
    - Custom weight functions: Compute weights manually and pass to GFLStructure
    - Multiple λ values: Build structure once, reuse across solution path
    - Custom initial estimates: Call compute_adaptive_weights directly
    - Non-standard penalties: Subclass GFLStructure
    
    The check_input parameter is propagated to all internal function calls,
    allowing for a fast path when building structures repeatedly (e.g., in
    cross-validation or bootstrap).
    """
    # Compute weights based on strategy
    if adaptive:
        # Step 1: Compute initial group estimates
        theta_init = compute_groupwise_ols(
            y, 
            groups,
            n_groups=n_groups,
            method=ols_method,
            check_input=check_input
        )
        
        # Step 2: Compute adaptive weights
        weights = compute_adaptive_weights(
            fusion_pairs,
            theta_init,
            gamma=gamma,
            w_max=w_max,
            check_inputs=check_input
        )
    else:
        # Uniform weights (standard fused lasso)
        weights = compute_uniform_weights(
            fusion_pairs,
            check_inputs=check_input
        )
    
    # Step 3: Infer n_groups if not provided
    if n_groups is None:
        groups_arr = np.asarray(groups, dtype=np.int64)
        n_groups = int(groups_arr.max()) + 1
    
    # Step 4: Build and return structure
    return GFLStructure(
        fusion_pairs,
        weights,
        n_params=n_groups,
        check_input=check_input
    )
