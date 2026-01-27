import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, Literal, Any

from gfl._types import DuplicateStrategy


def _ensure_array(
    arr: Any, 
    dtype: type, 
    name: str = "array"
) -> npt.NDArray:
    """Convert input to numpy array with specified dtype."""
    try:
        return np.asarray(arr, dtype=dtype)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Failed to convert {name} to {dtype.__name__}: {e}")


def _check_ndim(
    arr: npt.NDArray, 
    expected_dim: int, 
    name: str = "array"
) -> None:
    """Validate array dimensionality."""
    if arr.ndim != expected_dim:
        raise ValueError(
            f"{name} must be {expected_dim}D, "
            f"got {arr.ndim}D array"
        )
    
def _check_length(
    arr: npt.NDArray,
    expected_length: int,
    name: str = "array"
) -> None:
    """Check array length matches expected value."""
    if len(arr) != expected_length:
        raise ValueError(
            f"{name} has length {len(arr)}, expected {expected_length}"
        )

 
def _check_finite(
    arr: npt.NDArray, 
    name: str = "array"
) -> None:
    """Check array contains only finite values."""
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")


def _check_non_negative(
    arr: npt.NDArray, 
    name: str = "array"
) -> None:
    """Check array contains only non-negative values."""
    if np.any(arr < 0):
        raise ValueError(f"{name} must contain only non-negative values")
    

def _check_fitted(obj: Any, attributes: list, name: str = "estimator") -> None:
    """
    Check if object has been fitted by verifying attributes exist.
    
    Parameters
    ----------
    obj : object
        Object to check
    attributes : list of str
        List of attribute names that should exist if fitted
    name : str, default="estimator"
        Name for error messages
        
    Raises
    ------
    ValueError
        If any required attribute is None or doesn't exist
        
    Examples
    --------
    >>> class Encoder:
    ...     def __init__(self):
    ...         self.encoding_ = None
    >>> enc = Encoder()
    >>> _check_fitted(enc, ['encoding_'], 'encoder')  # Raises
    >>> enc.encoding_ = {}
    >>> _check_fitted(enc, ['encoding_'], 'encoder')  # OK
    """
    for attr in attributes:
        if not hasattr(obj, attr) or getattr(obj, attr) is None:
            raise ValueError(
                f"{name} is not fitted. Call fit() before using this method."
            )


def _check_positive(
    arr: npt.NDArray, 
    name: str = "array"
) -> None:
    """Check array contains only positive values."""
    if np.any(arr <= 0):
        raise ValueError(f"{name} must contain only positive values")
    

def _check_bounds(
    arr: npt.NDArray,
    *,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    name: str = "array",
    inclusive: bool = True
) -> None:
    """Check array values within bounds."""
    if lower is not None:
        if inclusive and np.any(arr < lower):
            raise ValueError(f"{name} must be >= {lower}")
        elif not inclusive and np.any(arr <= lower):
            raise ValueError(f"{name} must be > {lower}")
        
    if upper is not None:
        if inclusive and np.any(arr > upper):
            raise ValueError(f"{name} must be <= {upper}")
        elif not inclusive and np.any(arr >= upper):
            raise ValueError(f"{name} must be < {upper}")


def check_array(
    arr: Any,
    name: str,
    dtype: type,
    ndim: int = 1,
    *,
    allow_nan: bool = False,
    check_non_negative: bool = False,
    check_positive: bool = False
) -> npt.NDArray:
    """
    Validate and convert to array with checks.
    
    Parameters
    ----------
    arr : array-like
        Input to validate
    name : str
        Name for error messages
    dtype : type
        Target numpy dtype
    ndim : int, default=1
        Expected number of dimensions
    allow_nan : bool, default=False
        Whether to allow NaN/infinite values
    check_non_negative : bool, default=False
        Whether to check for non-negative values
    check_positive : bool, default=False
        Whether to check for positive values
        
    Returns
    -------
    ndarray
        Validated array
    """
    arr = _ensure_array(arr, dtype, name)
    _check_ndim(arr, ndim, name)

    if not allow_nan:
        _check_finite(arr, name)

    if check_non_negative:
        _check_non_negative(arr, name)

    if check_positive:
        _check_positive(arr, name)

    return arr


def check_1d_array(
    arr: Any,
    name: str,
    dtype: type,
    *,
    allow_nan: bool = False,
    check_non_negative: bool = False,
    check_positive: bool = False
) -> npt.NDArray:
    """
    Validate and convert to 1D array with checks.
    """
    return check_array(
        arr, name, dtype, ndim=1,
        allow_nan=allow_nan,
        check_non_negative=check_non_negative,
        check_positive=check_positive
    )


def check_groups(
    groups: npt.ArrayLike,
    n_samples: int,
    n_groups: Optional[int] = None
) -> Tuple[npt.NDArray[np.int64], int]:
    """
    Validate group assignments are in contiguous format [0, n_groups-1].

    Parameters
    ----------
    groups : array-like of shape (n_samples,)
        Group identifiers for each sample. Must be non-negative integers
        in contiguous format [0, 1, ..., n_groups-1].
        Each observation can belong to its own group or share a group with others.
    n_samples : int
        Expected number of samples
    n_groups : int, optional
        Total number of groups in the structure.
        If None, inferred as the number of unique groups.

    Returns
    -------
    groups : ndarray of shape (n_samples,)
        Validated group identifiers in [0, n_groups-1]
    n_groups : int
        Total number of groups

    Raises
    ------
    ValueError
        If groups are not in contiguous format [0, n_groups-1]
        
    Examples
    --------
    >>> groups = check_groups(np.array([0, 1, 2, 0]), n_samples=4)
    >>> groups
    (array([0, 1, 2, 0]), 3)
    
    >>> # With specified n_groups (some groups may be empty)
    >>> groups = check_groups(np.array([0, 2, 4]), n_samples=3, n_groups=5)
    >>> groups
    (array([0, 2, 4]), 5)
    """
    groups = check_1d_array(
        groups, 
        name="groups",
        dtype=np.int64, 
        check_non_negative=True
    )

    _check_length(groups, expected_length=n_samples, name="groups")

    if len(groups) == 0:
        n_groups = n_groups if n_groups is not None else 0
        return groups, n_groups
    
    unique_groups = np.unique(groups)
    n_unique = len(unique_groups)
    min_group = unique_groups[0]
    max_group = unique_groups[-1]

    # Determine n_groups
    if n_groups is None:
        n_groups = n_unique
        # Check that groups are exactly [0, 1, ..., n_unique-1]
        if min_group != 0 or max_group != n_groups - 1:
            raise ValueError(
                f"groups must be in contiguous format [0, ..., {n_groups-1}]. "
                f"Found groups in range [{min_group}, {max_group}]. "
                "Please relabel your groups to start from 0 with no gaps in observed values. "
                "Consider using GroupEncoder: from gfl.utils import GroupEncoder"
            )
    else:
        # Validate n_groups
        if not isinstance(n_groups, (int, np.integer)) or n_groups <= 0:
            raise ValueError(
                f"n_groups must be a positive integer, got {n_groups}"
            )
        
        if n_unique > n_groups:
            raise ValueError(
                f"groups contains {n_unique} unique values, "
                f"but n_groups={n_groups}. Cannot have more unique groups than n_groups."
            )
        
        # Check that all groups are in valid range [0, n_groups-1]
        if max_group >= n_groups:
            raise ValueError(
                f"groups contains values >= n_groups. "
                f"Max group ID is {max_group}, but n_groups={n_groups}. "
                f"All group IDs must be in [0, {n_groups-1}]."
            )
        
        if min_group != 0:
            raise ValueError(
                f"groups must start from 0. Found minimum group ID = {min_group}. "
                f"Please relabel your groups to [0, ..., {n_groups-1}]."
            )
    
    return groups, n_groups


def _ensure_pairs_format(
    pairs: npt.NDArray,
    name: str = "pairs"
) -> Tuple[npt.NDArray, int]:
    """
    Ensure pairs array is in (n_pairs, 2) format.
    
    Accepts either (n_pairs, 2) or (2, n_pairs) and transposes if needed.
    
    Parameters
    ----------
    pairs : ndarray
        2D array of pairs
    name : str
        Name for error messages
        
    Returns
    -------
    pairs : ndarray of shape (n_pairs, 2)
        Pairs in standard format
    n_pairs : int
        Number of pairs
    """
    if pairs.ndim != 2:
        raise ValueError(f"{name} must be 2D array, got {pairs.ndim}D")
    
    if pairs.shape[1] == 2:
        # Already in (n_pairs, 2) format
        return pairs, pairs.shape[0]
    elif pairs.shape[0] == 2:
        # In (2, n_pairs) format, transpose
        return pairs.T, pairs.shape[1]
    else:
        raise ValueError(
            f"{name} must have shape (n_pairs, 2) or (2, n_pairs), "
            f"got {pairs.shape}"
        )


def _canonicalize_pairs(
    fusion_pairs: npt.NDArray[np.int64],
    weights: npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """
    Remove self-loops and enforce undirected ordering (i < j).
    """
    # Remove self-loops
    mask = fusion_pairs[:, 0] != fusion_pairs[:, 1]
    fusion_pairs = fusion_pairs[mask]
    weights = weights[mask]

    if fusion_pairs.size == 0:
        return fusion_pairs.reshape(0, 2), weights[:0]
    
    # Enforce i < j
    i, j = fusion_pairs[:, 0], fusion_pairs[:, 1]
    u = np.minimum(i, j)
    v = np.maximum(i, j)
    fusion_pairs = np.column_stack((u, v))

    return fusion_pairs, weights


def _sort_pairs(
    fusion_pairs: npt.NDArray[np.int64],
    weights: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Lexicographically sort fusion pairs."""
    order = np.lexsort((fusion_pairs[:, 1], fusion_pairs[:, 0]))
    return fusion_pairs[order], weights[order]


def _has_duplicates(fusion_pairs: npt.NDArray[np.int64]):
    """Check if sorted fusion pairs contains duplicates."""
    if fusion_pairs.shape[0] <= 1:
        return False
    
    return np.any((fusion_pairs[1:] == fusion_pairs[:-1]).all(axis=1))


def _aggregate_weights(
        weights: npt.NDArray[np.float64],
        group_id: npt.NDArray[np.intp],
        strategy: str
) -> npt.NDArray[np.float64]:
    """Aggregate duplicate weights according to strategy."""
    n_groups = group_id.max() + 1
    
    if strategy == "first":
        idx = np.searchsorted(group_id, np.arange(n_groups))
        return weights[idx]
    
    if strategy == "sum":
        return np.bincount(group_id, weights=weights, minlength=n_groups)
    
    if strategy == "mean":
        sums = np.bincount(group_id, weights=weights, minlength=n_groups)
        counts = np.bincount(group_id, minlength=n_groups)
        return sums / counts
    
    if strategy == "max":
        out = np.full(n_groups, -np.inf)
        np.maximum.at(out, group_id, weights)
        return out
    
    if strategy == "min":
        out = np.full(n_groups, np.inf)
        np.minimum.at(out, group_id, weights)
        return out
    
    raise RuntimeError("Unhandled duplicate aggregation strategy")


def check_fusion_pairs(
        fusion_pairs: npt.ArrayLike,
        weights: npt.ArrayLike,
        *,
        n_params: Optional[int] = None,
        duplicate_strategy: DuplicateStrategy= 'first'
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], int]:
    """
    Validate and canonicalize fusion pairs and weights.

    Parameters
    ----------
    fusion_pairs : array-like of shape (n_pairs, 2) or (2, n_pairs)
        Fusion index pairs (i, j)
    weights : array-like of shape (n_pairs,)
        Positive fusion weights.
    n_params : int, optional
        Total number of parameters. If None, inferred as max(fusion_pairs) + 1
    uplicate_strategy : {'first','sum','mean','max','min'}, default='first'
        Strategy used to resolve duplicate fusion pairs:
        - 'first': Keep weight from first occurrence
        - 'sum': Sum weights of duplicates
        - 'mean': Average weights of duplicates
        - 'max': Take maximum weight
        - 'min': Take minimum weight

    Returns
    -------
    fusion_pairs : ndarray of shape (n_unique_pairs, 2)
        Canonicalized fusion pairs with i < j, sorted, deduplicated.
    weights : ndarray of shape (n_unique_pairs,)
        Corresponding weights after aggregation.
    n_params : int
        Number of parameters (inferred or user-provided).

    Examples
    --------
    >>> pairs = np.array([[1, 0], [0, 1], [1, 1], [2, 0]])
    >>> weights = np.array([1.0, 2.0, 0.5, 1.5])
    >>> pairs_norm, weights_norm, n = check_fusion_pairs(pairs, weights)
    >>> pairs_norm
    array([[0, 1],
           [0, 2]])
    >>> weights_norm
    array([1., 1.5])
    >>> n
    3

    Notes
    -----
    - Self-loops (i, i) are automatically removed
    - Pairs are ordered such that i < j
    - Time complexity: O(E log E) where E is the number of pairs
    """
    # Validate fusion_pairs
    fusion_pairs = check_array(
        fusion_pairs, name="fusion_pairs", 
        dtype=np.int64,
        ndim=2,
        check_non_negative=True
    )
    
    # Ensure correct format and get n_pairs
    fusion_pairs, n_pairs = _ensure_pairs_format(fusion_pairs, "fusion_pairs")

    # validate weights
    weights = check_1d_array(
        weights, 
        name="weights", 
        dtype=np.float64,
        allow_nan=False,
        check_positive=True
    )

    # Check length match
    _check_length(weights, n_pairs, "weights")

    # Check bounds if n_params provided
    if n_params is not None:
        if not isinstance(n_params, (int, np.integer)) or n_params <= 0:
            raise ValueError(f"n_params must be a positive integer, got {n_params}")
        
        _check_bounds(fusion_pairs, upper=n_params - 1, name="fusion_pairs")

    # Canonicalized
    fusion_pairs, weights = _canonicalize_pairs(fusion_pairs, weights)

    if fusion_pairs.size == 0:
        inferred_n_params = 0 if n_params is None else n_params
        return fusion_pairs, weights, inferred_n_params
    
    fusion_pairs, weights = _sort_pairs(fusion_pairs, weights)

    # Deduplicate if needed
    if _has_duplicates(fusion_pairs):
        view = np.ascontiguousarray(fusion_pairs).view(
        np.dtype((np.void, fusion_pairs.dtype.itemsize * 2))
        )
        _, unique_idx, group_id = np.unique(view, return_index=True, return_inverse=True)
        group_id = group_id.ravel()

        fusion_pairs = fusion_pairs[unique_idx]
        weights = _aggregate_weights(weights, group_id, duplicate_strategy)

    # Infer n_params if needed
    inferred_n_params = fusion_pairs.max() + 1 if n_params is None else n_params

    return fusion_pairs, weights, inferred_n_params


def check_groups_and_pairs(
    groups: npt.ArrayLike,
    n_samples: int,
    fusion_pairs: npt.ArrayLike,
    weights: npt.ArrayLike,
    n_groups: Optional[int] = None,
    duplicate_strategy: DuplicateStrategy = 'first'
) -> Tuple[npt.NDArray[np.int64], int, npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """
    Validate groups and fusion pairs together, ensuring consistency.
    
    Ensures that:
    - groups ∈ {0, …, n_groups-1}
    - fusion_pairs ⊆ {0, …, n_groups-1} x {0, …, n_groups-1}
    
    Parameters
    ----------
    groups : array-like of shape (n_samples,)
        Group identifiers
    n_samples : int
        Expected number of samples
    fusion_pairs : array-like of shape (n_pairs, 2) or (2, n_pairs)
        Fusion index pairs
    weights : array-like of shape (n_pairs,)
        Fusion weights
    n_groups : int, optional
        Total number of groups
    duplicate_strategy : {'first','sum','mean','max','min'}, default='first'
        How to handle duplicate fusion pairs
        
    Returns
    -------
    groups : ndarray of shape (n_samples,)
        Validated groups in [0, n_groups-1]
    n_groups : int
        Number of groups (= n_params)
    fusion_pairs : ndarray of shape (n_unique_pairs, 2)
        Canonicalized fusion pairs
    weights : ndarray of shape (n_unique_pairs,)
        Aggregated weights
        
    Raises
    ------
    ValueError
        If groups and fusion_pairs are inconsistent
        
    Examples
    --------
    >>> groups = np.array([0, 1, 2, 0])
    >>> pairs = np.array([[0, 1], [1, 2]])
    >>> weights = np.array([1.0, 2.0])
    >>> g, n, p, w = check_groups_and_pairs(groups, 4, pairs, weights)
    >>> n
    3
    """
    # Validate groups first
    groups, n_groups = check_groups(groups, n_samples, n_groups)
    
    # Validate fusion pairs with n_params = n_groups
    fusion_pairs, weights, n_params_inferred = check_fusion_pairs(
        fusion_pairs, weights, 
        n_params=n_groups,
        duplicate_strategy=duplicate_strategy
    )
    
    # Ensure consistency (should always pass if above validations passed)
    if n_params_inferred != n_groups:
        raise ValueError(
            f"Inconsistency detected: fusion_pairs infer n_params={n_params_inferred}, "
            f"but n_groups={n_groups}"
        )
    
    return groups, n_groups, fusion_pairs, weights
