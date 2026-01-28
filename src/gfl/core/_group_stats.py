"""Group operation utilities."""

import numpy as np
from typing import Optional, NamedTuple
from gfl.typing import IntArray, FloatArray


class GroupData(NamedTuple):
    """
    Container for group statistics from grouped data.

    Attributes
    ----------
    sizes : ndarray of shape (n_groups,)
        Number of samples in each group.
    sums : ndarray of shape (n_groups,)
        Sum of values for each group.
    means : ndarray of shape (n_groups,)
        Mean of values for each group (0 for empty groups).
    is_grouped_data : bool
        Whether the input corresponds to true grouping
        (False if each observation is its own group).
    """
    sizes: IntArray
    sums: FloatArray
    means: FloatArray
    is_grouped_data: bool


def compute_group_data(
    data: FloatArray,
    groups: IntArray,
    n_groups: Optional[int] = None
) -> GroupData:
    """
    Compute group statistics (sizes, sums, means) for grouped data.

    Parameters
    ----------
    data : ndarray of shape (n_samples,)
        Input data values.
    groups : ndarray of shape (n_samples,)
        Non-negative integer group identifiers.
    n_groups : int, optional
        Total number of groups. If None, inferred as max(group_ids) + 1.

    Returns
    -------
    GroupData
        sizes, sums, means, and grouping flag.
    """
    if data.ndim != 1 or groups.ndim != 1:
        raise ValueError("data and group_ids must be 1D arrays")

    if data.shape[0] != groups.shape[0]:
        raise ValueError(
            f"Length mismatch: data ({data.shape[0]}) != group_ids ({groups.shape[0]})"
        )

    if data.size == 0:
        return GroupData(
            sizes=np.array([], dtype=np.int64),
            sums=np.array([], dtype=np.float64),
            means=np.array([], dtype=np.float64),
            is_grouped_data=False
        )

    if np.any(groups < 0):
        raise ValueError("group_ids must be non-negative integers")

    if n_groups is None:
        n_groups = int(groups.max()) + 1
    else:
        n_groups = int(n_groups)
        if n_groups <= groups.max():
            raise ValueError("n_groups must exceed max(group_ids)")

    # Fast path: one observation per group
    if data.size == n_groups and np.all(groups == np.arange(n_groups)):
        data_f = data.astype(np.float64, copy=False)
        return GroupData(
            sizes=np.ones(n_groups, dtype=np.int64),
            sums=data_f,
            means=data_f,
            is_grouped_data=False
        )

    sizes = np.bincount(groups, minlength=n_groups)
    sums = np.bincount(groups, weights=data, minlength=n_groups).astype(np.float64)
    means = np.divide(sums, sizes, out=np.zeros_like(sums), where=sizes > 0)

    return GroupData(
        sizes=sizes,
        sums=sums,
        means=means,
        is_grouped_data=True
    )
