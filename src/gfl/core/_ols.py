"""Group-wise location estimators (1D)"""
import warnings
import numpy as np
import numpy.typing as npt
from typing import Literal, Optional

from gfl.utils import check_1d_array, check_groups
from gfl._types import OLSMethod


def _trimmed_mean_sorted(
    vals: npt.NDArray[np.float64], 
    trim: float
) -> float:
    """
    Trimmed mean by sorting and removing k from each tail where 
    k = floor(trim*n)
    """
    if not (0.0 <= trim <= 0.5):
        raise ValueError("trim must be in [0, 0.5]")
    
    n = len(vals)
    if n <= 2:
        return vals.mean()
    
    k = int(np.floor(trim * n))
    vals_sorted = np.sort(vals)
    if 2 * k >= n:
        # Too few observations to trim; fall back to median
        return np.median(vals)
    if k > 0:
        vals_sorted = vals_sorted[k:-k] # Trim k from each tail
    
    return vals_sorted.mean()


def _huber_irls(
    vals: npt.NDArray[np.float64],
    delta: float,
    max_iter: int,
    tol: float
) -> float:
    """Huber 1D M-estimator via IRLS, starting from median (robust)."""
    if len(vals) <= 2:
        return vals.mean()
    
    theta = np.median(vals)
    for _ in range(max_iter):
        r = vals - theta
        w = np.ones_like(r)
        mask = np.abs(r) > delta
        w[mask] = delta / np.abs(r[mask])

        theta_new = np.sum(w * vals) / np.sum(w)

        if abs(theta_new - theta) < tol:
            break
        theta = theta_new
    else:
        warnings.warn(
            f"Huber IRLS did not converge after {max_iter} iterations.",
            stacklevel=2
        )

    return float(theta)


def _lts(
    vals: npt.NDArray[np.float64], 
    trim: float,
    patience: int,
    tol: float
) -> float:
    """
    1D Least Trimmed Squares (LTS) location estimator using a sliding-window
    algorithm with Fast-LTS early stopping heuristic. Returns the exact LTS
    solution if all windows are scanned; early stopping may yield near-optimal
    solution.

    It minimizes the sum of square errors over all contiguous windows of length
    h, where h = ceil((1 - trim) * n) and returns the window mean.
    """
    if not (0.0 <= trim <= 0.5):
        raise ValueError("trim must be in [0, 0.5]")
    
    n = len(vals)
    if n <= 2:
        return vals.mean()
    
    h = int(np.ceil((1.0 - trim) * n))
    h = max(1, min(h, n))

    vals_sorted = np.sort(vals)

    # Prefix sums of values and squares
    ps1 = np.concatenate(([0.0], np.cumsum(vals_sorted)))
    ps2 = np.concatenate(([0.0], np.cumsum(vals_sorted ** 2)))

    best_sse = np.inf
    best_mean = vals_sorted[n // 2]     # Fall back
    no_improve = 0

    for k in range(n - h + 1):
        S1 = ps1[k + h] - ps1[k]
        S2 = ps2[k + h] - ps2[k]

        # SSE = sum(v - mean)^2 = S2 - 2*mean*S1 + h*mean^2 = S2 - h * mean^2
        # since S1 = h * mean -> sse = S2 - S1 * S1 / h
        sse = S2 - S1 * S1 / h

        if sse < best_sse - tol:
            best_sse = sse
            best_mean = S1 / h
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
        
    return float(best_mean)


def compute_groupwise_ols(
        y: npt.ArrayLike,
        groups: npt.ArrayLike,
        *,
        n_groups: Optional[int] = None,
        method: OLSMethod = "mean",
        trim: float = 0.1,
        huber_delta: float = 1.345,
        max_iter: int = 100,
        tol: float = 1e-6,
        lts_patience: int = 5,
        lts_tol: float = 1e-12,
        check_input: bool = True
) -> npt.NDArray[np.float64]: 
    """
    Compute group-wise unpenalized estimates (group-wise OLS and robust variants).

    This function estimates one scalar parameter per group under the model

        y_{gi} = θ_g + ε_{gi},

    **without any fusion or graph penalty**. In the classical case
    (`method="mean"`), this corresponds exactly to group-wise OLS.
    Robust alternatives (median, trimmed mean, Huber, and LTS) are provided
    to mitigate the influence of outliers prior to adaptive fusion.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Observed responses.
    groups : array-like of shape (n_samples,)
        Group labels in {0, ..., n_groups-1}.
    n_groups : int, optional
        Number of groups. If None, inferred from `groups`.
    method : {"mean", "median", "trimmed_mean", "huber", "lts"}
        Group-wise estimation method.
    trim : float, default=0.1
        Trimming fraction (0 <= trim <= 0.5).
        - For trimmed_mean: fraction to remove from *each tail*. When trim=0.5,
        effectively returns the median.
        - For LTS: fraction to discard *total* (trim=0.5 is the maximum breakdown point).
    huber_delta : float, default=1.345
        Huber tuning constant. 
    max_iter : int, default=100
        Maximum IRLS iterations for Huber estimation.
    tol : float, default=1e-6
        Convergence tolerance for the Huber IRLS algorithm.
    lts_patience : int, default=5
        Early stopping patience for the Fast-LTS sliding-window algorithm.
    lts_tol : float, default=1e-12
        Numerical tolerance for Fast-LTS early stopping. A new window is
        considered an improvement only if it reduces the LTS objective by
        at least `lts_tol`. This parameter affects early stopping only
        and does not change the exact LTS solution when all windows are scanned.
    check_input: bool, default=True
        Whether to perform input validation. If False, the function assumes
        that `y`, `groups`, and `n_groups` are already consistent and skips
        all checks. This is intended for internal use (e.g., when called
        repeatedly inside a solver) to avoid unnecessary overhead.

    Returns
    -------
    theta : ndarray of shape (n_groups,)
        Group-wise unpenalized estimates. Empty groups return NaN.

    Notes
    -----
    - No group is ever discarded; trimming is applied within groups only.
    - `tol` (Huber) and `lts_tol` (Fast-LTS) serve different purposes.
    - The Fast-LTS patience parameter enables early stopping. With `patience=5`,
      the algorithm stops if 5 consecutive windows don't improve the objective.
      For guaranteed exact LTS solution, set `patience=n` (but slower).
    - Small groups fall back to mean.
    """
    if check_input:
        y = check_1d_array(y, name='y', dtype=np.float64)
        groups, n_groups = check_groups(groups, n_samples=len(y), n_groups=n_groups)
    else:
        if n_groups is None:
            n_groups = int(groups.max()) + 1

    theta = np.empty(n_groups, dtype=np.float64)
    method = method.lower()

    for grp in range(n_groups):
        mask = groups == grp

        if not np.any(mask):
            theta[grp] = np.nan
            continue

        vals = y[mask]

        if method == "mean":
            theta[grp] = np.mean(vals)
        elif method == "median":
            theta[grp] = np.median(vals)
        elif method == "trimmed_mean":
            theta[grp] = _trimmed_mean_sorted(vals, trim=trim)
        elif method == "huber":
            theta[grp] = _huber_irls(
                vals, delta=huber_delta,
                max_iter=max_iter, tol=tol
            )
        elif method == "lts":
            theta[grp] = _lts(
                vals,
                trim=trim,
                patience=lts_patience,
                tol=lts_tol
            )
        else:
            raise ValueError(f"Unknown method: {method}. Choose one of the following "
                             "[mean, median, trimmed_mean, huber, lts]")
    
    return theta
