"""
Generalized Fused Lasso (GFL) structure.

This module defines `GFLStructure`, a lightweight container for specifying
pairwise fusion penalties between model parameters. Fusion pairs indicate
which coefficients are coupled by the penalty.

The structure validates and canonicalizes fusion pairs, and provides
derived representations (e.g., CSR-based neighbor access) for efficient
solver implementations.
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, Literal, List, Dict
from scipy.sparse import csr_array, coo_array

from gfl.utils import check_fusion_pairs
from gfl._types import DuplicateStrategy


class GFLStructure:
    """
    Structural container for generalized fused lasso fusion pairs.

    Parameters
    ----------
    fusion_pairs : array-like of shape (n_pairs, 2) or (2, n_pairs)
        Pair indices (i, j) indicating which parameters are coupled by the penalty.
    weights : array-like of shape (n_pairs,)
        Positive fusion weights for each pair.
    n_params : int, optional
        Total number of parameters. If None, inferred from fusion_pairs.
    check_input : bool, default=True
        Whether to validate and canonicalize input.
    duplicate_strategy : {"first","sum","mean","max","min"}, default="first"
        Strategy for combining weights for duplicate pairs during checking.

    Attributes
    ----------
    fusion_pairs : ndarray of shape (n_pairs, 2)
        Canonicalized fusion pairs.
    weights : ndarray of shape (n_pairs,)
        Corresponding weights.
    n_pairs : int
        Number of stored fusion pairs.
    n_params : int
        Parameter dimension.

    Examples
    --------
    >>> pairs = np.array([[0, 1], [1, 2], [0, 2]])
    >>> weights = np.array([1.0, 2.0, 1.5])
    >>> gfl = GFLStructure(pairs, weights)
    >>> gfl.get_neighbors(1)
    array([0, 2])
    >>> gfl.degree(1)
    2

    Notes
    -----
    - Neighbor/degree queries are implemented via a symmetric CSR matrix.
    - The CSR representation is *derived* and does not change the conceptual meaning
      of fusion pairs.
    """
    def __init__(
            self,
            fusion_pairs: npt.ArrayLike,
            weights: npt.ArrayLike,
            *,
            n_params: Optional[int] = None,
            check_input: bool = True,
            duplicate_strategy: DuplicateStrategy = "first"
    ):
        if check_input:
            fusion_pairs_arr, weights_arr, inferred_n_params = check_fusion_pairs(
                fusion_pairs, 
                weights, 
                n_params=n_params, 
                duplicate_strategy=duplicate_strategy
            )
            if n_params is None:
                n_params = inferred_n_params
            else:
                # Keep user-provided n_params
                pass
        else:
            fusion_pairs_arr = np.asarray(fusion_pairs, dtype=np.int64)
            weights_arr = np.asarray(weights, dtype=np.float64)
            if (fusion_pairs_arr.ndim == 2 and 
                fusion_pairs_arr.shape[0] == 2 and 
                fusion_pairs_arr.shape[1] != 2):
                fusion_pairs_arr = fusion_pairs_arr.T
            if n_params is None:
                n_params = int(fusion_pairs_arr.max() + 1) if fusion_pairs_arr.size else 0
        

        self.fusion_pairs = fusion_pairs_arr
        self.weights = weights_arr
        self.n_pairs = int(self.fusion_pairs.shape[0])
        self.n_params = int(n_params)

        self._build_csr()

    def _build_csr(self) -> None:
        """
        Build a symmetric CSR adjacency-like matrix for fast neighbor queries.

        Each fusion pair (i, j) contributes two entries:
          - (i, j) with weight w
          - (j, i) with weight w
        """
        n = self.n_params
        if self.n_pairs == 0:
            arr = csr_array((n, n), dtype=np.float64)
            self._csr_arr = arr
            return
        
        u = self.fusion_pairs[:, 0]
        v = self.fusion_pairs[:, 1]
        w = self.weights

        # Symmetric expansion
        rows = np.concatenate([u, v])
        cols = np.concatenate([v, u])
        data = np.concatenate([w, w])

        arr = coo_array((data, (rows, cols)), shape=(n, n), dtype=np.float64).tocsr()

        self._csr_arr = arr
    
    def get_neighbors(self, i: int) -> npt.NDArray[np.int64]:
        """
        Return neighbors of parameter i.
        
        Parameters
        ----------
        i : int
            Parameter index.
        
        Returns
        -------
        neighbors : ndarray
            Array of neighbor indices.
        """
        self._validate_index(i)
        start, end = self._csr_arr.indptr[i], self._csr_arr.indptr[i + 1]
        return self._csr_arr.indices[start:end]

    def get_neighbor_weights(self, i: int) -> npt.NDArray[np.float64]:
        """
        Return the weights aligned with `get_neighbors(i)`
        
        Parameters
        ----------
        i : int
            Parameter index.
        
        Returns
        -------
        weights : ndarray
            Weights corresponding to each neighbor
        """
        self._validate_index(i)
        start, end = self._csr_arr.indptr[i], self._csr_arr.indptr[i + 1]
        return self._csr_arr.data[start:end]

    def get_neighbor_data(self, i: int) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """
        Return (neighbors, weights) for parameter i.

        Returns
        -------
        neighbors : ndarray[int64]
        weights : ndarray[float64]
        """
        self._validate_index(i)
        start, end = self._csr_arr.indptr[i], self._csr_arr.indptr[i + 1]
        return self._csr_arr.indices[start:end], self._csr_arr.data[start:end]

    def degree(self, i: int) -> int:
        """
        Degree (number of neighbors) of parameter i.

        Parameters
        ----------
        i : int
            Parameter index.
        
        Returns
        -------
        degree : int
            Number of neighbors.
        """
        self._validate_index(i)
        return self._csr_arr.indptr[i + 1] - self._csr_arr.indptr[i]

    def degrees(self) -> npt.NDArray[np.int64]:
        """
        Degree of every parameter.

        Returns
        -------
        degrees : ndarray of shape (n_params,)
            Number of neigbors for each parameter.
        """
        return np.diff(self._csr_arr.indptr)

    def to_adjacency_matrix(self) -> csr_array:
        """Return the sparse adjacency matrix"""
        return self._csr_arr

    def to_incidence_matrix(self, *, weighted: bool = True) -> csr_array:
        """
        Build fusion pair incidence matrix.
        
        Parameters
        ----------
        weighted : bool, default=True
            Whether to scale by fusion weights.
        
        Returns
        -------
        B : csr_array of shape (n_params, n_pairs)
            Incidence matrix where for fusion pair k = (i_k, j_k):
            - B[i_k, k] = +w_k (or +1 if not weighted)
            - B[j_k, k] = -w_k (or -1 if not weighted)
        
        Notes
        -----
        This matrix represents the discrete difference operator.
        For coefficient vector θ, B.T @ θ gives pairwise differences θ_i - θ_j.
        """
        m = self.n_pairs
        n = self.n_params
        if m == 0:
            return csr_array((n, m), dtype=np.float64)
        
        i, j = self.fusion_pairs[:, 0], self.fusion_pairs[:, 1]
        
        # Build properly interleaved COO data
        # For each pair k: (i[k], k) gets +w[k], (j[k], k) gets -w[k]
        rows = np.empty(2 * m, dtype=np.int64)
        rows[0::2] = i  # Even positions: first index
        rows[1::2] = j  # Odd positions: second index
        
        cols = np.repeat(np.arange(m, dtype=np.int64), 2)
        
        if weighted:
            data = np.empty(2 * m, dtype=np.float64)
            data[0::2] = self.weights   # Even positions: +w
            data[1::2] = -self.weights  # Odd positions: -w
        else:
            data = np.tile([1.0, -1.0], m)
        
        return coo_array((data, (rows, cols)), shape=(n, m), dtype=np.float64).tocsr()

    def to_adjacency_list(self) -> List[Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]]:
        """
        Convert to an adjacency-list representation.

        Returns
        -------
        adjacency : list of tuples
            adjacency[i] = (neighbors_i, weights_i)
            where neighbors_i and weights_i are 1D arrays aligned by position.
        """
        return [self.get_neighbor_data(i) for i in range(self.n_params)]

    def to_adjacency_dict(self) -> Dict[int, Dict[int, float]]:
        """
        Convert to nested dictionary representation.

        Returns
        -------
        adj_dict : dict
            Nested dictionary where adj_dict[u][v] = weight of pair (u, v).
        """
        adj_dict = {i: {} for i in range(self.n_params)}
        for i in range(self.n_params):
            nbrs, ws = self.get_neighbor_data(i)
            for j, w in zip(nbrs, ws):
                adj_dict[i][int(j)] = float(w)
        return adj_dict
    
    def _validate_index(self, i: int) -> None:
        """Validate the parameter index"""
        if not 0 <= i < self.n_params:
            raise IndexError(f"Parameter index {i} out of range [0, {self.n_params})")
        
    def __len__(self) -> int:
        """Return the number of parameters."""
        return self.n_params
    
    def __repr__(self):
        mean_deg = self.degrees().mean() if self.n_params > 0 else 0.0
        return (
            f"GFLStructure(n_params={self.n_params}, "
            f"n_pairs={self.n_pairs}, "
            f"mean_degree={mean_deg:.2f})"
        )
