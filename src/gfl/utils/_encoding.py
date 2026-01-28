"""
Label encoding utilities for group-structured data.
"""

import numpy as np
from typing import Dict, Optional, Union

from .validation import _check_fitted, _ensure_pairs_format, _check_ndim
from gfl.typing import IntArray, EdgeArrayLike, ArrayLike


def _check_no_nan(groups: np.ndarray, name: str = "groups") -> None:
    """
    Check that array contains no NaN or None values.
    """
    # Check for NaN in floating point arrays
    if np.issubdtype(groups.dtype, np.floating):
        if np.any(np.isnan(groups)):
            raise ValueError(f"{name} cannot contain NaN values")
    
    # Check for None or NaN in object arrays
    if groups.dtype == object:
        if any(x is None or (isinstance(x, float) and np.isnan(x)) for x in groups):
            raise ValueError(f"{name} cannot contain None or NaN values")


class GroupEncoder:
    """
    Encode non-contiguous group labels to contiguous [0, n_groups-1] format.
    
    This encoder creates a mapping from arbitrary group identifiers (numeric or string)
    to contiguous integers required by the solver, and provides inverse transformation
    for interpreting clustering results.
    
    Parameters
    ----------
    sort : bool, default=True
        If True, assign labels in sorted order of original values.
        If False, assign labels in order of first appearance.
    
    Attributes
    ----------
    classes_ : ndarray
        Unique group labels in original space (sorted if sort=True)
    n_groups_ : int
        Number of unique groups
    encoding_ : dict
        Mapping from original labels to encoded labels {original: encoded}
    decoding_ : dict
        Mapping from encoded labels to original labels {encoded: original}
    
    Examples
    --------
    >>> # Numeric groups
    >>> encoder = GroupEncoder()
    >>> groups = np.array([10, 5, 10, 15, 5])
    >>> groups_encoded = encoder.fit_transform(groups)
    >>> groups_encoded
    array([1, 0, 1, 2, 0])
    >>> encoder.inverse_transform(groups_encoded)
    array([10, 5, 10, 15, 5])
    
    >>> # String groups
    >>> encoder = GroupEncoder()
    >>> groups = np.array(['Norfolk', 'Hampton', 'Norfolk', 'Virginia Beach'])
    >>> groups_encoded = encoder.fit_transform(groups)
    >>> groups_encoded
    array([1, 0, 1, 2])
    >>> encoder.inverse_transform(groups_encoded)
    array(['Norfolk', 'Hampton', 'Norfolk', 'Virginia Beach'], dtype='<U14')
    
    >>> # Encode fusion pairs with same mapping
    >>> pairs = np.array([['Hampton', 'Norfolk'], ['Norfolk', 'Virginia Beach']])
    >>> pairs_encoded = encoder.transform_pairs(pairs)
    >>> pairs_encoded
    array([[0, 1],
           [1, 2]])
    """
    
    def __init__(self, sort: bool = True):
        self.sort = sort
        self.classes_: Optional[np.ndarray] = None
        self.n_groups_: Optional[int] = None
        self.encoding_: Optional[Dict] = None
        self.decoding_: Optional[Dict] = None
    
    def fit(self, groups: ArrayLike) -> "GroupEncoder":
        """
        Learn the encoding from group labels.
        
        Parameters
        ----------
        groups : array-like of shape (n_samples,)
            Group identifiers (numeric or string)
            
        Returns
        -------
        self : GroupEncoder
            Fitted encoder
        """
        groups = np.asarray(groups)
        
        _check_ndim(groups, expected_dim=1, name="groups")
        
        _check_no_nan(groups, name="groups")
        
        # Get unique classes (works for any comparable type)
        unique_groups = np.unique(groups)
        
        if not self.sort:
            # Use order of first appearance
            _, idx = np.unique(groups, return_index=True)
            unique_groups = groups[np.sort(idx)]
        
        self.classes_ = unique_groups
        self.n_groups_ = len(unique_groups)
        
        # Create bidirectional mappings
        self.encoding_ = {old: new for new, old in enumerate(unique_groups)}
        self.decoding_ = {new: old for new, old in enumerate(unique_groups)}
        
        return self
    
    def transform(self, groups: Union[np.ndarray, list]) -> IntArray:
        """
        Transform group labels to contiguous encoding.
        
        Parameters
        ----------
        groups : array-like of shape (n_samples,)
            Group identifiers to encode (must match type from fit)
            
        Returns
        -------
        groups_encoded : ndarray of shape (n_samples,)
            Encoded group labels in [0, n_groups-1]
            
        Raises
        ------
        ValueError
            If encoder is not fitted or unknown labels encountered
        """
        _check_fitted(self, ['encoding_'], 'encoder')
        
        groups = np.asarray(groups)
        
        _check_ndim(groups, expected_dim=1, name="groups")
        
        _check_no_nan(groups, name="groups")
        
        # Check for unknown labels
        unique_input = np.unique(groups)
        unknown = np.setdiff1d(unique_input, self.classes_)
        
        if len(unknown) > 0:
            raise ValueError(
                f"Unknown group labels encountered: {unknown.tolist()}. "
                f"Known labels: {self.classes_.tolist()}"
            )
        
        # Vectorized transform
        groups_encoded = np.array([self.encoding_[g] for g in groups], dtype=np.int_)
        
        return groups_encoded
    
    def fit_transform(self, groups: Union[np.ndarray, list]) -> IntArray:
        """
        Fit encoder and transform in one step.
        
        Parameters
        ----------
        groups : array-like of shape (n_samples,)
            Group identifiers (numeric or string)
            
        Returns
        -------
        groups_encoded : ndarray of shape (n_samples,)
            Encoded group labels in [0, n_groups-1]
        """
        return self.fit(groups).transform(groups)
    
    def inverse_transform(
        self, 
        groups_encoded: Union[np.ndarray, list]
    ) -> np.ndarray:
        """
        Transform encoded labels back to original labels.
        
        Parameters
        ----------
        groups_encoded : array-like of shape (n_samples,)
            Encoded group labels in [0, n_groups-1]
            
        Returns
        -------
        groups_original : ndarray of shape (n_samples,)
            Original group labels (same type as fit data)
            
        Raises
        ------
        ValueError
            If encoder is not fitted or labels are out of range
        """
        _check_fitted(self, ['decoding_'], 'encoder')
        
        groups_encoded = np.asarray(groups_encoded, dtype=np.int64)
        
        _check_ndim(groups_encoded, expected_dim=1, name="groups_encoded")
        
        # Validate range
        if np.any(groups_encoded < 0) or np.any(groups_encoded >= self.n_groups_):
            raise ValueError(
                f"Encoded labels must be in [0, {self.n_groups_-1}], "
                f"got range [{groups_encoded.min()}, {groups_encoded.max()}]"
            )
        
        groups_original = np.array(
            [self.decoding_[g] for g in groups_encoded],
            dtype=self.classes_.dtype
        )
        
        return groups_original
    
    def transform_pairs(
        self, 
        fusion_pairs: ArrayLike
    ) -> IntArray:
        """
        Transform fusion pairs using the same encoding.
        
        Parameters
        ----------
        fusion_pairs : array-like of shape (n_pairs, 2) or (2, n_pairs)
            Fusion pairs with original group labels (must match type from fit)
            
        Returns
        -------
        pairs_encoded : ndarray of shape (n_pairs, 2)
            Fusion pairs with encoded labels
            
        Examples
        --------
        >>> encoder = GroupEncoder()
        >>> groups = np.array([5, 10, 15])
        >>> encoder.fit(groups)
        >>> pairs = np.array([[5, 10], [10, 15]])
        >>> encoder.transform_pairs(pairs)
        array([[0, 1],
               [1, 2]])
        """
        _check_fitted(self, ['encoding_'], 'encoder')
        
        fusion_pairs = np.asarray(fusion_pairs)
        
        _check_ndim(fusion_pairs, expected_dim=2, name="fusion_pairs")
        
        # Ensure correct format
        fusion_pairs, _ = _ensure_pairs_format(fusion_pairs, "fusion_pairs")
        
        # Check for unknown labels
        unique_pairs = np.unique(fusion_pairs)
        unknown = np.setdiff1d(unique_pairs, self.classes_)
        
        if len(unknown) > 0:
            raise ValueError(
                f"Unknown labels in fusion_pairs: {unknown.tolist()}. "
                f"Known labels: {self.classes_.tolist()}"
            )
        
        # Transform
        pairs_encoded = np.array([
            [self.encoding_[i], self.encoding_[j]]
            for i, j in fusion_pairs
        ], dtype=np.int64)
        
        return pairs_encoded
    
    def inverse_transform_pairs(
        self, 
        pairs_encoded: EdgeArrayLike
    ) -> np.ndarray:
        """
        Transform encoded fusion pairs back to original labels.
        
        Parameters
        ----------
        pairs_encoded : array-like of shape (n_pairs, 2)
            Fusion pairs with encoded labels
            
        Returns
        -------
        pairs_original : ndarray of shape (n_pairs, 2)
            Fusion pairs with original labels (same type as fit data)
            
        Raises
        ------
        ValueError
            If encoder is not fitted
        """
        _check_fitted(self, ['decoding_'], 'encoder')
        
        pairs_encoded = np.asarray(pairs_encoded, dtype=np.int64)

        _check_ndim(pairs_encoded, expected_dim=2, name="pairs_encoded")
        
        # Ensure correct format
        pairs_encoded, _ = _ensure_pairs_format(pairs_encoded, "pairs_encoded")
        
        pairs_original = np.array([
            [self.decoding_[i], self.decoding_[j]]
            for i, j in pairs_encoded
        ], dtype=self.classes_.dtype
        )
        
        return pairs_original
    
    def is_fitted(self) -> bool:
        """Check if encoder has been fitted."""
        return self.encoding_ is not None
    
    def __repr__(self) -> str:
        if self.is_fitted():
            # Show first few classes if many
            if self.n_groups_ <= 5:
                classes_str = str(self.classes_.tolist())
            else:
                classes_str = f"[{self.classes_[0]}, {self.classes_[1]}, ..., {self.classes_[-1]}]"
            
            return (
                f"GroupEncoder(n_groups={self.n_groups_}, "
                f"classes={classes_str})"
            )
        return "GroupEncoder(not fitted)"
