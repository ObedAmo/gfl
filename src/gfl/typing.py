"""
Type aliases for the GFL package.
"""

from typing import Literal, TypeAlias, Sequence, Union, Any
import numpy as np
import numpy.typing as npt

__all__ = [
    "IntArray",
    "FloatArray",
    "IntArrayLike",
    "FloatArrayLike",
    "EdgeArrayLike",
    "ArrayLike",
    "OLSMethod",
    "DuplicateStrategy",
    "SelectionCriterion",
    "SelectionRule",

]


IntArray: TypeAlias = npt.NDArray[np.int64]
FloatArray: TypeAlias = npt.NDArray[np.float64]

IntArrayLike: TypeAlias = Union[
    IntArray,
    Sequence[int],
    Sequence[Sequence[int]]
]

FloatArrayLike: TypeAlias = Union[
    FloatArray,
    Sequence[float]
]

EdgeArrayLike: TypeAlias = Union[
    IntArray,
    Sequence[tuple[int, int]],
    Sequence[list[int]]
]

ArrayLike = Union[
    npt.NDArray[Any],
    Sequence[Any],
    Sequence[Sequence[Any]]
]

# OLS estimation methods
OLSMethod = Literal[
    "mean",
    "median",
    "trimmed_mean",
    "huber",
    "lts",
]
"""
Location estimation methods for group-wise OLS.

- "mean": Standard group mean (OLS)
- "median": Group median
- "trimmed_mean": Trimmed mean
- "huber": Huber M-estimator
- "lts": Least Trimmed Squares
"""

# Duplicate pair handling strategies
DuplicateStrategy = Literal[
    "first",
    "sum",
    "mean",
    "max",
    "min"
]
"""
Strategies for handling duplicate fusion pairs.

- "first": Keep first occurrence
- "sum": Sum weights
- "mean": Average weights
- "max": Maximum weight
- "min": Minimum weight
"""

# Model selection criteria
SelectionCriterion = Literal[
    "gcv",
    "aic",
    "bic",
    "ebic",
]
"""
Information criteria for model selection.

- "gcv": Generalized Cross-Validation
- "aic": Akaike Information Criterion
- "bic": Bayesian Information Criterion
- "ebic": Extended Bayesian Information Criterion
"""

SelectionRule = Literal["min", "1se", "elbow"]
"""
Rules for selecting optimal lambda from information criterion curve.

- "min": Global minimum
- "1se": Sparsest model within threshold
- "elbow": Elbow point detection
"""
