"""
Type aliases for the GFL package.
"""

from typing import Literal

__all__ = [
    "OLSMethod",
    "DuplicateStrategy",
    "SelectionCriterion",
    
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

