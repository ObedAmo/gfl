"""Utility functions for GFL."""

from .validation import (
    check_array,
    check_1d_array,
    check_groups,
    check_fusion_pairs,
    check_groups_and_pairs,
)
from ._uf import UnionFind
from ._encoding import GroupEncoder


__all__ = [
    "check_array",
    "check_1d_array",
    "check_groups",
    "check_fusion_pairs",
    "check_groups_and_pairs",
    "UnionFind",
    "GroupEncoder",

]
