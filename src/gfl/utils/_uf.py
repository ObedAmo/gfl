"""
Union-Find (Disjoint Set Union) Data Structure
==============================================

This module implements a Union-Find data structure with path compression
and union by rank optimizations. 
It's used to track groups of fused coefficients in the fused lasso algorithm.

The Union-Find data structure supports two main operations:
- find(x): Find the representative (root) of the set containing x
- union(x, y): Merge the sets containing x and y

With path compression and union by rank, both operations have nearly O(1)
amortized time complexity.
"""
from typing import List, Dict
from collections import defaultdict


class UnionFind:
    """
    Union-Find data structure with path compression and union by rank.

    This implementation allows us to track groups of fused parameters
    in the fused lasso algorithm.

    Parameters
    ----------
    n_elements : int
        The number of elements (e.g., parameters) to manage.
    """
    def __init__(self, n_elements: int):
        self.n_elements = n_elements
        self.parent = list(range(n_elements))
        self.rank = [0] * n_elements
        self.n_components = n_elements

    def find(self, x: int) -> int:
        """
        Find the representative (root) of the set 
        containing x with path compression.

        Parameters
        ----------
        x : int
            The element to find the root for.

        Returns
        -------
        int
            The root of the set containing x.
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """
        Merge the sets containing x and y using union by rank.

        Union by rank ensures that the tree with a smaller rank becomes
        a subtree of the tree with a larger rank, keeping the trees balanced.

        Parameters
        ----------
        x : int
            First element.
        y : int
            Second element.

        Returns
        -------
        bool
            True if the union was successful (x and y were in different sets),
            False if x and y were already in the same set.
        """
        root_x = self.find(x)
        root_y = self.find(y)

        # Already in the same set
        if root_x == root_y:
            return False

        # Union by rank: choose arbitrarily and increment the rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        self.n_components -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        """
        Check if elements x and y are in the same set.

        Parameters
        ----------
        x : int
            First element.
        y : int
            Second element.

        Returns
        -------
        bool
            True if x and y are in the same set, False otherwise.
        """
        return self.find(x) == self.find(y)

    def get_components(self) -> Dict[int, List[int]]:
        """
        Get the current components (sets) as a dictionary 
        mapping root to list of elements.

        Returns
        -------
        Dict[int, List[int]]
            A dictionary where keys are root elements 
            and values are lists of elements in each set.
        """
        components = defaultdict(list)
        for element in range(self.n_elements):
            root = self.find(element)
            components[root].append(element)
        return dict(components)

    def get_component_sizes(self) -> Dict[int, int]:
        """
        Get the sizes of each component (set).

        Returns
        -------
        Dict[int, int]
            A dictionary where keys are root elements and values are sizes of each set.
        """
        components = self.get_components()
        return {root: len(members) for root, members in components.items()}

    def get_largest_component(self) -> List[int]:
        """
        Get the largest component (set) of fused elements.

        Returns
        -------
        List[int]
            A list of elements in the largest set.
        """
        components = self.get_components()
        largest_root = max(components, key=lambda root: len(components[root]))
        return components[largest_root]

    def get_multi_element_components(self) -> Dict[int, List[int]]:
        """
        Get components (sets) that contain more than one element.

        Returns
        -------
        Dict[int, List[int]]
            A dictionary where keys are root elements and values are lists of elements in each set
            that contains more than one element.
        """
        components = self.get_components()
        return {root: members for root, members in components.items() if len(members) > 1}
    
    def reset(self) -> None:
        """
        Reset the Union-Find structure to initial state.
        
        Each element becomes its own component.
        """
        self.parent = list(range(self.n_elements))
        self.rank = [0] * self.n_elements
        self.n_components = self.n_elements

    def __len__(self):
        return self.n_elements

    def __repr__(self):
        return f"UnionFind(n_elements={self.n_elements}, n_components={self.n_components})"

    def __str__(self):
        multi_comp = self.get_multi_element_components()
        if multi_comp:
            return (f"UnionFind(n_elements={self.n_elements}, "
                    f"n_components={self.n_components}, "
                    f"multi_element_components={len(multi_comp)})")
        else:
            return (f"UnionFind(n_elements={self.n_elements}, "
                    f"n_components={self.n_components})")
