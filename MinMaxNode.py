"""
MinMax with Alpha-Beta Pruning
==============================
Module to evaluate game trees using MinMax strategy with Alpha-Beta optimization.
Returns root value and number of leaf nodes visited.
"""

from typing import List, Tuple, Optional

class MinMaxNode:
    """Node in a MinMax tree."""
    def __init__(self, value: Optional[int] = None, children: Optional[List['MinMaxNode']] = None):
        self.value = value        # Leaf node has integer value, internal nodes have None
        self.children = children or []  # List of MinMaxNode
        self.is_leaf = value is not None

def alphabeta(node: MinMaxNode, depth: int, alpha: float, beta: float, maximizing: bool, leaf_counter: List[int]) -> int:
    """
    Alpha-Beta pruning algorithm.
    
    Args:
        node: MinMaxNode to evaluate
        depth: current depth (optional, mainly for debugging)
        alpha: best value for maximizer so far
        beta: best value for minimizer so far
        maximizing: True if current node is maximizing
        leaf_counter: single-element list to count visited leaves (mutable)
    
    Returns:
        Value of node after MinMax evaluation
    """
    if node.is_leaf:
        leaf_counter[0] += 1
        return node.value
    
    if maximizing:
        value = float('-inf')
        for child in node.children:
            value = max(value, alphabeta(child, depth+1, alpha, beta, False, leaf_counter))
            alpha = max(alpha, value)
            if beta <= alpha:
                break  # Beta cut-off
        return value
    else:
        value = float('inf')
        for child in node.children:
            value = min(value, alphabeta(child, depth+1, alpha, beta, True, leaf_counter))
            beta = min(beta, value)
            if beta <= alpha:
                break  # Alpha cut-off
        return value

def evaluate_tree(root: MinMaxNode, maximizing: bool = True) -> Tuple[int, int]:
    """
    Evaluate the MinMax tree with Alpha-Beta pruning.
    
    Args:
        root: root of the tree
        maximizing: True if root is maximizing
    
    Returns:
        Tuple: (root value, number of leaves visited)
    """
    leaf_counter = [0]
    value = alphabeta(root, depth=0, alpha=float('-inf'), beta=float('inf'), maximizing=maximizing, leaf_counter=leaf_counter)
    return value, leaf_counter[0]

# Example usage
if __name__ == "__main__":
    # Sample tree: root (max)
    #          [ ]
    #        /  |  \
    #      [3] [5]  [ ]
    #             /  \
    #           [2]  [9]
    
    leaf1 = MinMaxNode(3)
    leaf2 = MinMaxNode(5)
    leaf3 = MinMaxNode(2)
    leaf4 = MinMaxNode(9)
    internal = MinMaxNode(children=[leaf3, leaf4])
    root = MinMaxNode(children=[leaf1, leaf2, internal])
    
    value, leaves_visited = evaluate_tree(root, maximizing=True)
    print(f"Root value: {value}")
    print(f"Leaf nodes visited: {leaves_visited}")
