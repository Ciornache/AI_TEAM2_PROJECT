#!/usr/bin/env python
"""Test the validation logic directly without Flask"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'web_interface'))

from app import _validate_graph_coloring, _validate_nqueens, _validate_strategy

# Test 1: Correct Graph Coloring Answer
print("=" * 60)
print("TEST 1: Correct Graph Coloring Answer")
print("=" * 60)
result = _validate_graph_coloring(
    "V0=Red, V1=Green, V2=Blue, V3=Green, V4=Blue",
    "gc_1",
    {
        'n_vertices': 5,
        'edges': [[0,1], [1,2], [2,3], [3,4], [4,0], [0,2]],
        'n_colors': 3
    }
)
print("Result:")
print(f"  ✓ Correct: {result['is_correct']}")
print(f"  ✓ Score: {result['score']}")
print(f"  ✓ Feedback: {result['feedback']}")
print(f"  ✓ Expected: {result['correct_answer']}")
print(f"  ✓ Explanation: {result['explanation']}")

# Test 2: Incorrect Graph Coloring (constraint violation)
print("\n" + "=" * 60)
print("TEST 2: Incorrect Graph Coloring (Constraint Violation)")
print("=" * 60)
result = _validate_graph_coloring(
    "V0=Red, V1=Red, V2=Blue, V3=Green, V4=Blue",
    "gc_1",
    {
        'n_vertices': 5,
        'edges': [[0,1], [1,2], [2,3], [3,4], [4,0], [0,2]],
        'n_colors': 3
    }
)
print("Result:")
print(f"  ✓ Correct: {result['is_correct']}")
print(f"  ✓ Score: {result['score']}")
print(f"  ✓ Feedback: {result['feedback']}")
print(f"  ✓ Expected: {result['correct_answer']}")
print(f"  ✓ Explanation: {result['explanation']}")

# Test 3: N-Queens Correct
print("\n" + "=" * 60)
print("TEST 3: Correct N-Queens Answer")
print("=" * 60)
result = _validate_nqueens(
    "Q0=1, Q1=3, Q2=5, Q3=0, Q4=2, Q5=4",
    "nq_1",
    {
        'n': 6,
        'placed_queens': {0: 1, 1: 3, 2: 5}
    }
)
print("Result:")
print(f"  ✓ Correct: {result['is_correct']}")
print(f"  ✓ Score: {result['score']}")
print(f"  ✓ Feedback: {result['feedback']}")
print(f"  ✓ Expected: {result['correct_answer']}")
print(f"  ✓ Explanation: {result['explanation']}")

# Test 4: Strategy Question
print("\n" + "=" * 60)
print("TEST 4: Strategy Question - Good Answer")
print("=" * 60)
result = _validate_strategy(
    "AC-3 should be used for large problems (n > 10) with dense constraint graphs because the preprocessing overhead pays off through strong constraint propagation",
    "st_1"
)
print("Result:")
print(f"  ✓ Correct: {result['is_correct']}")
print(f"  ✓ Score: {result['score']}")
print(f"  ✓ Feedback: {result['feedback']}")
print(f"  ✓ Expected: {result['correct_answer']}")
print(f"  ✓ Explanation: {result['explanation']}")

print("\n" + "=" * 60)
print("✅ All tests completed!")
print("=" * 60)

