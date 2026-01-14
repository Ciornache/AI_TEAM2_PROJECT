from web_interface.app import _validate_graph_coloring, _validate_nqueens, _validate_strategy

print("=" * 70)
print("TEST 1: CORRECT GRAPH COLORING ANSWER")
print("=" * 70)

result = _validate_graph_coloring(
    'V0=Red, V1=Green, V2=Blue, V3=Green, V4=Blue',
    'gc_1',
    {
        'n_vertices': 5,
        'edges': [[0,1], [1,2], [2,3], [3,4], [4,0], [0,2]],
        'n_colors': 3
    }
)

print(f"Correct: {result['is_correct']}")
print(f"Score: {result['score']}/100")
print(f"Feedback: {result['feedback']}")
print(f"Expected: {result['correct_answer']}")
print(f"Explanation: {result['explanation']}")

print("\n" + "=" * 70)
print("TEST 2: INCORRECT GRAPH COLORING (CONSTRAINT VIOLATION)")
print("=" * 70)

result = _validate_graph_coloring(
    'V0=Red, V1=Red, V2=Blue, V3=Green, V4=Blue',
    'gc_1',
    {
        'n_vertices': 5,
        'edges': [[0,1], [1,2], [2,3], [3,4], [4,0], [0,2]],
        'n_colors': 3
    }
)

print(f"Correct: {result['is_correct']}")
print(f"Score: {result['score']}/100")
print(f"Feedback: {result['feedback']}")
print(f"Expected: {result['correct_answer']}")
print(f"Explanation: {result['explanation']}")

print("\n" + "=" * 70)
print("TEST 3: N-QUEENS CORRECT ANSWER")
print("=" * 70)

result = _validate_nqueens(
    'Q0=1, Q1=3, Q2=5, Q3=0, Q4=2, Q5=4',
    'nq_1',
    {
        'n': 6,
        'placed_queens': {0: 1, 1: 3, 2: 5}
    }
)

print(f"Correct: {result['is_correct']}")
print(f"Score: {result['score']}/100")
print(f"Feedback: {result['feedback']}")
print(f"Expected: {result['correct_answer']}")
print(f"Explanation: {result['explanation']}")

print("\n✅ All validation tests completed!")

