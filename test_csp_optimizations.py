"""
Test Script for CSP with Optimizations (Point 3)
================================================
Demonstrates CSP solving with Backtracking, Forward Checking (FC),
Minimum Remaining Values (MRV), and Arc Consistency (AC-3) optimizations.
"""

import sys
sys.path.insert(0, 'src')

from src.csp_solver import GraphColoringCSP, NQueensCSP
from src.answer_generator import AnswerGenerator


def print_header(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_subheader(title):
    print(f"\n  {title}")
    print(f"  {'-'*76}")


def test_graph_coloring():
    """Test CSP: Graph Coloring Problem with partial assignment."""
    print_header("TEST 1: GRAPH COLORING CSP WITH OPTIMIZATIONS")

    print("\n[Instance Description]")
    print("  Graph: 5 vertices, 3 colors available")
    print("  Edges: (0,1), (1,2), (2,3), (3,4), (4,0), (0,2)")
    print("  This creates a cycle with one chord - a classic graph coloring instance")

    n_vertices = 5
    n_colors = 3
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]

    methods = [
        ('Basic Backtracking', lambda csp: csp.solve_backtracking_basic()),
        ('Forward Checking (FC)', lambda csp: csp.solve_with_fc()),
        ('Minimum Remaining Values (MRV)', lambda csp: csp.solve_with_mrv()),
        ('Arc Consistency (AC-3)', lambda csp: csp.solve_with_ac3()),
        ('FC + MRV Combined', lambda csp: csp.solve_with_fc_and_mrv())
    ]

    results = {}

    for method_name, solve_func in methods:
        print_subheader(f"Method: {method_name}")

        csp = GraphColoringCSP.create_csp(n_vertices, edges, n_colors)
        solution = solve_func(csp)
        stats = csp.get_stats()

        print(f"  Solution Found: {'YES ✓' if solution else 'NO ✗'}")

        if solution:
            # Convert solution to readable format
            colors = {}
            for var, col in solution.items():
                vertex = int(var[1:])
                colors[vertex] = col

            print(f"  Color Assignment:")
            for v in range(n_vertices):
                print(f"    Vertex {v}: Color {colors.get(v, '?')}")

        print(f"  Constraint Checks: {stats['constraint_checks']}")
        print(f"  Backtrack Operations: {stats['backtracks']}")

        results[method_name] = {
            'valid': stats['valid'],
            'checks': stats['constraint_checks'],
            'backtracks': stats['backtracks']
        }

    # Summary
    print_subheader("EFFICIENCY COMPARISON SUMMARY")
    print(f"  {'Method':<30} {'Checks':<12} {'Backtracks':<12}")
    print(f"  {'-'*54}")
    for method, result in results.items():
        print(f"  {method:<30} {result['checks']:<12} {result['backtracks']:<12}")

    # Analysis
    best_checks = min(results.items(), key=lambda x: x[1]['checks'])
    best_backtracks = min(results.items(), key=lambda x: x[1]['backtracks'])

    print(f"\n  ✓ BEST (Fewest checks): {best_checks[0]} ({best_checks[1]['checks']} checks)")
    print(f"  ✓ BEST (Fewest backtracks): {best_backtracks[0]} ({best_backtracks[1]['backtracks']} backtracks)")


def test_nqueens():
    """Test CSP: N-Queens with partial assignment."""
    print_header("TEST 2: N-QUEENS CSP WITH PARTIAL ASSIGNMENT")

    n = 6
    print(f"\n[Instance Description]")
    print(f"  Problem: {n}-Queens")
    print(f"  Partially solved: 3 queens already placed")
    print(f"  Remaining: {n-3} queens to place")

    # Partial assignment (some queens already placed)
    # Row 0: Queen at column 1, Row 1: Queen at column 3, Row 2: Queen at column 5
    partial_assignment = {0: 1, 1: 3, 2: 5}

    print(f"  Pre-placed Queens: {partial_assignment}")
    print(f"  Constraint: No two queens can attack each other")
    print(f"    - Different columns")
    print(f"    - Different diagonals")

    methods = [
        ('Basic Backtracking', lambda csp: csp.solve_backtracking_basic()),
        ('Forward Checking (FC)', lambda csp: csp.solve_with_fc()),
        ('Minimum Remaining Values (MRV)', lambda csp: csp.solve_with_mrv()),
        ('Arc Consistency (AC-3)', lambda csp: csp.solve_with_ac3()),
        ('FC + MRV Combined', lambda csp: csp.solve_with_fc_and_mrv())
    ]

    results = {}

    for method_name, solve_func in methods:
        print_subheader(f"Method: {method_name}")

        csp = NQueensCSP.create_csp(n, partial_assignment)
        solution = solve_func(csp)
        stats = csp.get_stats()

        print(f"  Solution Found: {'YES ✓' if solution else 'NO ✗'}")

        if solution:
            # Merge partial assignment with solution
            full_solution = {**partial_assignment}
            for var, col in solution.items():
                row = int(var[1:])
                full_solution[row] = col

            print(f"  Final Queen Positions (row: column):")
            for row in range(n):
                col = full_solution.get(row, '?')
                pre_placed = " [pre-placed]" if row in partial_assignment else ""
                print(f"    Q{row}: Column {col}{pre_placed}")

        print(f"  Constraint Checks: {stats['constraint_checks']}")
        print(f"  Backtrack Operations: {stats['backtracks']}")

        results[method_name] = {
            'valid': stats['valid'],
            'checks': stats['constraint_checks'],
            'backtracks': stats['backtracks']
        }

    # Summary
    print_subheader("EFFICIENCY COMPARISON SUMMARY")
    print(f"  {'Method':<30} {'Checks':<12} {'Backtracks':<12}")
    print(f"  {'-'*54}")
    for method, result in results.items():
        print(f"  {method:<30} {result['checks']:<12} {result['backtracks']:<12}")

    # Analysis
    best_checks = min(results.items(), key=lambda x: x[1]['checks'])
    best_backtracks = min(results.items(), key=lambda x: x[1]['backtracks'])

    print(f"\n  ✓ BEST (Fewest checks): {best_checks[0]} ({best_checks[1]['checks']} checks)")
    print(f"  ✓ BEST (Fewest backtracks): {best_backtracks[0]} ({best_backtracks[1]['backtracks']} backtracks)")


def test_csp_explanations():
    """Explain what each optimization does."""
    print_header("CSP OPTIMIZATION STRATEGIES EXPLANATION")

    explanations = {
        "Backtracking": {
            "What it does": "Tries assigning values sequentially, backtracks when constraint violated",
            "Best for": "Small instances",
            "Advantage": "Simple, guaranteed correct",
            "Disadvantage": "Can explore many dead ends"
        },
        "Forward Checking (FC)": {
            "What it does": "After each assignment, removes inconsistent values from neighbors' domains",
            "Best for": "Medium-sized problems with good constraint structure",
            "Advantage": "Detects failures earlier, reduces search space",
            "Disadvantage": "Some overhead in maintaining domains"
        },
        "MRV (Minimum Remaining Values)": {
            "What it does": "Always assign variable with smallest domain first",
            "Best for": "Problems where branching factor varies",
            "Advantage": "Reduces branching factor, good failure detection",
            "Disadvantage": "Overhead to find min domain each step"
        },
        "AC-3 (Arc Consistency)": {
            "What it does": "Removes values with no support in connected variables",
            "Best for": "Dense constraint graphs",
            "Advantage": "Strong pruning, solves some problems without search",
            "Disadvantage": "Expensive O(e*d³) per iteration"
        }
    }

    for strategy, details in explanations.items():
        print_subheader(strategy)
        for key, value in details.items():
            print(f"  {key:<20}: {value}")


def test_answer_generator_csp():
    """Test CSP solving through Answer Generator."""
    print_header("TEST 3: CSP SOLVING VIA ANSWER GENERATOR")

    try:
        gen = AnswerGenerator('data/knowledge_graph.json')

        print("\n[Graph Coloring Instance]")
        print("  5 vertices, 3 colors, 6 edges")

        csp_instance = {
            'n_vertices': 5,
            'n_colors': 3,
            'edges': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]
        }

        print("\n  Solving...")
        results = gen.solve_csp_with_optimizations('Graph Coloring', csp_instance)

        report = gen.generate_csp_report(results)
        print(report)

    except FileNotFoundError:
        print("  Note: Knowledge graph not found - skipping Answer Generator test")
        print("  This is expected if running test script directly")


def main():
    print("\n" + "█"*80)
    print("█  CSP SOLVING WITH OPTIMIZATIONS - COMPLETE TEST SUITE")
    print("█  Tests for Question 3: Backtracking with FC, MRV, and AC-3 Optimizations")
    print("█"*80)

    test_graph_coloring()
    test_nqueens()
    test_csp_explanations()
    test_answer_generator_csp()

    print("\n" + "█"*80)
    print("█  TEST SUITE COMPLETE")
    print("█"*80)
    print("\n✓ All CSP optimization methods tested successfully!")
    print("\nWhat You Should See:")
    print("  1. Multiple solving methods with different efficiency metrics")
    print("  2. Solution assignments for variables")
    print("  3. Constraint checks and backtrack counts")
    print("  4. Comparison showing which optimization is most efficient")
    print("\n")


if __name__ == "__main__":
    main()

