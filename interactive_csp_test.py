"""
Interactive CSP Solver Test
============================
Allows testing CSP with custom instances and comparing optimization strategies.
"""

import sys
sys.path.insert(0, 'src')

from src.csp_solver import GraphColoringCSP, NQueensCSP


def menu():
    """Display main menu."""
    print("\n" + "="*80)
    print("  INTERACTIVE CSP SOLVER - TEST OPTIMIZATION STRATEGIES")
    print("="*80)
    print("\n  1. Test Graph Coloring with default instance")
    print("  2. Test Graph Coloring with custom instance")
    print("  3. Test N-Queens with default instance")
    print("  4. Test N-Queens with custom instance")
    print("  5. Show all optimization methods explanation")
    print("  0. Exit")
    print("\n" + "-"*80)
    choice = input("  Select option (0-5): ").strip()
    return choice


def solve_and_display(csp, method_name, solve_func):
    """Solve and display results for a method."""
    print(f"\n    [{method_name}]")
    solution = solve_func(csp)
    stats = csp.get_stats()

    if solution:
        print(f"      ✓ Solution found")
        print(f"      Assignment: {solution}")
    else:
        print(f"      ✗ No solution found")

    print(f"      Constraint checks: {stats['constraint_checks']}")
    print(f"      Backtrack operations: {stats['backtracks']}")

    return {
        'valid': stats['valid'],
        'checks': stats['constraint_checks'],
        'backtracks': stats['backtracks']
    }


def test_graph_coloring_default():
    """Test graph coloring with default instance."""
    print("\n[Graph Coloring - Default Instance]")
    print("  Instance: 5 vertices, 3 colors")
    print("  Graph: 5-cycle with one chord")

    n_vertices = 5
    n_colors = 3
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]

    methods = [
        ('Backtracking', lambda csp: csp.solve_backtracking_basic()),
        ('Forward Checking', lambda csp: csp.solve_with_fc()),
        ('MRV', lambda csp: csp.solve_with_mrv()),
        ('AC-3', lambda csp: csp.solve_with_ac3()),
        ('FC+MRV', lambda csp: csp.solve_with_fc_and_mrv())
    ]

    results = {}
    for method_name, solve_func in methods:
        csp = GraphColoringCSP.create_csp(n_vertices, edges, n_colors)
        results[method_name] = solve_and_display(csp, method_name, solve_func)

    # Summary
    print("\n  [Summary]")
    print(f"  {'Method':<20} {'Checks':<10} {'Backtracks':<10}")
    print(f"  {'-'*40}")
    for method, result in results.items():
        print(f"  {method:<20} {result['checks']:<10} {result['backtracks']:<10}")

    best = min(results.items(), key=lambda x: x[1]['checks'])
    print(f"\n  ✓ Most efficient: {best[0]} ({best[1]['checks']} checks)")


def test_graph_coloring_custom():
    """Test graph coloring with custom instance."""
    print("\n[Graph Coloring - Custom Instance]")

    try:
        n_vertices = int(input("  Number of vertices (4-8): "))
        if n_vertices < 4 or n_vertices > 8:
            print("  ✗ Invalid number of vertices")
            return

        n_colors = int(input("  Number of colors (3-4): "))
        if n_colors < 3 or n_colors > 4:
            print("  ✗ Invalid number of colors")
            return

        print(f"  Enter edges (format: 'u,v' per line, empty line to stop)")
        edges = []
        while True:
            edge_input = input(f"  Edge {len(edges)+1}: ").strip()
            if not edge_input:
                break
            try:
                u, v = map(int, edge_input.split(','))
                if 0 <= u < n_vertices and 0 <= v < n_vertices and u != v:
                    if (u, v) not in edges and (v, u) not in edges:
                        edges.append((u, v))
                        print(f"    ✓ Added edge ({u},{v})")
                    else:
                        print(f"    ✗ Edge already exists")
                else:
                    print(f"    ✗ Invalid vertices")
            except:
                print(f"    ✗ Invalid format")

        if not edges:
            print("  ✗ Need at least one edge")
            return

        print(f"\n  Instance: {n_vertices} vertices, {n_colors} colors, {len(edges)} edges")

        methods = [
            ('Backtracking', lambda csp: csp.solve_backtracking_basic()),
            ('Forward Checking', lambda csp: csp.solve_with_fc()),
            ('FC+MRV', lambda csp: csp.solve_with_fc_and_mrv())
        ]

        results = {}
        for method_name, solve_func in methods:
            csp = GraphColoringCSP.create_csp(n_vertices, edges, n_colors)
            results[method_name] = solve_and_display(csp, method_name, solve_func)

        print("\n  [Summary]")
        print(f"  {'Method':<20} {'Checks':<10} {'Backtracks':<10}")
        print(f"  {'-'*40}")
        for method, result in results.items():
            print(f"  {method:<20} {result['checks']:<10} {result['backtracks']:<10}")

    except Exception as e:
        print(f"  ✗ Error: {e}")


def test_nqueens_default():
    """Test N-Queens with default instance."""
    print("\n[N-Queens - Default Instance]")
    print("  Instance: 6-Queens with 3 pre-placed queens")
    print("  Pre-placed: {0: 1, 1: 3, 2: 5}")

    n = 6
    partial_assignment = {0: 1, 1: 3, 2: 5}

    methods = [
        ('Backtracking', lambda csp: csp.solve_backtracking_basic()),
        ('Forward Checking', lambda csp: csp.solve_with_fc()),
        ('MRV', lambda csp: csp.solve_with_mrv()),
        ('AC-3', lambda csp: csp.solve_with_ac3()),
        ('FC+MRV', lambda csp: csp.solve_with_fc_and_mrv())
    ]

    results = {}
    for method_name, solve_func in methods:
        csp = NQueensCSP.create_csp(n, partial_assignment)
        results[method_name] = solve_and_display(csp, method_name, solve_func)

    print("\n  [Summary]")
    print(f"  {'Method':<20} {'Checks':<10} {'Backtracks':<10}")
    print(f"  {'-'*40}")
    for method, result in results.items():
        print(f"  {method:<20} {result['checks']:<10} {result['backtracks']:<10}")

    best = min(results.items(), key=lambda x: x[1]['checks'])
    print(f"\n  ✓ Most efficient: {best[0]} ({best[1]['checks']} checks)")


def test_nqueens_custom():
    """Test N-Queens with custom instance."""
    print("\n[N-Queens - Custom Instance]")

    try:
        n = int(input("  Board size (4-8): "))
        if n < 4 or n > 8:
            print("  ✗ Invalid board size")
            return

        n_prime = int(input(f"  Number of pre-placed queens (0-{n-2}): "))
        if n_prime < 0 or n_prime >= n:
            print("  ✗ Invalid number")
            return

        partial_assignment = {}
        print(f"  Enter {n_prime} pre-placed queens (format: 'row,col' per line)")

        for i in range(n_prime):
            while True:
                try:
                    rc_input = input(f"  Queen {i+1}: ").strip()
                    row, col = map(int, rc_input.split(','))
                    if 0 <= row < n and 0 <= col < n and row not in partial_assignment:
                        partial_assignment[row] = col
                        print(f"    ✓ Queen at row {row}, column {col}")
                        break
                    else:
                        print(f"    ✗ Invalid position")
                except:
                    print(f"    ✗ Invalid format")

        print(f"\n  Instance: {n}-Queens with {n_prime} pre-placed")

        methods = [
            ('Backtracking', lambda csp: csp.solve_backtracking_basic()),
            ('Forward Checking', lambda csp: csp.solve_with_fc()),
            ('FC+MRV', lambda csp: csp.solve_with_fc_and_mrv())
        ]

        results = {}
        for method_name, solve_func in methods:
            csp = NQueensCSP.create_csp(n, partial_assignment)
            results[method_name] = solve_and_display(csp, method_name, solve_func)

        print("\n  [Summary]")
        print(f"  {'Method':<20} {'Checks':<10} {'Backtracks':<10}")
        print(f"  {'-'*40}")
        for method, result in results.items():
            print(f"  {method:<20} {result['checks']:<10} {result['backtracks']:<10}")

    except Exception as e:
        print(f"  ✗ Error: {e}")


def show_explanations():
    """Show detailed explanations of optimization methods."""
    explanations = {
        "BASIC BACKTRACKING": {
            "Algoritm": "Încearcă valori secvențial, revine la eșec",
            "Avantaje": "Simplu, garantat corect",
            "Dezavantaje": "Poate explora mulți cai morți",
            "Best for": "Instanțe mici (n < 5)"
        },
        "FORWARD CHECKING (FC)": {
            "Algoritm": "După asignare, elimină valori inconsistente din domenii vecine",
            "Avantaje": "Detectează eșecuri mai devreme",
            "Dezavantaje": "Overhead pentru menținere domenii",
            "Best for": "Instanțe medii cu bună structură"
        },
        "MRV (Minimum Remaining Values)": {
            "Algoritm": "Alege variabila cu cel mai mic domeniu",
            "Avantaje": "Reduce factor de ramificare, bună detectare eșecuri",
            "Dezavantaje": "Overhead pentru calcul min domeniu",
            "Best for": "Probleme cu factor ramificare variabil"
        },
        "AC-3 (Arc Consistency)": {
            "Algoritm": "Elimină valori fără suport în variabile conectate",
            "Avantaje": "Propagare puternică",
            "Dezavantaje": "O(e*d³) scump",
            "Best for": "Grafuri dense cu constrângeri stricte"
        }
    }

    for strategy, details in explanations.items():
        print(f"\n  [{strategy}]")
        for key, value in details.items():
            print(f"    {key:<15}: {value}")


def main():
    """Main interactive loop."""
    print("\n" + "█"*80)
    print("█  INTERACTIVE CSP SOLVER TEST - PUNCTUL 3")
    print("█"*80)

    while True:
        choice = menu()

        if choice == '0':
            print("\n  Goodbye! ✓\n")
            break
        elif choice == '1':
            test_graph_coloring_default()
        elif choice == '2':
            test_graph_coloring_custom()
        elif choice == '3':
            test_nqueens_default()
        elif choice == '4':
            test_nqueens_custom()
        elif choice == '5':
            show_explanations()
        else:
            print("  ✗ Invalid option")


if __name__ == "__main__":
    main()

