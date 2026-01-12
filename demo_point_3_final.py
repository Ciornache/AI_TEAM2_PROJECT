"""
FINAL DEMONSTRATION - PUNCTUL 3
================================
Complete demonstration of CSP solving with all optimizations
"""

import sys
sys.path.insert(0, 'src')

from src.csp_solver import GraphColoringCSP, NQueensCSP
from src.answer_generator import AnswerGenerator


def print_title(text):
    print("\n" + "█"*100)
    print(f"█  {text}")
    print("█"*100)


def print_section(text):
    print("\n" + "="*100)
    print(f"  {text}")
    print("="*100)


def demo_graph_coloring():
    """Demonstrate Graph Coloring CSP."""
    print_section("DEMONSTRAȚIE 1: GRAPH COLORING - 5 VÂRFURI, 3 CULORI")

    print("\n[1] DESCRIEREA INSTANȚEI")
    print("   ├─ Variabile: V0, V1, V2, V3, V4 (5 vârfuri)")
    print("   ├─ Domenii: {1, 2, 3} (3 culori disponibile)")
    print("   ├─ Constrângeri: Vârfuri adiacente au culori diferite")
    print("   └─ Graf: (0-1), (1-2), (2-3), (3-4), (4-0), (0-2)")

    n_vertices = 5
    n_colors = 3
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]

    print("\n[2] REZOLVARE CU 5 STRATEGII DIFERITE")

    strategies = [
        ('BACKTRACKING BASIC', lambda csp: csp.solve_backtracking_basic()),
        ('FORWARD CHECKING (FC)', lambda csp: csp.solve_with_fc()),
        ('MRV (Min Remaining Values)', lambda csp: csp.solve_with_mrv()),
        ('AC-3 (Arc Consistency)', lambda csp: csp.solve_with_ac3()),
        ('FC + MRV COMBINED', lambda csp: csp.solve_with_fc_and_mrv())
    ]

    results = {}

    for i, (strategy_name, solve_func) in enumerate(strategies, 1):
        print(f"\n   {i}. {strategy_name}")
        print(f"      {'-'*80}")

        csp = GraphColoringCSP.create_csp(n_vertices, edges, n_colors)
        solution = solve_func(csp)
        stats = csp.get_stats()

        if solution:
            print(f"      ✓ Soluție găsită!")
            print(f"      Asignări:")
            for var in sorted(solution.keys()):
                vertex = int(var[1:])
                color = solution[var]
                print(f"         V{vertex} = Culoare {color}")
        else:
            print(f"      ✗ Nicio soluție")

        print(f"      Constrângeri evaluate: {stats['constraint_checks']}")
        print(f"      Reveniri (backtracks): {stats['backtracks']}")

        results[strategy_name] = {
            'checks': stats['constraint_checks'],
            'backtracks': stats['backtracks']
        }

    print("\n[3] COMPARAȚIE EFICIENȚĂ")
    print(f"\n   {'Strategie':<30} {'Checks':<15} {'Backtracks':<15}")
    print(f"   {'-'*60}")
    for strategy, data in results.items():
        print(f"   {strategy:<30} {data['checks']:<15} {data['backtracks']:<15}")

    best_checks = min(results.items(), key=lambda x: x[1]['checks'])
    print(f"\n   ✓ CÂȘTIGĂTOR (Fewest checks): {best_checks[0]} ({best_checks[1]['checks']})")


def demo_nqueens():
    """Demonstrate N-Queens CSP with partial assignment."""
    print_section("DEMONSTRAȚIE 2: N-QUEENS - 6 REGINE, 3 PRE-PLASATE")

    print("\n[1] DESCRIEREA INSTANȚEI")
    print("   ├─ Variabile: Q0, Q1, Q2, Q3, Q4, Q5 (6 regine pe tablă 6x6)")
    print("   ├─ Domenii: Coloanele {0,1,2,3,4,5}")
    print("   ├─ Asignare parțială (pre-plasare):")
    print("   │  ├─ Q0 (rândul 0) → Coloana 1")
    print("   │  ├─ Q1 (rândul 1) → Coloana 3")
    print("   │  └─ Q2 (rândul 2) → Coloana 5")
    print("   ├─ De rezolvat: Q3, Q4, Q5 (3 regine rămase)")
    print("   └─ Constrângere: Fără două regine pe aceeași linie, coloană sau diagonală")

    n = 6
    partial_assignment = {0: 1, 1: 3, 2: 5}

    print("\n[2] REZOLVARE CU 5 STRATEGII")

    strategies = [
        ('BACKTRACKING BASIC', lambda csp: csp.solve_backtracking_basic()),
        ('FORWARD CHECKING (FC)', lambda csp: csp.solve_with_fc()),
        ('MRV (Min Remaining Values)', lambda csp: csp.solve_with_mrv()),
        ('AC-3 (Arc Consistency)', lambda csp: csp.solve_with_ac3()),
        ('FC + MRV COMBINED', lambda csp: csp.solve_with_fc_and_mrv())
    ]

    results = {}
    first_solution = None

    for i, (strategy_name, solve_func) in enumerate(strategies, 1):
        print(f"\n   {i}. {strategy_name}")
        print(f"      {'-'*80}")

        csp = NQueensCSP.create_csp(n, partial_assignment)
        solution = solve_func(csp)
        stats = csp.get_stats()

        if solution:
            if first_solution is None:
                first_solution = solution

            print(f"      ✓ Soluție găsită!")
            print(f"      Asignări finale:")

            # Merge partial with solution
            full_assignment = {**partial_assignment}
            for var, col in solution.items():
                row = int(var[1:])
                full_assignment[row] = col

            for row in range(n):
                col = full_assignment.get(row)
                status = "[pre-plasat]" if row in partial_assignment else "[rezolvat]"
                print(f"         Q{row} (rândul {row}) → Coloana {col} {status}")
        else:
            print(f"      ✗ Nicio soluție")

        print(f"      Constrângeri evaluate: {stats['constraint_checks']}")
        print(f"      Reveniri (backtracks): {stats['backtracks']}")

        results[strategy_name] = {
            'checks': stats['constraint_checks'],
            'backtracks': stats['backtracks']
        }

    print("\n[3] COMPARAȚIE EFICIENȚĂ")
    print(f"\n   {'Strategie':<30} {'Checks':<15} {'Backtracks':<15}")
    print(f"   {'-'*60}")
    for strategy, data in results.items():
        print(f"   {strategy:<30} {data['checks']:<15} {data['backtracks']:<15}")

    best_checks = min(results.items(), key=lambda x: x[1]['checks'])
    print(f"\n   ✓ CÂȘTIGĂTOR (Fewest checks): {best_checks[0]} ({best_checks[1]['checks']})")


def demo_strategy_explanation():
    """Explain each strategy."""
    print_section("EXPLICAȚII: CE FACE FIECARE STRATEGIE")

    strategies = {
        "BACKTRACKING": {
            "Definiție": "Algoritm de căutare în profunzime care revine la puncte de decizie",
            "Algoritm": """
   1. Alege variabilă nerezolvată
   2. Pentru fiecare valoare din domeniu:
      a. Asignează valoarea
      b. Verifică dacă respectă constrângerile cu variabile asignate
      c. Dacă OK: continuă recursiv
      d. Dacă EȘEC: revine, șterge asignarea (backtrack)
            """,
            "Complex. Timp": "O(d^n) worst-case",
            "Complex. Spațiu": "O(n)",
            "Avantaje": "Simplu, garantat corect, low overhead",
            "Dezavantaje": "Poate explora mulți cai morți",
            "Best for": "Instanțe FOARTE mici (n < 5)"
        },
        "FORWARD CHECKING (FC)": {
            "Definiție": "Backtracking cu propagare: elimină valori din domenii vecine",
            "Algoritm": """
   1. Backtracking normal
   2. După asignare X=val:
      - Pentru fiecare variabilă Y nerezolvată:
        - Elimina din domeniu(Y) valorile care încalcă constrângere(X=val, Y=?)
   3. Dacă vreo domeniu se golește: EȘEC, revine imediat
            """,
            "Complex. Timp": "O(e*d²) per nod de căutare",
            "Complex. Spațiu": "O(n*d) pentru domenii",
            "Avantaje": "Detectează eșecuri MULT mai devreme",
            "Dezavantaje": "Overhead pentru menținere domenii",
            "Best for": "Instanțe medii (n ~ 5-8)"
        },
        "MRV (Minimum Remaining Values)": {
            "Definiție": "Heuristic de selecție: alege variabila cu domeniu cel mai mic",
            "Algoritm": """
   1. La fiecare pas, localizează variabila nerezolvată cu |domeniu| minim
   2. Alege acea variabilă (în loc de arbitrară)
   3. Continuă backtracking normal
            """,
            "Complex. Timp": "Backtracking + O(n) pentru găsit min",
            "Complex. Spațiu": "O(n)",
            "Avantaje": "Reduce factor de ramificare SEMNIFICATIV",
            "Dezavantaje": "Overhead pentru calcul domeniu minim",
            "Best for": "Instanțe cu factor ramificare variabil"
        },
        "AC-3 (ARC CONSISTENCY)": {
            "Definiție": "Algoritm de propagare: elimină valori fără suport",
            "Algoritm": """
   1. Inițializare: Coadă = TOATE arcele din CSP
   2. Bucla:
      - Pop arc (X, Y) din coadă
      - Revise(X, Y):
        * Pentru fiecare v din domeniu(X):
          - Dacă ∄ w în domeniu(Y) cu constr(v,w)=TRUE:
            - Elimina v din domeniu(X)
      - Dacă domeniu(X) s-a schimbat:
        - Adauga (Z, X) pentru toți vecinii Z
            """,
            "Complex. Timp": "O(e*d³) inițial, O(e*d²) per update",
            "Complex. Spațiu": "O(e) pentru coadă",
            "Avantaje": "Propagare FOARTE puternică, reduce dramatic",
            "Dezavantaje": "SCUMP pentru instanțe mici",
            "Best for": "Instanțe DENSE cu constrângeri stricte"
        },
        "FC + MRV COMBINED": {
            "Definiție": "Combinația optimă: FC pentru propagare + MRV pentru selecție",
            "Algoritm": """
   1. La fiecare pas:
      a. MRV: Alege variabila cu domeniu minim
      b. FC: După asignare, elimina valori inconsistente din vecini
      c. Dacă vreo domeniu se golește: Backtrack
            """,
            "Complex. Timp": "O(e*d²) + O(n) per nod",
            "Complex. Spațiu": "O(n*d)",
            "Avantaje": "Best of both worlds - propagare + selecție smart",
            "Dezavantaje": "Overhead combinat",
            "Best for": "SAFE DEFAULT pentru orice instanță"
        }
    }

    for strategy_name, details in strategies.items():
        print(f"\n[{strategy_name}]")
        print("-" * 100)
        for key, value in details.items():
            if key == "Algoritm":
                print(f"   {key}:")
                print(value)
            else:
                print(f"   {key}: {value}")


def demo_summary():
    """Final summary and recommendations."""
    print_section("REZUMAT ȘI RECOMANDĂRI")

    print("\n[1] DIMENSIUNI PROBLEMĂ ȘI ALEGEREA STRATEGIEI")
    print("""
   ┌─────────────────┬─────────────────────┬──────────────────────────┐
   │ Dimensiune      │ Recomandare         │ Motiv                    │
   ├─────────────────┼─────────────────────┼──────────────────────────┤
   │ Foarte mică (n<5)│ BACKTRACKING BASIC │ Overhead zero, viteză   │
   │ Mică (n~5-7)    │ FC sau MRV          │ Bun echilibru             │
   │ Medie (n~8-12)  │ FC + MRV            │ Recomandare generală     │
   │ Mare (n>12)     │ AC-3 + MRV          │ Propagare puternică      │
   │ Foarte deasă    │ AC-3 + FC + MRV     │ Prelucrare agresivă       │
   └─────────────────┴─────────────────────┴──────────────────────────┘
    """)

    print("\n[2] TRADE-OFFS CHEIE")
    print("""
   BACKTRACKING BASIC:
   ├─ Avantaje: Simplu, rapid pentru mici, low overhead
   └─ Dezavantaje: Pierde pe instanțe mari

   FORWARD CHECKING:
   ├─ Avantaje: Detectează eșecuri devreme
   └─ Dezavantaje: Overhead pentru medii

   MRV HEURISTIC:
   ├─ Avantaje: Reduce factor ramificare
   └─ Dezavantaje: Calcul min domeniu la fiecare pas

   AC-3:
   ├─ Avantaje: Propagare FOARTE puternică
   └─ Dezavantaje: O(e*d³) SCUMP pentru mici

   FC + MRV:
   ├─ Avantaje: COMBINAȚIE OPTIMĂ
   └─ Dezavantaje: Overhead combinat
    """)

    print("\n[3] METRICE IMPORTANTE")
    print("""
   CONSTRAINT CHECKS:
   ├─ Măsoară: De câte ori evaluez o constrângere
   ├─ Scăzut = Bun (mai puțin calcul)
   └─ Formula: Suma tuturor evaluări de constrângeri

   BACKTRACKS:
   ├─ Măsoară: De câte ori revin din cale greșită
   ├─ Zero = Perfect (cale dreaptă la soluție)
   └─ Formula: Suma tuturor revenituri din eșec

   TRADE-OFF: Checks vs Backtracks
   ├─ AC-3: Mulți checks inițial, dar puțini backtracks
   ├─ Backtracking: Puțini checks, dar mai mulți backtracks
   └─ FC+MRV: Echilibru între cele două
    """)

    print("\n[4] APLICAȚII PRACTICE")
    print("""
   GRAPH COLORING:
   ├─ Hartă: Colorare cu culori minime
   ├─ Zboruri: Planificare cu conflict avoidance
   └─ Register Allocation: Compilatoare
   
   N-QUEENS:
   ├─ Puzzle: Rezolvare matematică
   ├─ Scheduling: Asignare non-conflictual
   └─ Test benchmark pentru algoritmi CSP
   
   ALTE: Sudoku, Map Coloring, Task Scheduling, AI Planning
    """)


def main():
    print_title("DEMONSTRAȚIE COMPLETĂ - PUNCTUL 3: CSP cu BACKTRACKING și OPTIMIZĂRI")

    print("\n📋 DESCRIERE: Rezolvarea Constraint Satisfaction Problems cu 5 strategii diferite")
    print("   ✓ Backtracking Basic")
    print("   ✓ Forward Checking (FC)")
    print("   ✓ Minimum Remaining Values (MRV)")
    print("   ✓ Arc Consistency (AC-3)")
    print("   ✓ FC + MRV Combined")

    print("\n🎯 COMPARAȚIE: Constraint checks și Backtracks pentru fiecare metodă")

    # Run demonstrations
    demo_graph_coloring()
    demo_nqueens()
    demo_strategy_explanation()
    demo_summary()

    print_title("DEMONSTRAȚIE FINALIZATĂ ✓")

    print("\n📊 CONCLUZII:")
    print("""
   1. ✓ Backtracking Basic: Fast pentru instanțe MICI
   2. ✓ FC: Bun compromis pentru MEDII
   3. ✓ MRV: Selectare inteligentă REDUCING branching
   4. ✓ AC-3: Putere pentru DENSE graphs
   5. ✓ FC+MRV: OPTIMAL pentru GENERAL USE
   """)

    print("\n🔍 CE OBSERVI:")
    print("""
   • Instanțe mici: Backtracking e greu de bătut
   • Instanțe medii: FC și MRV devin competitive
   • Instanțe mari: Combinații de optimizări necesare
   • Trade-off: Preprocessing (AC-3) vs Search (FC+MRV)
   """)

    print("\n📁 FIȘIERE CONEXE:")
    print("""
   ✓ src/csp_solver.py - Implementare CSP cu 5 optimizări
   ✓ test_csp_optimizations.py - Test automatizat
   ✓ interactive_csp_test.py - Test interactiv
   ✓ CSP_PUNCTUL_3_GHID.md - Ghid complet
   ✓ docs/POINT_3_CSP_EXPLANATION.md - Documentație
   """)

    print("\n" + "█"*100)
    print("█  GATA! Punctul 3 este COMPLET și DEMONSTRAT! 🎉")
    print("█"*100 + "\n")


if __name__ == "__main__":
    main()

