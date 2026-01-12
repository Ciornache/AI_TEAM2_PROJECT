"""
Constraint Satisfaction Problem (CSP) Solver
==============================================
Implements Backtracking with optimizations: Forward Checking (FC),
Minimum Remaining Values (MRV), and Arc Consistency (AC-3).
"""

from typing import Dict, List, Tuple, Set, Optional
from copy import deepcopy
from collections import defaultdict


class CSPSolver:
    """Solves Constraint Satisfaction Problems with various optimizations."""

    def __init__(self, variables: List[str], domains: Dict[str, List],
                 constraints: Dict[Tuple[str, str], callable]):
        """
        Initialize CSP.

        variables: list of variable names
        domains: dict mapping variable name to list of possible values
        constraints: dict mapping (var1, var2) to constraint function
                    constraint(val1, val2) returns True if assignment is valid
        """
        self.variables = variables
        self.domains = {var: list(domain) for var, domain in domains.items()}
        self.constraints = constraints
        self.assignment = {}
        self.constraint_checks = 0
        self.backtracks = 0

    def solve_backtracking_basic(self) -> Optional[Dict]:
        """Basic backtracking without optimizations."""
        self.assignment = {}
        self.constraint_checks = 0
        self.backtracks = 0

        if self._backtrack_basic():
            return self.assignment
        return None

    def solve_with_fc(self) -> Optional[Dict]:
        """Backtracking with Forward Checking (FC)."""
        self.assignment = {}
        self.constraint_checks = 0
        self.backtracks = 0

        # Initialize domains copy
        domains = {var: list(domain) for var, domain in self.domains.items()}

        if self._backtrack_fc(domains):
            return self.assignment
        return None

    def solve_with_mrv(self) -> Optional[Dict]:
        """Backtracking with Minimum Remaining Values (MRV) heuristic."""
        self.assignment = {}
        self.constraint_checks = 0
        self.backtracks = 0

        domains = {var: list(domain) for var, domain in self.domains.items()}

        if self._backtrack_mrv(domains):
            return self.assignment
        return None

    def solve_with_ac3(self) -> Optional[Dict]:
        """Backtracking with AC-3 (Arc Consistency) preprocessing."""
        self.assignment = {}
        self.constraint_checks = 0
        self.backtracks = 0

        domains = {var: list(domain) for var, domain in self.domains.items()}

        # Apply AC-3 initially
        if not self._ac3(domains):
            return None  # Inconsistent from the start

        if self._backtrack_ac3(domains):
            return self.assignment
        return None

    def solve_with_fc_and_mrv(self) -> Optional[Dict]:
        """Backtracking with both FC and MRV."""
        self.assignment = {}
        self.constraint_checks = 0
        self.backtracks = 0

        domains = {var: list(domain) for var, domain in self.domains.items()}

        if self._backtrack_fc_mrv(domains):
            return self.assignment
        return None

    # ==================== BASIC BACKTRACKING ====================

    def _backtrack_basic(self) -> bool:
        """Recursively try to assign values using basic backtracking."""
        if len(self.assignment) == len(self.variables):
            return True  # All variables assigned

        # Select unassigned variable (arbitrary)
        var = next(v for v in self.variables if v not in self.assignment)

        # Try each value in the variable's domain
        for value in self.domains[var]:
            if self._is_consistent_basic(var, value):
                # Make assignment
                self.assignment[var] = value

                # Recursive call
                if self._backtrack_basic():
                    return True

                # Backtrack
                del self.assignment[var]
                self.backtracks += 1

        return False

    def _is_consistent_basic(self, var: str, value) -> bool:
        """Check if assigning var=value is consistent with current assignment."""
        for other_var, other_value in self.assignment.items():
            constraint_key = (var, other_var) if var < other_var else (other_var, var)

            if constraint_key in self.constraints:
                self.constraint_checks += 1
                if not self.constraints[constraint_key](value, other_value):
                    return False

        return True

    # ==================== FORWARD CHECKING ====================

    def _backtrack_fc(self, domains: Dict) -> bool:
        """Backtracking with Forward Checking."""
        if len(self.assignment) == len(self.variables):
            return True

        var = next(v for v in self.variables if v not in self.assignment)

        for value in list(domains[var]):
            if self._is_consistent_basic(var, value):
                # Make assignment
                self.assignment[var] = value

                # Save domains before FC
                saved_domains = deepcopy(domains)

                # Forward checking: remove inconsistent values from neighbors' domains
                if self._forward_check(var, value, domains):
                    if self._backtrack_fc(domains):
                        return True

                # Restore domains on backtrack
                domains.update(saved_domains)
                del self.assignment[var]
                self.backtracks += 1

        return False

    def _forward_check(self, var: str, value, domains: Dict) -> bool:
        """
        Forward check: remove values from neighbor domains that conflict with var=value.
        Returns False if any domain becomes empty.
        """
        for other_var in self.variables:
            if other_var in self.assignment or other_var == var:
                continue

            to_remove = []
            for other_value in domains[other_var]:
                constraint_key = (var, other_var) if var < other_var else (other_var, var)

                if constraint_key in self.constraints:
                    self.constraint_checks += 1
                    if not self.constraints[constraint_key](value, other_value):
                        to_remove.append(other_value)

            # Remove inconsistent values
            for val in to_remove:
                domains[other_var].remove(val)

            # If any domain is empty, FC fails
            if not domains[other_var]:
                return False

        return True

    # ==================== MINIMUM REMAINING VALUES (MRV) ====================

    def _backtrack_mrv(self, domains: Dict) -> bool:
        """Backtracking with Minimum Remaining Values heuristic."""
        if len(self.assignment) == len(self.variables):
            return True

        # Select variable with minimum remaining values (MRV)
        var = self._select_unassigned_variable_mrv(domains)

        if var is None:
            return False

        for value in list(domains[var]):
            if self._is_consistent_basic(var, value):
                self.assignment[var] = value

                saved_domains = deepcopy(domains)

                # Reduce domain to just the chosen value
                domains[var] = [value]

                # Forward check with MRV
                if self._forward_check(var, value, domains):
                    if self._backtrack_mrv(domains):
                        return True

                domains.update(saved_domains)
                del self.assignment[var]
                self.backtracks += 1

        return False

    def _select_unassigned_variable_mrv(self, domains: Dict) -> Optional[str]:
        """Select unassigned variable with minimum remaining values."""
        unassigned = [v for v in self.variables if v not in self.assignment]

        if not unassigned:
            return None

        # Sort by domain size (MRV)
        unassigned.sort(key=lambda v: len(domains[v]))

        return unassigned[0]

    # ==================== ARC CONSISTENCY (AC-3) ====================

    def _backtrack_ac3(self, domains: Dict) -> bool:
        """Backtracking with AC-3 preprocessing and maintenance."""
        if len(self.assignment) == len(self.variables):
            return True

        var = next(v for v in self.variables if v not in self.assignment)

        for value in list(domains[var]):
            if self._is_consistent_basic(var, value):
                self.assignment[var] = value

                saved_domains = deepcopy(domains)

                # Set domain to single value
                domains[var] = [value]

                # Apply AC-3 to reduce domains
                if self._ac3(domains) and self._backtrack_ac3(domains):
                    return True

                domains.update(saved_domains)
                del self.assignment[var]
                self.backtracks += 1

        return False

    def _ac3(self, domains: Dict) -> bool:
        """
        AC-3 algorithm: removes values that have no support.
        Returns False if any domain becomes empty.
        """
        # Initialize queue with all arcs
        queue = []
        for var in self.variables:
            for other_var in self.variables:
                if var != other_var:
                    queue.append((var, other_var))

        while queue:
            xi, xj = queue.pop(0)

            if self._revise(domains, xi, xj):
                # Domain of xi changed
                if not domains[xi]:
                    return False  # Empty domain

                # Add neighbors of xi back to queue
                for xk in self.variables:
                    if xk != xi and xk != xj:
                        queue.append((xk, xi))

        return True

    def _revise(self, domains: Dict, xi: str, xj: str) -> bool:
        """
        Remove values from xi's domain that have no support in xj's domain.
        Returns True if domain was revised.
        """
        revised = False
        to_remove = []

        for vi in domains[xi]:
            # Check if there exists a value in xj's domain that supports vi
            has_support = False

            for vj in domains[xj]:
                constraint_key = (xi, xj) if xi < xj else (xj, xi)

                if constraint_key in self.constraints:
                    self.constraint_checks += 1
                    if self.constraints[constraint_key](vi, vj):
                        has_support = True
                        break
                else:
                    # No constraint = they're compatible
                    has_support = True
                    break

            if not has_support:
                to_remove.append(vi)
                revised = True

        for val in to_remove:
            domains[xi].remove(val)

        return revised

    # ==================== FC + MRV COMBINED ====================

    def _backtrack_fc_mrv(self, domains: Dict) -> bool:
        """Backtracking with both FC and MRV."""
        if len(self.assignment) == len(self.variables):
            return True

        # Select unassigned variable with MRV
        var = self._select_unassigned_variable_mrv(domains)

        if var is None:
            return False

        for value in list(domains[var]):
            if self._is_consistent_basic(var, value):
                self.assignment[var] = value

                saved_domains = deepcopy(domains)

                # Forward checking
                if self._forward_check(var, value, domains):
                    if self._backtrack_fc_mrv(domains):
                        return True

                domains.update(saved_domains)
                del self.assignment[var]
                self.backtracks += 1

        return False

    # ==================== UTILITY METHODS ====================

    def get_stats(self) -> Dict:
        """Return solver statistics."""
        return {
            'assignment': self.assignment,
            'constraint_checks': self.constraint_checks,
            'backtracks': self.backtracks,
            'valid': len(self.assignment) == len(self.variables)
        }


class GraphColoringCSP:
    """Graph Coloring as a CSP problem."""

    @staticmethod
    def create_csp(n_vertices: int, edges: List[Tuple[int, int]],
                   n_colors: int) -> CSPSolver:
        """Create a CSP for graph coloring."""
        variables = [f'V{i}' for i in range(n_vertices)]
        colors = list(range(1, n_colors + 1))
        domains = {var: colors for var in variables}

        # Constraints: adjacent vertices must have different colors
        constraints = {}
        for u, v in edges:
            var_u = f'V{u}'
            var_v = f'V{v}'
            constraint_key = (var_u, var_v) if var_u < var_v else (var_v, var_u)
            constraints[constraint_key] = lambda c1, c2: c1 != c2

        return CSPSolver(variables, domains, constraints)


class NQueensCSP:
    """N-Queens as a CSP problem."""

    @staticmethod
    def create_csp(n: int, partial_assignment: Dict = None) -> CSPSolver:
        """
        Create a CSP for N-Queens.
        partial_assignment: dict of already placed queens {row: col}
        """
        variables = [f'Q{i}' for i in range(n)]
        domains = {var: list(range(n)) for var in variables}

        # Apply partial assignment if provided
        if partial_assignment:
            for row, col in partial_assignment.items():
                variables.remove(f'Q{row}')
                domains = {v: d for v, d in domains.items() if v != f'Q{row}'}

        # Constraints: no two queens can attack each other
        constraints = {}
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                var_i = variables[i]
                var_j = variables[j]

                # Extract original row numbers
                row_i = int(var_i[1:])
                row_j = int(var_j[1:])

                # No attacks constraint
                def no_attack(col_i, col_j, ri=row_i, rj=row_j):
                    return (col_i != col_j and  # Different columns
                           abs(ri - rj) != abs(col_i - col_j))  # Different diagonals

                constraint_key = (var_i, var_j) if var_i < var_j else (var_j, var_i)
                constraints[constraint_key] = no_attack

        return CSPSolver(variables, domains, constraints)


if __name__ == "__main__":
    print("Testing CSP Solver")
    print("=" * 80)

    # Test Graph Coloring
    print("\n[Test 1] Graph Coloring (4 vertices, 3 colors)")
    print("-" * 80)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    csp = GraphColoringCSP.create_csp(4, edges, 3)

    print("\nBasic Backtracking:")
    result = csp.solve_backtracking_basic()
    stats = csp.get_stats()
    print(f"Solution: {result}")
    print(f"Constraint checks: {stats['constraint_checks']}, Backtracks: {stats['backtracks']}")

    print("\nWith Forward Checking (FC):")
    csp2 = GraphColoringCSP.create_csp(4, edges, 3)
    result = csp2.solve_with_fc()
    stats = csp2.get_stats()
    print(f"Solution: {result}")
    print(f"Constraint checks: {stats['constraint_checks']}, Backtracks: {stats['backtracks']}")

    print("\nWith MRV:")
    csp3 = GraphColoringCSP.create_csp(4, edges, 3)
    result = csp3.solve_with_mrv()
    stats = csp3.get_stats()
    print(f"Solution: {result}")
    print(f"Constraint checks: {stats['constraint_checks']}, Backtracks: {stats['backtracks']}")

    print("\nWith AC-3:")
    csp4 = GraphColoringCSP.create_csp(4, edges, 3)
    result = csp4.solve_with_ac3()
    stats = csp4.get_stats()
    print(f"Solution: {result}")
    print(f"Constraint checks: {stats['constraint_checks']}, Backtracks: {stats['backtracks']}")

    print("\nWith FC + MRV:")
    csp5 = GraphColoringCSP.create_csp(4, edges, 3)
    result = csp5.solve_with_fc_and_mrv()
    stats = csp5.get_stats()
    print(f"Solution: {result}")
    print(f"Constraint checks: {stats['constraint_checks']}, Backtracks: {stats['backtracks']}")

