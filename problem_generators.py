"""
Problem Instance Generators
============================
Generates valid instances for AI search problems using backtracking.
"""

import random
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from copy import deepcopy


@dataclass
class ProblemInstance:
    """Base class for problem instances."""
    problem_type: str
    instance_data: dict
    
    def to_dict(self):
        return {
            'problem_type': self.problem_type,
            'instance_data': self.instance_data
        }


class NQueensGenerator:
    """Generate N-Queens problem instances."""
    
    @staticmethod
    def generate(n: int, n_prime: int) -> ProblemInstance:
        """
        Generate N-Queens instance with n_prime queens already placed.
        n: board size (4-7)
        n_prime: number of pre-placed queens (0 to n-1)
        """
        if n < 4 or n > 7:
            raise ValueError("n must be between 4 and 7")
        if n_prime < 0 or n_prime >= n:
            raise ValueError(f"n_prime must be between 0 and {n-1}")
        
        # Generate a valid solution first using backtracking
        board = NQueensGenerator._generate_solution(n)
        
        # Select n_prime queens to keep
        if n_prime > 0:
            queens_positions = [(i, board[i]) for i in range(n)]
            random.shuffle(queens_positions)
            selected = queens_positions[:n_prime]
        else:
            selected = []
        
        # Create initial board with selected queens
        initial_board = [[0 for _ in range(n)] for _ in range(n)]
        for row, col in selected:
            initial_board[row][col] = 1
        
        return ProblemInstance(
            problem_type="N-Queens",
            instance_data={
                'n': n,
                'n_prime': n_prime,
                'board': initial_board,
                'placed_queens': selected
            }
        )
    
    @staticmethod
    def _generate_solution(n: int) -> List[int]:
        """Generate a valid N-Queens solution. Returns column positions for each row."""
        def is_safe(board, row, col):
            # Check column
            for i in range(row):
                if board[i] == col:
                    return False
            # Check diagonals
            for i in range(row):
                if abs(board[i] - col) == abs(i - row):
                    return False
            return True
        
        def solve(board, row):
            if row == n:
                return True
            
            cols = list(range(n))
            random.shuffle(cols)
            
            for col in cols:
                if is_safe(board, row, col):
                    board[row] = col
                    if solve(board, row + 1):
                        return True
                    board[row] = -1
            return False
        
        board = [-1] * n
        solve(board, 0)
        return board


class HanoiGenerator:
    """Generate Tower of Hanoi problem instances."""
    
    @staticmethod
    def generate(n_disks: int, random_config: bool = False) -> ProblemInstance:
        """
        Generate Tower of Hanoi instance.
        n_disks: number of disks (3-5)
        random_config: if True, generate random valid configuration; else all on peg A
        """
        if n_disks < 3 or n_disks > 5:
            raise ValueError("n_disks must be between 3 and 5")
        
        if random_config:
            # Generate random valid configuration
            configuration = HanoiGenerator._generate_random_config(n_disks)
        else:
            # All disks on peg A (standard starting position)
            configuration = {
                'A': list(range(n_disks, 0, -1)),
                'B': [],
                'C': []
            }
        
        # Goal: all disks on peg C
        goal = {
            'A': [],
            'B': [],
            'C': list(range(n_disks, 0, -1))
        }
        
        return ProblemInstance(
            problem_type="Tower of Hanoi",
            instance_data={
                'n_disks': n_disks,
                'initial': configuration,
                'goal': goal
            }
        )
    
    @staticmethod
    def _generate_random_config(n_disks: int) -> dict:
        """Generate random valid Hanoi configuration."""
        pegs = {'A': [], 'B': [], 'C': []}
        disks = list(range(1, n_disks + 1))
        
        for disk in disks:
            # Choose random peg
            peg = random.choice(['A', 'B', 'C'])
            pegs[peg].append(disk)
        
        # Sort each peg (larger disks at bottom)
        for peg in pegs:
            pegs[peg].sort(reverse=True)
        
        return pegs


class GraphColoringGenerator:
    """Generate Graph Coloring problem instances."""
    
    @staticmethod
    def generate(n_vertices: int, n_colors: int, density: float = 0.5) -> ProblemInstance:
        """
        Generate Graph Coloring instance.
        n_vertices: number of vertices (4-8)
        n_colors: number of colors (3-4)
        density: edge density (0.3-0.7)
        """
        if n_vertices < 4 or n_vertices > 8:
            raise ValueError("n_vertices must be between 4 and 8")
        if n_colors < 3 or n_colors > 4:
            raise ValueError("n_colors must be between 3 and 4")
        
        # Generate random graph with given density
        edges = []
        for i in range(n_vertices):
            for j in range(i + 1, n_vertices):
                if random.random() < density:
                    edges.append((i, j))
        
        # Ensure graph is connected
        if not GraphColoringGenerator._is_connected(n_vertices, edges):
            # Add edges to make it connected
            edges = GraphColoringGenerator._make_connected(n_vertices, edges)
        
        return ProblemInstance(
            problem_type="Graph Coloring",
            instance_data={
                'n_vertices': n_vertices,
                'n_colors': n_colors,
                'edges': edges,
                'adjacency_list': GraphColoringGenerator._to_adjacency_list(n_vertices, edges)
            }
        )
    
    @staticmethod
    def _is_connected(n_vertices: int, edges: List[Tuple[int, int]]) -> bool:
        """Check if graph is connected using BFS."""
        if n_vertices <= 1:
            return True
        
        adj = [[] for _ in range(n_vertices)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        visited = [False] * n_vertices
        queue = [0]
        visited[0] = True
        count = 1
        
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
                    count += 1
        
        return count == n_vertices
    
    @staticmethod
    def _make_connected(n_vertices: int, edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Ensure graph is connected by adding minimum edges."""
        edges = list(edges)
        for i in range(n_vertices - 1):
            edge = (i, i + 1)
            if edge not in edges:
                edges.append(edge)
        return edges
    
    @staticmethod
    def _to_adjacency_list(n_vertices: int, edges: List[Tuple[int, int]]) -> dict:
        """Convert edge list to adjacency list."""
        adj = {i: [] for i in range(n_vertices)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj


class KnightTourGenerator:
    """Generate Knight's Tour problem instances."""
    
    @staticmethod
    def generate(board_size: int, start_pos: Optional[Tuple[int, int]] = None) -> ProblemInstance:
        """
        Generate Knight's Tour instance.
        board_size: size of chessboard (5-6)
        start_pos: starting position (random if None)
        """
        if board_size < 5 or board_size > 6:
            raise ValueError("board_size must be 5 or 6")
        
        if start_pos is None:
            start_pos = (random.randint(0, board_size - 1), random.randint(0, board_size - 1))
        
        # Generate a partial tour (some squares visited)
        tour = KnightTourGenerator._generate_partial_tour(board_size, start_pos)
        
        return ProblemInstance(
            problem_type="Knight's Tour",
            instance_data={
                'board_size': board_size,
                'start_pos': start_pos,
                'visited': tour,
                'n_visited': len(tour)
            }
        )
    
    @staticmethod
    def _generate_partial_tour(board_size: int, start_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Generate a partial knight's tour."""
        moves = [
            (2, 1), (1, 2), (-1, 2), (-2, 1),
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]
        
        def is_valid(x, y, visited):
            return 0 <= x < board_size and 0 <= y < board_size and (x, y) not in visited
        
        visited = [start_pos]
        current = start_pos
        
        # Make random valid moves (3-8 moves)
        max_moves = random.randint(3, min(8, board_size * board_size - 1))
        
        for _ in range(max_moves):
            valid_moves = []
            for dx, dy in moves:
                nx, ny = current[0] + dx, current[1] + dy
                if is_valid(nx, ny, visited):
                    valid_moves.append((nx, ny))
            
            if not valid_moves:
                break
            
            next_pos = random.choice(valid_moves)
            visited.append(next_pos)
            current = next_pos
        
        return visited


class Puzzle8Generator:
    """Generate 8-Puzzle problem instances."""
    
    @staticmethod
    def generate(n_moves_from_goal: int = 10) -> ProblemInstance:
        """
        Generate 8-Puzzle instance by making n random moves from goal state.
        n_moves_from_goal: number of random moves to make (5-15)
        """
        # Goal state
        goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        
        # Start from goal and make random moves
        current = deepcopy(goal)
        blank_pos = (2, 2)
        
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        prev_move = None
        
        for _ in range(n_moves_from_goal):
            valid_moves = []
            for dx, dy in moves:
                nx, ny = blank_pos[0] + dx, blank_pos[1] + dy
                if 0 <= nx < 3 and 0 <= ny < 3:
                    # Avoid undoing previous move
                    if prev_move is None or (nx, ny) != prev_move:
                        valid_moves.append((nx, ny))
            
            if valid_moves:
                next_pos = random.choice(valid_moves)
                # Swap blank with tile
                current[blank_pos[0]][blank_pos[1]], current[next_pos[0]][next_pos[1]] = \
                    current[next_pos[0]][next_pos[1]], current[blank_pos[0]][blank_pos[1]]
                prev_move = blank_pos
                blank_pos = next_pos
        
        return ProblemInstance(
            problem_type="8-Puzzle",
            instance_data={
                'initial': current,
                'goal': goal,
                'blank_pos': blank_pos
            }
        )


def generate_all_instances(n_instances_per_problem: int = 2) -> dict:
    """Generate instances for all problems."""
    instances = {
        'n_queens': [],
        'hanoi': [],
        'graph_coloring': [],
        'knight_tour': [],
        'puzzle_8': []
    }
    
    # N-Queens: vary n from 4 to 6, vary n_prime
    for _ in range(n_instances_per_problem):
        n = random.randint(4, 6)
        n_prime = random.randint(0, n - 2)
        instances['n_queens'].append(NQueensGenerator.generate(n, n_prime))
    
    # Tower of Hanoi: vary disks from 3 to 4
    for _ in range(n_instances_per_problem):
        n_disks = random.randint(3, 4)
        random_config = random.choice([False, True])
        instances['hanoi'].append(HanoiGenerator.generate(n_disks, random_config))
    
    # Graph Coloring: vary vertices and colors
    for _ in range(n_instances_per_problem):
        n_vertices = random.randint(5, 7)
        n_colors = random.randint(3, 4)
        density = random.uniform(0.4, 0.6)
        instances['graph_coloring'].append(GraphColoringGenerator.generate(n_vertices, n_colors, density))
    
    # Knight's Tour: vary board size
    for _ in range(n_instances_per_problem):
        board_size = random.choice([5, 6])
        instances['knight_tour'].append(KnightTourGenerator.generate(board_size))
    
    # 8-Puzzle: vary difficulty
    for _ in range(n_instances_per_problem):
        n_moves = random.randint(8, 15)
        instances['puzzle_8'].append(Puzzle8Generator.generate(n_moves))
    
    return instances


if __name__ == "__main__":
    # Test generators
    print("Testing Problem Generators")
    print("=" * 80)
    
    print("\n1. N-Queens (n=5, n_prime=2):")
    nq = NQueensGenerator.generate(5, 2)
    print(f"   Board size: {nq.instance_data['n']}")
    print(f"   Pre-placed queens: {nq.instance_data['n_prime']}")
    
    print("\n2. Tower of Hanoi (4 disks):")
    hanoi = HanoiGenerator.generate(4, random_config=True)
    print(f"   Disks: {hanoi.instance_data['n_disks']}")
    print(f"   Initial: {hanoi.instance_data['initial']}")
    
    print("\n3. Graph Coloring (6 vertices, 3 colors):")
    gc = GraphColoringGenerator.generate(6, 3)
    print(f"   Vertices: {gc.instance_data['n_vertices']}")
    print(f"   Colors: {gc.instance_data['n_colors']}")
    print(f"   Edges: {len(gc.instance_data['edges'])}")
    
    print("\n4. Knight's Tour (5x5):")
    kt = KnightTourGenerator.generate(5)
    print(f"   Board: {kt.instance_data['board_size']}x{kt.instance_data['board_size']}")
    print(f"   Start: {kt.instance_data['start_pos']}")
    print(f"   Visited: {kt.instance_data['n_visited']} squares")
    
    print("\n5. 8-Puzzle:")
    p8 = Puzzle8Generator.generate(12)
    print(f"   Initial state:")
    for row in p8.instance_data['initial']:
        print(f"   {row}")
