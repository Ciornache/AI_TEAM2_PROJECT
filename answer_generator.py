"""
Answer Generator
================
Generates answers about best solving strategies based on knowledge graph and instance analysis.
"""

import json
from typing import Dict, List, Tuple, Optional
from MinMaxNode import MinMaxNode, evaluate_tree


class AnswerGenerator:
    """Generate answers about best solving strategies from knowledge graph and instance."""
    
    def __init__(self, knowledge_graph_path: str):
        """Load knowledge graph."""
        with open(knowledge_graph_path, 'r', encoding='utf-8') as f:
            self.kg = json.load(f)
        
        # Build indices
        self.nodes_by_id = {node['id']: node for node in self.kg['nodes']}
        self.nodes_by_name = {node['name'].lower(): node for node in self.kg['nodes']}
        
        # Build edge indices
        self.edges_from = {}
        self.edges_to = {}
        for edge in self.kg['edges']:
            if edge['source'] not in self.edges_from:
                self.edges_from[edge['source']] = []
            self.edges_from[edge['source']].append(edge)
            
            if edge['target'] not in self.edges_to:
                self.edges_to[edge['target']] = []
            self.edges_to[edge['target']].append(edge)
    
    def find_problem_node(self, problem_name: str) -> Optional[dict]:
        """Find problem node by name."""
        problem_name_lower = problem_name.lower()
        
        # Direct match
        if problem_name_lower in self.nodes_by_name:
            return self.nodes_by_name[problem_name_lower]
        
        # Partial match
        for name, node in self.nodes_by_name.items():
            if problem_name_lower in name or name in problem_name_lower:
                if node['type'] == 'problem':
                    return node
        
        return None
    
    def get_solving_algorithms(self, problem_name: str, instance_analysis: Dict = None) -> List[Tuple[str, str, float, Dict]]:
        """
        Get algorithms that solve this problem from KG.
        Filter by instance characteristics if provided.
        Returns: [(algorithm_name, relation_type, confidence, edge_scores), ...]
        """
        problem_node = self.find_problem_node(problem_name)
        if not problem_node:
            return []
        
        algorithms = []
        
        # Find edges pointing to this problem (algorithm -> problem)
        if problem_node['id'] in self.edges_to:
            for edge in self.edges_to[problem_node['id']]:
                if edge['relation_type'] in ['solves', 'solved_by', 'applicable_to']:
                    algo_node = self.nodes_by_id.get(edge['source'])
                    if algo_node and algo_node['type'] == 'algorithm':
                        # Check instance conditions if provided
                        if instance_analysis and not self._edge_matches_instance(edge, instance_analysis):
                            continue
                        
                        # Extract scoring components
                        edge_scores = {
                            'proximity': edge.get('proximity_score', 0.0),
                            'frequency': edge.get('frequency_score', 0.0),
                            'sentiment': edge.get('sentiment_score', 0.5),
                            'confidence': edge.get('confidence', 1.0),
                            'sources': edge.get('source_documents', [])
                        }
                        
                        algorithms.append((
                            algo_node['name'],
                            edge['relation_type'],
                            edge.get('confidence', 1.0),
                            edge_scores
                        ))
        
        return algorithms
    
    def _edge_matches_instance(self, edge: Dict, instance_analysis: Dict) -> bool:
        """Check if edge conditions match instance characteristics."""
        conditions = edge.get('instance_conditions', {})
        
        # No conditions = applies to all instances
        if not conditions:
            return True
        
        # Check size condition
        if 'size' in conditions:
            if conditions['size'] != instance_analysis.get('complexity_level'):
                return False
        
        # Check other conditions (can be extended)
        # For now, if conditions exist and size matches or no size specified, accept
        return True
    
    def get_algorithm_properties(self, algorithm_name: str) -> Dict:
        """Get algorithm properties from KG."""
        algo_name_lower = algorithm_name.lower()
        
        if algo_name_lower in self.nodes_by_name:
            node = self.nodes_by_name[algo_name_lower]
            if node['type'] == 'algorithm':
                return node.get('properties', {})
        
        return {}
    
    def get_algorithm_complexity(self, algorithm_name: str) -> Dict[str, str]:
        """Get time and space complexity for algorithm from KG."""
        algo_name_lower = algorithm_name.lower()
        
        if algo_name_lower not in self.nodes_by_name:
            return {}
        
        algo_node = self.nodes_by_name[algo_name_lower]
        complexity = {}
        
        if algo_node['id'] in self.edges_from:
            for edge in self.edges_from[algo_node['id']]:
                if edge['relation_type'] == 'has_time_complexity':
                    target = self.nodes_by_id.get(edge['target'])
                    if target:
                        complexity['time'] = target['name']
                elif edge['relation_type'] == 'has_space_complexity':
                    target = self.nodes_by_id.get(edge['target'])
                    if target:
                        complexity['space'] = target['name']
        
        return complexity
    
    def get_algorithm_category(self, algorithm_name: str) -> Optional[str]:
        """Get algorithm category from KG."""
        algo_name_lower = algorithm_name.lower()
        
        if algo_name_lower not in self.nodes_by_name:
            return None
        
        algo_node = self.nodes_by_name[algo_name_lower]
        
        # Check edges for classification
        if algo_node['id'] in self.edges_from:
            for edge in self.edges_from[algo_node['id']]:
                if edge['relation_type'] == 'classified_as':
                    target = self.nodes_by_id.get(edge['target'])
                    if target:
                        return target['name'].lower()
        
        return None
    
    def get_heuristics_for_problem(self, problem_name: str) -> List[str]:
        """Get heuristics commonly used for this problem."""
        problem_lower = problem_name.lower()
        
        heuristics_map = {
            '8-puzzle': ['Manhattan Distance', 'Misplaced Tiles'],
            'n-queens': ['Number of Conflicts', 'Attacking Pairs'],
            'graph coloring': ['Degree Heuristic', 'Minimum Remaining Values'],
            'knight\'s tour': ['Warnsdorff\'s Rule', 'Accessibility Heuristic'],
            'tower of hanoi': ['Number of Disks to Move']
        }
        
        for key, heuristics in heuristics_map.items():
            if key in problem_lower or problem_lower in key:
                return heuristics
        
        # Check knowledge graph for heuristics
        heuristics = []
        for node in self.kg['nodes']:
            if node['type'] == 'heuristic':
                heuristics.append(node['name'])
        
        return heuristics
    
    def generate_answer(self, problem_name: str, instance_data: dict) -> Dict[str, any]:
        """
        Generate comprehensive answer using KG and instance analysis.
        Handles MinMax problems if tree is provided.
        """
        # 1️⃣ Eval MinMax tree if applicable
        minmax_result = None
        problem_lower = problem_name.lower()
        if ('minmax' in problem_lower or 'alpha-beta' in problem_lower) and 'tree' in instance_data:
            minmax_result = self.evaluate_minmax_tree(instance_data['tree'])

        # 2️⃣ Analyze the specific instance
        instance_analysis = self._analyze_instance(problem_name, instance_data)

        # 3️⃣ Find algorithms from knowledge graph
        kg_algorithms = self._get_algorithms_from_kg()

        # 4️⃣ Get explicit solvers from knowledge graph (instance-aware)
        solving_algos = self.get_solving_algorithms(problem_name, instance_analysis)

        # 5️⃣ Score and rank algorithms based on instance and KG
        recommendations = self._score_and_rank_algorithms(
            problem_name,
            instance_analysis,
            kg_algorithms,
            solving_algos
        )

        # 6️⃣ Generate reasoning based on KG and instance
        reasoning = self._generate_reasoning_from_kg(
            problem_name, 
            instance_analysis, 
            recommendations
        )

        # 7️⃣ Include MinMax results if applicable
        if minmax_result:
            reasoning += "\n**MinMax Evaluation:**\n"
            reasoning += f"- Root value: {minmax_result['root_value']}\n"
            reasoning += f"- Leaf nodes visited: {minmax_result['leaves_visited']}\n"

        # 8️⃣ Return final structured answer
        answer = {
            'problem_name': problem_name,
            'instance_analysis': instance_analysis,
            'explicit_solvers': solving_algos,
            'recommendations': recommendations,
            'heuristics': self.get_heuristics_for_problem(problem_name),
            'reasoning': reasoning,
        }

        if minmax_result:
            answer['minmax_result'] = minmax_result

        return answer

    def _get_algorithms_from_kg(self) -> List[Dict]:
        """Get all algorithms from knowledge graph with their properties."""
        algorithms = []
        
        for node in self.kg['nodes']:
            if node['type'] == 'algorithm':
                algo_info = {
                    'name': node['name'],
                    'id': node['id'],
                    'properties': node.get('properties', {}),
                    'category': self.get_algorithm_category(node['name'])
                }
                
                # Get complexity from edges
                algo_info['complexity'] = self.get_algorithm_complexity(node['name'])
                
                algorithms.append(algo_info)
        
        return algorithms
    
    def _analyze_instance(self, problem_name: str, instance_data: dict) -> Dict:
        """Analyze specific instance characteristics."""
        analysis = {
            'problem_type': problem_name,
            'instance_size': 'unknown',
            'complexity_level': 'medium',
            'completeness': 0.0,
            'remaining_work': 0,
            'characteristics': [],
            'requires_optimality': False,
            'state_space_estimate': 'unknown'
        }
        
        problem_lower = problem_name.lower()
        
        # N-Queens analysis
        if 'n-queens' in problem_lower or 'queens' in problem_lower:
            n = instance_data.get('n', 8)
            n_prime = instance_data.get('n_prime', 0)
            remaining_queens = n - n_prime
            
            analysis['instance_size'] = f'{n}×{n} board'
            analysis['completeness'] = n_prime / n if n > 0 else 0
            analysis['remaining_work'] = remaining_queens
            analysis['state_space_estimate'] = f'~{n}^{remaining_queens} configurations to explore'
            
            if n <= 4 or remaining_queens <= 2:
                analysis['complexity_level'] = 'small'
                analysis['characteristics'].append(f'Small problem: {n}×{n} board with only {remaining_queens} queens to place')
                analysis['characteristics'].append('Simple backtracking sufficient - small search space')
            elif n <= 6 or remaining_queens <= 3:
                analysis['complexity_level'] = 'medium'
                analysis['characteristics'].append(f'Medium problem: {n}×{n} board with {remaining_queens} queens to place')
                analysis['characteristics'].append('Backtracking with constraint checking recommended')
            else:
                analysis['complexity_level'] = 'large'
                analysis['characteristics'].append(f'Large problem: {n}×{n} board with {remaining_queens} queens to place')
                analysis['characteristics'].append('Need aggressive pruning and forward checking')
            
            analysis['characteristics'].append('Constraint satisfaction: no two queens can attack each other')
            analysis['requires_optimality'] = False
            
        # Tower of Hanoi analysis
        elif 'hanoi' in problem_lower:
            n_disks = instance_data.get('n_disks', 3)
            initial = instance_data.get('initial', {})
            disks_on_target = len(initial.get('C', []))
            
            analysis['instance_size'] = f'{n_disks} disks'
            analysis['completeness'] = disks_on_target / n_disks if n_disks > 0 else 0
            analysis['remaining_work'] = n_disks - disks_on_target
            optimal_moves = 2**n_disks - 1
            analysis['state_space_estimate'] = f'Optimal solution: {optimal_moves} moves'
            
            if n_disks <= 3:
                analysis['complexity_level'] = 'small'
                analysis['characteristics'].append(f'Small problem: {n_disks} disks (optimal: {optimal_moves} moves)')
                analysis['characteristics'].append('Simple recursive solution works perfectly')
            elif n_disks == 4:
                analysis['complexity_level'] = 'medium'
                analysis['characteristics'].append(f'Medium problem: {n_disks} disks (optimal: {optimal_moves} moves)')
                analysis['characteristics'].append('Recursive DFS or BFS both reasonable')
            else:
                analysis['complexity_level'] = 'large'
                analysis['characteristics'].append(f'Large problem: {n_disks} disks (optimal: {optimal_moves} moves)')
                analysis['characteristics'].append('Prefer memory-efficient DFS/recursive approach')
            
            analysis['characteristics'].append('Well-defined recursive structure with known optimal solution')
            analysis['requires_optimality'] = True
            
        # Graph Coloring analysis
        elif 'graph coloring' in problem_lower or 'coloring' in problem_lower:
            n_vertices = instance_data.get('n_vertices', 5)
            n_colors = instance_data.get('n_colors', 3)
            edges = instance_data.get('edges', [])
            edge_density = len(edges) / (n_vertices * (n_vertices - 1) / 2) if n_vertices > 1 else 0
            
            analysis['instance_size'] = f'{n_vertices} vertices, {n_colors} colors'
            analysis['completeness'] = 0.0
            analysis['remaining_work'] = n_vertices
            analysis['state_space_estimate'] = f'{n_colors}^{n_vertices} = {n_colors**n_vertices} possible colorings'
            
            if n_vertices <= 5 or edge_density < 0.4:
                analysis['complexity_level'] = 'small'
                analysis['characteristics'].append(f'Small sparse graph: {n_vertices} vertices, {len(edges)} edges (density: {edge_density:.2f})')
                analysis['characteristics'].append('Backtracking works well for sparse graphs')
            elif n_vertices <= 7 or edge_density < 0.6:
                analysis['complexity_level'] = 'medium'
                analysis['characteristics'].append(f'Medium graph: {n_vertices} vertices, {len(edges)} edges (density: {edge_density:.2f})')
                analysis['characteristics'].append('Need pruning with degree heuristics (MRV)')
            else:
                analysis['complexity_level'] = 'large'
                analysis['characteristics'].append(f'Large dense graph: {n_vertices} vertices, {len(edges)} edges (density: {edge_density:.2f})')
                analysis['characteristics'].append('Dense graph requires aggressive pruning')
            
            analysis['characteristics'].append('NP-complete CSP - any valid coloring acceptable')
            analysis['requires_optimality'] = False
            
        # Knight's Tour analysis
        elif 'knight' in problem_lower:
            board_size = instance_data.get('board_size', 8)
            n_visited = instance_data.get('n_visited', 1)
            total_squares = board_size * board_size
            
            analysis['instance_size'] = f'{board_size}×{board_size} board'
            analysis['completeness'] = n_visited / total_squares
            analysis['remaining_work'] = total_squares - n_visited
            analysis['state_space_estimate'] = f'{analysis["remaining_work"]} squares remaining to visit'
            
            if board_size <= 5:
                analysis['complexity_level'] = 'small'
                analysis['characteristics'].append(f'Small board: {board_size}×{board_size} with {analysis["remaining_work"]} squares to visit')
                analysis['characteristics'].append('Backtracking feasible for small boards')
            elif board_size <= 6:
                analysis['complexity_level'] = 'medium'
                analysis['characteristics'].append(f'Medium board: {board_size}×{board_size} with {analysis["remaining_work"]} squares to visit')
                analysis['characteristics'].append('Warnsdorff heuristic strongly recommended')
            else:
                analysis['complexity_level'] = 'large'
                analysis['characteristics'].append(f'Large board: {board_size}×{board_size} with {analysis["remaining_work"]} squares to visit')
                analysis['characteristics'].append('Must use Warnsdorff heuristic - too large without it')
            
            analysis['characteristics'].append('Path-finding with backtracking - find any complete tour')
            analysis['requires_optimality'] = False
            
        # 8-Puzzle analysis
        elif '8-puzzle' in problem_lower or 'puzzle' in problem_lower:
            initial = instance_data.get('initial', [[1,2,3],[4,5,6],[7,8,0]])
            goal = instance_data.get('goal', [[1,2,3],[4,5,6],[7,8,0]])
            
            # Calculate misplaced tiles
            misplaced = sum(1 for i in range(3) for j in range(3) 
                          if initial[i][j] != 0 and initial[i][j] != goal[i][j])
            
            analysis['instance_size'] = '3×3 grid'
            analysis['completeness'] = (9 - misplaced) / 9
            analysis['remaining_work'] = misplaced
            analysis['state_space_estimate'] = f'~{misplaced * 3} moves estimated (Manhattan heuristic)'
            
            if misplaced <= 3:
                analysis['complexity_level'] = 'small'
                analysis['characteristics'].append(f'Close to goal: only {misplaced} tiles misplaced')
                analysis['characteristics'].append('Even uninformed search (BFS) works quickly')
            elif misplaced <= 6:
                analysis['complexity_level'] = 'medium'
                analysis['characteristics'].append(f'Moderate distance: {misplaced} tiles misplaced')
                analysis['characteristics'].append('Informed search (A*) recommended for efficiency')
            else:
                analysis['complexity_level'] = 'large'
                analysis['characteristics'].append(f'Far from goal: {misplaced} tiles misplaced')
                analysis['characteristics'].append('A* with good heuristic essential for reasonable performance')
            
            analysis['characteristics'].append('Sliding tile puzzle - optimal (shortest) solution desired')
            analysis['requires_optimality'] = True
        
        return analysis
    
    def _score_and_rank_algorithms(
        self, 
        problem_name: str, 
        instance_analysis: Dict,
        kg_algorithms: List[Dict],
        explicit_solvers: List[Tuple]
    ) -> List[Dict]:
        """Score and rank algorithms based on KG properties, edge scores, and instance."""
        scored_algorithms = []
        
        complexity_level = instance_analysis['complexity_level']
        requires_optimality = instance_analysis['requires_optimality']
        
        for algo in kg_algorithms:
            score = 0.0
            reasons = []
            kg_scores = {}
            
            algo_name = algo['name']
            properties = algo['properties']
            category = algo.get('category', '')
            complexity = algo.get('complexity', {})
            
            # NEW: Check if this algorithm has an edge to the problem
            edge_info = None
            for solver_tuple in explicit_solvers:
                if solver_tuple[0] == algo_name:
                    edge_info = solver_tuple[3] if len(solver_tuple) > 3 else {}
                    break
            
            # NEW: Use edge scores if available
            if edge_info:
                # Major bonus for KG relationship
                score += 30
                kg_scores = edge_info
                
                # Weight by edge sentiment (positive relationship = higher score)
                sentiment = edge_info.get('sentiment', 0.5)
                sentiment_bonus = (sentiment - 0.5) * 20  # -10 to +10 based on sentiment
                score += sentiment_bonus
                
                # Weight by edge proximity (closer mentions = more relevant)
                proximity = edge_info.get('proximity', 0.0)
                proximity_bonus = proximity * 15  # 0 to 15 based on proximity
                score += proximity_bonus
                
                # Weight by frequency (more mentions = more evidence)
                frequency = edge_info.get('frequency', 0.0)
                frequency_bonus = frequency * 10  # 0 to 10 based on frequency
                score += frequency_bonus
                
                reasons.append(f'✓ KG: Found as solver (conf: {edge_info.get("confidence", 0):.2f}, sent: {sentiment:.2f})')
                
                # Multi-document evidence
                sources = edge_info.get('sources', [])
                if len(sources) > 1:
                    score += 5
                    reasons.append(f'✓ Mentioned in {len(sources)} documents')
            
            # Optimality scoring
            if requires_optimality:
                if properties.get('optimal'):
                    score += 25
                    reasons.append('✓ Guarantees optimal solution (required)')
                else:
                    score -= 15
            
            # Completeness scoring
            if properties.get('complete'):
                score += 15
                reasons.append('✓ Complete - finds solution if one exists')
            
            # Problem-specific scoring
            problem_lower = problem_name.lower()
            
            if 'puzzle' in problem_lower:
                # Puzzles HEAVILY favor informed search with admissible heuristics
                if category == 'informed' and properties.get('admissible'):
                    score += 50  # Major bonus for A*, IDA*
                    reasons.append('✓ Admissible heuristic guarantees optimal solution for puzzles')
                elif category == 'informed':
                    score += 35
                    reasons.append('✓ Informed search ideal for sliding puzzles')
                elif algo_name == 'BFS' and complexity_level == 'small':
                    score += 10
                    reasons.append('✓ BFS acceptable for small puzzle instances')
                elif category == 'uninformed' and not (algo_name == 'BFS'):
                    # Penalize uninformed search (except BFS) for puzzles
                    score -= 20
                    reasons.append('⚠ Uninformed search inefficient for puzzles')
                    
            elif 'queens' in problem_lower or 'coloring' in problem_lower:
                if algo_name in ['DFS', 'Backtracking']:
                    score += 20
                    reasons.append('✓ DFS/Backtracking perfect for CSP')
                if category == 'constraint_based':
                    score += 25
                    reasons.append('✓ Constraint-based algorithm ideal for CSP')
                    
            elif 'hanoi' in problem_lower:
                if algo_name == 'DFS':
                    score += 25
                    reasons.append('✓ DFS matches recursive structure of Hanoi')
                elif algo_name == 'BFS' and complexity_level != 'large':
                    score += 15
                    reasons.append('✓ BFS finds optimal but uses more memory')
                    
            elif 'knight' in problem_lower:
                if algo_name == 'DFS':
                    score += 20
                    reasons.append('✓ DFS standard for knight tour backtracking')
            
            # Instance size considerations
            if complexity_level == 'small':
                if category == 'uninformed':
                    score += 10
                    reasons.append('✓ Simple search sufficient for small instance')
            elif complexity_level == 'large':
                if category == 'informed' and properties.get('admissible'):
                    score += 20
                    reasons.append('✓ Admissible heuristic essential for large instance')
                if 'O(d)' in complexity.get('space', '') or 'O(n)' in complexity.get('space', ''):
                    score += 10
                    reasons.append('✓ Space-efficient for large problems')
            
            if score > 0 or edge_info:
                scored_algorithms.append({
                    'algorithm': algo_name,
                    'score': score,
                    'priority': 0,
                    'reason': ' | '.join(reasons) if reasons else 'General-purpose algorithm',
                    'properties': properties,
                    'complexity': complexity,
                    'category': category,
                    'kg_scores': kg_scores,  # NEW: Include KG edge scores
                    'when_to_use': self._generate_when_to_use(algo_name, instance_analysis)
                })
        
        # Sort and assign priorities
        scored_algorithms.sort(key=lambda x: x['score'], reverse=True)
        for i, algo in enumerate(scored_algorithms, 1):
            algo['priority'] = i
        
        return scored_algorithms[:3]
    
    def _generate_when_to_use(self, algorithm: str, instance_analysis: Dict) -> str:
        """Generate usage guidance based on instance."""
        complexity = instance_analysis['complexity_level']
        
        usage_map = {
            'A*': 'Best for optimal solutions with good heuristic',
            'IDA*': 'When memory limited but optimality required',
            'BFS': 'For guaranteed optimal when no heuristic',
            'DFS': 'Ideal for backtracking and CSP problems',
            'GBFS': 'Fast approximate solutions',
            'UCS': 'When actions have varying costs',
        }
        
        return usage_map.get(algorithm, f'For {complexity} instances')
    
    def _generate_reasoning_from_kg(
        self, 
        problem_name: str, 
        instance_analysis: Dict, 
        recommendations: List[Dict]
    ) -> str:
        """Generate reasoning based on KG and instance."""
        if not recommendations:
            return "No suitable algorithms found in knowledge graph for this problem."
        
        best = recommendations[0]
        
        reasoning = f"**Best Strategy: {best['algorithm']}**\n\n"
        reasoning += f"**Instance Analysis:**\n"
        reasoning += f"- Size: {instance_analysis['instance_size']}\n"
        reasoning += f"- Complexity Level: {instance_analysis['complexity_level']}\n"
        reasoning += f"- Remaining Work: {instance_analysis['remaining_work']} items to solve\n"
        reasoning += f"- State Space: {instance_analysis['state_space_estimate']}\n\n"
        
        reasoning += f"**Why {best['algorithm']}?**\n"
        reasoning += f"{best['reason']}\n\n"
        
        if best.get('complexity'):
            comp = best['complexity']
            reasoning += f"**Complexity (from Knowledge Graph):**\n"
            if 'time' in comp:
                reasoning += f"- Time: {comp['time']}\n"
            if 'space' in comp:
                reasoning += f"- Space: {comp['space']}\n"
            reasoning += "\n"
        
        if best.get('properties'):
            props = best['properties']
            prop_list = []
            if props.get('optimal'):
                prop_list.append('Optimal')
            if props.get('complete'):
                prop_list.append('Complete')
            if props.get('admissible'):
                prop_list.append('Admissible')
            
            if prop_list:
                reasoning += f"**Algorithm Properties (from KG):** {', '.join(prop_list)}\n\n"
        
        if len(recommendations) > 1:
            reasoning += f"**Alternative Strategies:**\n"
            for alt in recommendations[1:]:
                reasoning += f"- **{alt['algorithm']}**: {alt['when_to_use']}\n"
        
        return reasoning
    def evaluate_minmax_tree(self, tree_structure: dict) -> dict:
        def build_node(subtree):
            if 'value' in subtree:
                return MinMaxNode(value=subtree['value'])
            children = [build_node(child) for child in subtree.get('children', [])]
            return MinMaxNode(children=children)
        
        root = build_node(tree_structure)
        value, leaves_visited = evaluate_tree(root, maximizing=True)
        return {'root_value': value, 'leaves_visited': leaves_visited}


if __name__ == "__main__":
    print("Testing Answer Generator with Knowledge Graph")
    print("=" * 80)
    
    gen = AnswerGenerator('knowledge_graph.json')
    
    # Test N-Queens
    print("\n[Test 1] N-Queens (5×5, 2 queens placed, 3 remaining):")
    print("-" * 80)
    answer = gen.generate_answer('N-Queens', {'n': 5, 'n_prime': 2})
    print(f"Top Recommendation: {answer['recommendations'][0]['algorithm']}")
    print(f"Score: {answer['recommendations'][0]['score']}")
    print(f"Reason: {answer['recommendations'][0]['reason']}")
    
    # Test 8-Puzzle
    print("\n[Test 2] 8-Puzzle (6 tiles misplaced):")
    print("-" * 80)
    answer = gen.generate_answer('8-Puzzle', {'initial': [[1,2,3],[4,0,5],[7,6,8]], 'goal': [[1,2,3],[4,5,6],[7,8,0]]})
    print(f"Top Recommendation: {answer['recommendations'][0]['algorithm']}")
    print(f"Score: {answer['recommendations'][0]['score']}")
    print(f"\nFull Reasoning:\n{answer['reasoning']}")
