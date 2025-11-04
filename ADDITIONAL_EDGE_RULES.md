# Additional Edge Creation Rules for Knowledge Graph

## Current Rules Summary

### 1. **Type-Based Inference** (`_infer_relation_type`)

- Algorithm → Problem: `solves`
- Problem → Algorithm: `solved_by`
- Algorithm → Heuristic: `uses`
- Heuristic → Algorithm: `used_by`
- Algorithm → Algorithm: `related_to`
- Problem → Problem: `related_to`
- Algorithm → Optimization: `can_use`

### 2. **Pattern-Based Detection** (RelationshipDetector)

- `solves`: "A\* solves 8-Puzzle"
- `uses`: "A\* uses Manhattan Distance"
- `outperforms`: "A\* is better than BFS"
- `variant_of`: "IDA* is a variant of A*"
- `requires`: "A\* requires admissible heuristic"
- `applied_to`: "BFS applied to shortest path"

### 3. **Co-occurrence Based** (proximity + frequency + sentiment)

- Creates edges between entities mentioned near each other

---

## 🆕 Additional Rules to Implement

### **Category 1: Complexity-Based Relationships**

#### Rule 1.1: Dominated By (Complexity)

**Logic:** If Algorithm A has strictly worse time/space complexity than Algorithm B for the same problem class, create edge.

```python
def _infer_complexity_dominance(self):
    """Algorithm A dominated_by Algorithm B if B has better complexity."""
    algorithms = self.knowledge_graph.get_nodes_by_type("algorithm")

    for algo1 in algorithms:
        for algo2 in algorithms:
            if algo1.id == algo2.id:
                continue

            # Compare time complexities
            time1 = self._get_complexity_order(algo1, 'time')
            time2 = self._get_complexity_order(algo2, 'time')

            # If algo2 is strictly better (e.g., O(n log n) vs O(n²))
            if time1 > time2:  # Higher order = worse
                self.knowledge_graph.add_edge(Edge(
                    source=algo1.id,
                    target=algo2.id,
                    relation_type="dominated_by",
                    confidence=0.8,
                    properties={"reason": "worse_time_complexity"}
                ))
```

**Example Edges:**

- `BFS dominated_by A*` (exponential vs polynomial with good h)
- `Bubble Sort dominated_by Merge Sort` (O(n²) vs O(n log n))

---

#### Rule 1.2: Comparable Performance

**Logic:** If two algorithms have same complexity class, create bidirectional edge.

```python
def _infer_comparable_complexity(self):
    """Algorithms with same complexity are comparable."""
    algorithms = self.knowledge_graph.get_nodes_by_type("algorithm")
    complexity_groups = defaultdict(list)

    # Group by complexity
    for algo in algorithms:
        time = self._get_complexity_order(algo, 'time')
        space = self._get_complexity_order(algo, 'space')
        key = (time, space)
        complexity_groups[key].append(algo)

    # Create edges within groups
    for algos in complexity_groups.values():
        if len(algos) > 1:
            for i, algo1 in enumerate(algos):
                for algo2 in algos[i+1:]:
                    self.knowledge_graph.add_edge(Edge(
                        source=algo1.id,
                        target=algo2.id,
                        relation_type="comparable_complexity",
                        confidence=0.9
                    ))
```

**Example Edges:**

- `BFS comparable_complexity DFS` (both O(b^d) time)
- `Merge Sort comparable_complexity Quick Sort` (both O(n log n) average)

---

### **Category 2: Problem Characteristic-Based Relationships**

#### Rule 2.1: Specialized For

**Logic:** If algorithm properties match problem characteristics, create edge.

```python
def _infer_specialization(self):
    """Algorithm specialized for problem types based on properties."""
    algorithms = self.knowledge_graph.get_nodes_by_type("algorithm")
    problems = self.knowledge_graph.get_nodes_by_type("problem")

    for algo in algorithms:
        for prob in problems:
            score = 0
            reasons = []

            # Check if algo properties match problem needs
            if 'optimal' in algo.properties and 'requires_optimal' in prob.properties:
                score += 30
                reasons.append("guarantees_optimality")

            if 'complete' in algo.properties and 'needs_completeness' in prob.properties:
                score += 20
                reasons.append("guarantees_completeness")

            # Memory-constrained problems
            if 'memory_limited' in prob.properties:
                if self._is_memory_efficient(algo):
                    score += 25
                    reasons.append("memory_efficient")

            if score >= 40:  # Threshold
                self.knowledge_graph.add_edge(Edge(
                    source=algo.id,
                    target=prob.id,
                    relation_type="specialized_for",
                    confidence=score / 100,
                    properties={"reasons": reasons}
                ))
```

**Example Edges:**

- `A* specialized_for 8-Puzzle` (optimal + complete + heuristic-friendly)
- `IDA* specialized_for Memory-Constrained-Search` (optimal + O(d) space)

---

#### Rule 2.2: Unsuitable For

**Logic:** Negative relationships based on incompatibility.

```python
def _infer_unsuitability(self):
    """Create negative edges for known incompatibilities."""
    patterns = [
        # Uninformed search unsuitable for large state spaces
        ("uninformed", "large_state_space", 0.7),
        # Non-optimal algorithms unsuitable for optimality-required problems
        ("non_optimal", "requires_optimal", 0.8),
        # Exponential space algorithms unsuitable for memory-limited
        ("exponential_space", "memory_limited", 0.9),
    ]

    for algo in self.knowledge_graph.get_nodes_by_type("algorithm"):
        for prob in self.knowledge_graph.get_nodes_by_type("problem"):
            for algo_trait, prob_trait, conf in patterns:
                if algo_trait in algo.properties and prob_trait in prob.properties:
                    self.knowledge_graph.add_edge(Edge(
                        source=algo.id,
                        target=prob.id,
                        relation_type="unsuitable_for",
                        confidence=conf,
                        sentiment_score=0.1,  # Negative relationship
                        properties={"reason": f"{algo_trait}_conflicts_with_{prob_trait}"}
                    ))
```

**Example Edges:**

- `DFS unsuitable_for Shortest-Path` (not optimal)
- `BFS unsuitable_for Large-State-Space` (exponential memory)

---

### **Category 3: Algorithm Family/Hierarchy Relationships**

#### Rule 3.1: Instance Of (Algorithm Family)

**Logic:** Detect algorithm families and create hierarchy.

```python
def _infer_algorithm_families(self):
    """Group algorithms into families based on naming patterns."""
    families = {
        'A*': ['A*', 'IDA*', 'SMA*', 'RBFS'],  # A* family
        'DFS': ['DFS', 'DLS', 'IDDFS'],         # DFS family
        'Genetic': ['Genetic Algorithm', 'Genetic Programming'],
        'Hill Climbing': ['Hill Climbing', 'Simulated Annealing', 'Tabu Search']
    }

    for family_name, members in families.items():
        # Create family node if doesn't exist
        family_id = f"family_{family_name.lower().replace(' ', '_')}"
        family_node = Node(
            id=family_id,
            name=f"{family_name} Family",
            type="algorithm_family"
        )
        self.knowledge_graph.add_node(family_node)

        # Link members to family
        for member_name in members:
            member = self._find_algorithm(member_name)
            if member:
                self.knowledge_graph.add_edge(Edge(
                    source=member.id,
                    target=family_id,
                    relation_type="instance_of",
                    confidence=1.0
                ))
```

**Example Edges:**

- `IDA* instance_of A*_Family`
- `IDDFS instance_of DFS_Family`
- `Simulated Annealing instance_of Local_Search_Family`

---

#### Rule 3.2: Improves Upon

**Logic:** Detect when one algorithm is explicitly designed to improve another.

```python
def _infer_improvements(self):
    """Detect improvement relationships from text patterns."""
    improvement_patterns = [
        r'{algo1}\s+(?:improves?|enhances?|optimizes?)\s+{algo2}',
        r'{algo1}\s+(?:is\s+)?(?:an?\s+)?improved\s+version\s+of\s+{algo2}',
        r'{algo1}\s+addresses\s+the\s+(?:limitations?|problems?)\s+of\s+{algo2}',
        r'{algo1}\s+reduces\s+the\s+(?:time|space|memory)\s+(?:of|used\s+by)\s+{algo2}'
    ]

    # Search in document content for these patterns
    # Create edges when found
```

**Example Edges:**

- `A* improves_upon Dijkstra` (adds heuristic)
- `IDA* improves_upon A*` (reduces memory)
- `RBFS improves_upon DFS` (adds backtracking with f-limit)

---

### **Category 4: Heuristic Relationships**

#### Rule 4.1: Admissible For

**Logic:** Heuristics proven admissible for specific problems.

```python
def _infer_heuristic_admissibility(self):
    """Link admissible heuristics to problems."""
    known_admissible = {
        'Manhattan Distance': ['8-Puzzle', '15-Puzzle', 'Grid Navigation'],
        'Euclidean Distance': ['TSP', 'Route Finding'],
        'Minimum Spanning Tree': ['TSP'],
        'Pattern Database': ['8-Puzzle', '15-Puzzle']
    }

    for heur_name, problem_names in known_admissible.items():
        heur = self._find_heuristic(heur_name)
        if not heur:
            continue

        for prob_name in problem_names:
            prob = self._find_problem(prob_name)
            if prob:
                self.knowledge_graph.add_edge(Edge(
                    source=heur.id,
                    target=prob.id,
                    relation_type="admissible_for",
                    confidence=1.0,
                    properties={"proof": "proven_admissible"}
                ))
```

**Example Edges:**

- `Manhattan Distance admissible_for 8-Puzzle`
- `Euclidean Distance admissible_for Route Finding`

---

#### Rule 4.2: Dominates (Heuristic)

**Logic:** Heuristic h1 dominates h2 if h1(n) ≥ h2(n) for all n.

```python
def _infer_heuristic_dominance(self):
    """Create dominance relationships between heuristics."""
    known_dominance = [
        ('Pattern Database', 'Manhattan Distance'),  # PDB dominates Manhattan
        ('Manhattan Distance', 'Misplaced Tiles'),   # Manhattan dominates Misplaced
    ]

    for dominant_name, dominated_name in known_dominance:
        dominant = self._find_heuristic(dominant_name)
        dominated = self._find_heuristic(dominated_name)

        if dominant and dominated:
            self.knowledge_graph.add_edge(Edge(
                source=dominant.id,
                target=dominated.id,
                relation_type="dominates",
                confidence=1.0,
                properties={"type": "heuristic_dominance"}
            ))
```

**Example Edges:**

- `Manhattan Distance dominates Misplaced Tiles`
- `Pattern Database dominates Manhattan Distance`

---

### **Category 5: Context-Aware Relationships**

#### Rule 5.1: Effective When (Conditional)

**Logic:** Extract conditions from sentiment context.

```python
def _infer_conditional_effectiveness(self):
    """Detect conditional relationships from context."""
    conditional_patterns = [
        (r'{algo}\s+works\s+well\s+(?:when|if)\s+(.+)', 'effective_when'),
        (r'{algo}\s+(?:is\s+)?effective\s+(?:when|if)\s+(.+)', 'effective_when'),
        (r'(?:use|apply)\s+{algo}\s+when\s+(.+)', 'recommended_when'),
        (r'{algo}\s+fails\s+(?:when|if)\s+(.+)', 'fails_when'),
    ]

    for mention in self.entity_mentions:
        if mention.entity_type != 'algorithm':
            continue

        context = mention.sentence
        for pattern, relation in conditional_patterns:
            # Extract condition
            match = re.search(pattern.replace('{algo}', mention.entity_name), context, re.I)
            if match:
                condition = match.group(1)
                # Create edge with condition stored
                self.knowledge_graph.add_edge(Edge(
                    source=mention.entity_name,
                    target="condition_node",
                    relation_type=relation,
                    properties={"condition": condition},
                    context=context
                ))
```

**Example Edges:**

- `A* effective_when "heuristic is admissible"`
- `Greedy BFS fails_when "heuristic is misleading"`
- `BFS recommended_when "all costs are equal"`

---

#### Rule 5.2: Prerequisite

**Logic:** One algorithm/technique is required before another.

```python
def _infer_prerequisites(self):
    """Detect prerequisite relationships."""
    prereq_patterns = [
        r'{algo1}\s+requires\s+{algo2}',
        r'{algo1}\s+depends\s+on\s+{algo2}',
        r'(?:first|before)\s+{algo2},\s+(?:then|use)\s+{algo1}',
        r'{algo1}\s+(?:needs|assumes)\s+{algo2}'
    ]

    # Search for patterns and create prerequisite edges
```

**Example Edges:**

- `A* requires Admissible Heuristic`
- `IDA* prerequisite Depth-First Search`
- `Alpha-Beta Pruning prerequisite Minimax`

---

### **Category 6: Quantitative Relationships**

#### Rule 6.1: Performance Ratio

**Logic:** Create weighted edges based on performance metrics.

```python
def _infer_performance_ratios(self):
    """Extract performance comparisons from numbers in text."""
    patterns = [
        r'{algo1}\s+is\s+(\d+)x\s+faster\s+than\s+{algo2}',
        r'{algo1}\s+uses\s+(\d+)%\s+less\s+memory\s+than\s+{algo2}',
        r'{algo1}\s+expands\s+(\d+)%\s+(?:fewer|less)\s+nodes\s+than\s+{algo2}'
    ]

    # Create weighted edges with performance_ratio property
```

**Example Edges:**

- `A* outperforms BFS` (confidence: 0.9, properties: {"speedup": "10x"})
- `IDA* memory_efficient_than A*` (properties: {"reduction": "90%"})

---

#### Rule 6.2: Branching Factor Dependency

**Logic:** Link algorithms to their sensitivity to branching factor.

```python
def _infer_branching_sensitivity(self):
    """Create edges based on branching factor sensitivity."""
    high_sensitivity = ['BFS', 'DFS', 'UCS']  # Exponential in b
    low_sensitivity = ['A*', 'IDA*']          # Reduced by heuristic

    for algo_name in high_sensitivity:
        algo = self._find_algorithm(algo_name)
        if algo:
            algo.properties['branching_sensitive'] = 'high'

    # Create comparative edges
```

---

### **Category 7: Problem Domain Relationships**

#### Rule 7.1: Same Domain

**Logic:** Problems in the same domain should be linked.

```python
def _infer_problem_domains(self):
    """Group problems by domain."""
    domains = {
        'CSP': ['N-Queens', 'Graph Coloring', 'Sudoku', 'Map Coloring'],
        'Puzzle': ['8-Puzzle', '15-Puzzle', 'Rubik\'s Cube'],
        'Path Finding': ['Route Finding', 'Maze', 'Grid Navigation'],
        'Game Playing': ['Chess', 'Tic-Tac-Toe', 'Go']
    }

    for domain_name, problems in domains.items():
        # Create bidirectional edges between problems in same domain
        for i, p1 in enumerate(problems):
            for p2 in problems[i+1:]:
                prob1 = self._find_problem(p1)
                prob2 = self._find_problem(p2)
                if prob1 and prob2:
                    self.knowledge_graph.add_edge(Edge(
                        source=prob1.id,
                        target=prob2.id,
                        relation_type="same_domain",
                        confidence=0.9,
                        properties={"domain": domain_name}
                    ))
```

**Example Edges:**

- `N-Queens same_domain Graph Coloring` (both CSP)
- `8-Puzzle same_domain 15-Puzzle` (both sliding puzzles)

---

#### Rule 7.2: Problem Reduction

**Logic:** One problem can be reduced to another.

```python
def _infer_problem_reductions(self):
    """Detect problem reduction relationships."""
    reduction_patterns = [
        r'{prob1}\s+(?:can\s+be\s+)?reduced\s+to\s+{prob2}',
        r'{prob1}\s+is\s+(?:a\s+)?(?:special\s+)?case\s+of\s+{prob2}',
        r'solving\s+{prob1}\s+(?:is\s+)?equivalent\s+to\s+solving\s+{prob2}'
    ]
```

**Example Edges:**

- `8-Queens reduces_to N-Queens`
- `Shortest Path reduces_to Single-Source Shortest Path`

---

### **Category 8: Optimization Technique Relationships**

#### Rule 8.1: Compatible With

**Logic:** Optimizations that work well together.

```python
def _infer_optimization_compatibility(self):
    """Detect compatible optimization techniques."""
    compatible_pairs = [
        ('Pruning', 'Forward Checking'),
        ('Memoization', 'Dynamic Programming'),
        ('Alpha-Beta Pruning', 'Move Ordering'),
        ('Iterative Deepening', 'Transposition Tables')
    ]

    for opt1_name, opt2_name in compatible_pairs:
        opt1 = self._find_optimization(opt1_name)
        opt2 = self._find_optimization(opt2_name)

        if opt1 and opt2:
            self.knowledge_graph.add_edge(Edge(
                source=opt1.id,
                target=opt2.id,
                relation_type="compatible_with",
                confidence=0.8
            ))
```

**Example Edges:**

- `Pruning compatible_with Forward Checking`
- `Memoization compatible_with Dynamic Programming`

---

### **Category 9: Temporal/Historical Relationships**

#### Rule 9.1: Precedes (Historical)

**Logic:** Track chronological development of algorithms.

```python
def _infer_historical_sequence(self):
    """Create timeline of algorithm development."""
    chronology = [
        ('DFS', 'IDDFS', 1975),      # IDDFS developed after DFS
        ('Dijkstra', 'A*', 1968),    # A* extends Dijkstra
        ('A*', 'IDA*', 1985),        # IDA* improves A*
    ]

    for earlier, later, year in chronology:
        early_algo = self._find_algorithm(earlier)
        later_algo = self._find_algorithm(later)

        if early_algo and later_algo:
            self.knowledge_graph.add_edge(Edge(
                source=early_algo.id,
                target=later_algo.id,
                relation_type="precedes",
                confidence=1.0,
                properties={"year": year, "type": "historical"}
            ))
```

---

### **Category 10: Multi-hop Inference Rules**

#### Rule 10.1: Transitive Suitability

**Logic:** If A solves B and B reduces_to C, then A solves C.

```python
def _infer_transitive_suitability(self):
    """
    If Algorithm A solves Problem B
    AND Problem B reduces_to Problem C
    THEN Algorithm A can solve Problem C (with lower confidence)
    """
    for algo in self.knowledge_graph.get_nodes_by_type("algorithm"):
        # Find problems this algorithm solves
        solves_edges = self.knowledge_graph.get_outgoing_edges(algo.id, "solves")

        for solve_edge in solves_edges:
            prob = self.knowledge_graph.get_node(solve_edge.target)

            # Find what this problem reduces to
            reduction_edges = self.knowledge_graph.get_outgoing_edges(prob.id, "reduces_to")

            for reduction in reduction_edges:
                target_prob = reduction.target

                # Create indirect solving edge
                self.knowledge_graph.add_edge(Edge(
                    source=algo.id,
                    target=target_prob,
                    relation_type="can_solve",
                    confidence=min(solve_edge.confidence, reduction.confidence) * 0.7,
                    properties={
                        "inferred": True,
                        "via_problem": prob.id,
                        "reasoning": "transitive_through_reduction"
                    }
                ))
```

---

## Summary: Prioritized Implementation Order

### **High Priority** (Implement First)

1. ✅ **Complexity Dominance** - Clear, objective criteria
2. ✅ **Specialization Rules** - Match algo properties to problem needs
3. ✅ **Algorithm Families** - Organize knowledge hierarchy
4. ✅ **Admissible Heuristics** - Critical for informed search

### **Medium Priority**

5. **Conditional Effectiveness** - Extract "when to use" rules
6. **Problem Domain Grouping** - Organize problems semantically
7. **Unsuitable For** - Negative relationships prevent bad recommendations
8. **Heuristic Dominance** - Rank heuristics

### **Low Priority** (Nice to Have)

9. **Historical Relationships** - Interesting but not critical
10. **Performance Ratios** - Requires parsing numbers from text
11. **Multi-hop Inference** - Can be complex and error-prone

---

## Implementation Strategy

### Phase 1: Rule-Based (Deterministic)

- Algorithm families
- Complexity comparisons
- Known admissible heuristics
- Problem domains

### Phase 2: Pattern-Based (Text Mining)

- Conditional effectiveness
- Improvement relationships
- Prerequisites
- Performance comparisons

### Phase 3: Inference-Based (Reasoning)

- Transitive relationships
- Unsuitability detection
- Compatibility checks
- Multi-hop inference

---

## Expected Impact

**Before:** 123 edges (mostly co-occurrence)
**After:** 300-500 edges (structured + inferred + semantic)

**Benefits:**

- 🎯 More accurate recommendations
- 🧠 Deeper semantic understanding
- 🔗 Richer graph connectivity
- 📊 Better coverage of algorithm space
- ⚡ Stronger justifications in answers
