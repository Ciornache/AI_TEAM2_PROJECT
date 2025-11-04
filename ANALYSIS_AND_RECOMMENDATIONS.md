# Knowledge Graph Analysis & Recommendations

## Your Goal

**Generate questions for a given problem where:**

- Each question has multiple instances of that problem
- For each instance, determine which algorithm works best + reasoning
- The answer should be determined by the knowledge graph

---

## Current Implementation Analysis

### ✅ **What Works Well**

1. **Entity Extraction**

   - Comprehensive pre-seeded entities (algorithms, problems, heuristics)
   - Pattern-based extraction with aliases (handles multiple languages)
   - Context-aware extraction (before/after windows, sentence context)

2. **Current Relationship Detection**

   - Pattern matching for explicit relationships (`solves`, `uses`, `outperforms`)
   - Co-occurrence analysis within sliding windows (50 words)
   - Transitivity inference (A uses B, B solves C => A applicable to C)
   - Category-based relationships

3. **Answer Generation**
   - Instance-specific analysis (size, complexity level, remaining work)
   - Scoring algorithms based on KG properties + instance characteristics
   - Reasoning generation from KG

### ❌ **Current Limitations**

1. **Edge Scoring Issues**

   - **Current approach**: Co-occurrence strength = `count / max_count`
   - **Problem**: Doesn't consider:
     - Distance between entities in text
     - Document length (normalization)
     - Frequency relative to document size
     - Multiple co-occurrences in same vs different documents

2. **Missing Entity Types**

   - ❌ No explicit **Time Complexity** nodes (only as properties)
   - ❌ No explicit **Memory Complexity** nodes
   - ❌ No **constraint-based** category for algorithms

3. **Missing Attributes**

   - ❌ Algorithms not categorized as Informed/Uninformed/Constraint-based
   - ❌ No performance characteristics for different instance types

4. **Instance-Specific Guidance**
   - Current: General complexity analysis
   - Missing: "A\* works well for sparse graphs with N < 100"
   - Missing: "DFS optimal for N-Queens with N > 8"

---

## Your Proposed Solution

### **Entities**

```
- Algorithm (Informed, Uninformed, Constraint-based)
- Problem
- Heuristic
- Time Complexity
- Memory Complexity
```

### **Edge Scoring**

**Your idea**: Score based on proximity and frequency

- Average distance between entities in document
- Normalize by document length
- Weight by frequency of co-occurrence

**Example**:

```
If (A*, Manhattan) appear:
- 3 times in a 1000-word document at distances [20, 35, 15] words
- Average distance: 23.3 words
- Document-normalized score: f(frequency=3, avg_distance=23.3, doc_length=1000)
```

---

## 🚨 **Critical Analysis: Will This Work?**

### ✅ **YES, with modifications** - Here's why:

### **What Will Work**

1. ✅ **Distance-based scoring is EXCELLENT for your goal**

   - Entities mentioned close together = stronger relationship
   - Averaging distances across mentions = good signal
   - Better than current "just count co-occurrences"

2. ✅ **Document normalization is SMART**

   - Prevents bias toward longer documents
   - Pair appearing 3x in 500-word doc ≠ 3x in 5000-word doc

3. ✅ **Adding explicit complexity nodes will HELP**

   - Allows querying: "Which algorithms have O(n log n) time?"
   - Enables: "Show me all O(n) space algorithms for Problem X"

4. ✅ **Constraint-based category is NECESSARY**
   - CSP algorithms (backtracking, constraint propagation) are fundamentally different
   - N-Queens, Graph Coloring, Sudoku all need this

---

## ⚠️ **Problems with Your Approach**

### **Issue 1: Distance Alone is Insufficient**

**Problem**: Two entities close together ≠ strong relationship

**Example from docs**:

```
"Unlike BFS, DFS does not guarantee optimal solutions.
Instead, use A* for optimality."
```

- `(DFS, optimal)` distance = 5 words
- But relationship is **NEGATIVE** ("does not guarantee")

**Your scoring would give**:

- High score (close distance)
- But actually means DFS ≠ optimal

### **Issue 2: Relationship Type Matters More Than Score**

**Your goal**: Determine which algorithm is best for an instance

**What matters**:

- **Positive relationships**: "A\* solves 8-Puzzle efficiently"
- **Negative relationships**: "BFS fails for large instances"
- **Conditional relationships**: "DFS works IF memory limited"

**Proximity score doesn't capture**:

- Sentiment (positive/negative/conditional)
- Causality ("A\* is fast BECAUSE of heuristic")
- Context ("optimal ONLY for admissible heuristics")

### **Issue 3: Multi-Document Knowledge Merging**

You merge multiple PDFs. Consider:

**Document 1** (100 pages):

- Mentions (A\*, 8-Puzzle) 50 times, avg distance: 30 words

**Document 2** (10 pages):

- Mentions (A\*, 8-Puzzle) 5 times, avg distance: 15 words

**Your normalization**:

- Doc1: score(50, 30, 100_pages)
- Doc2: score(5, 15, 10_pages)

**Problem**: How to merge? Average? Max? Sum?

- If you average: Doc2 (tighter connection) might override Doc1 (more evidence)
- If you sum: Longer docs dominate

---

## ✅ **My Recommendations**

### **1. Hybrid Scoring System**

**Combine multiple signals** (not just distance):

```python
edge_score = weighted_average([
    proximity_score,      # Your idea (distance-based)
    frequency_score,      # How often mentioned together
    sentiment_score,      # Positive/negative relationship
    contextual_score,     # Based on surrounding words
    source_credibility    # Weight different documents differently
])
```

#### **Proximity Score** (Your idea - KEEP IT!)

```python
def proximity_score(mentions_list, doc_length):
    """
    mentions_list: [(pos1_entity1, pos1_entity2), (pos2_entity1, pos2_entity2), ...]
    """
    distances = [abs(pos1 - pos2) for pos1, pos2 in mentions_list]
    avg_distance = sum(distances) / len(distances)

    # Normalize by document length
    normalized_distance = avg_distance / doc_length

    # Inverse: closer = higher score
    # Use sigmoid to bound between 0 and 1
    score = 1 / (1 + normalized_distance * 10)  # Adjust multiplier

    return score
```

#### **Frequency Score** (Already have this - KEEP IT!)

```python
def frequency_score(co_occurrence_count, total_mentions_entity1, total_mentions_entity2):
    """
    How often they appear together vs independently
    """
    # PMI (Pointwise Mutual Information) - standard metric
    p_together = co_occurrence_count / (total_mentions_entity1 + total_mentions_entity2)
    p_entity1 = total_mentions_entity1 / total_document_words
    p_entity2 = total_mentions_entity2 / total_document_words

    pmi = log(p_together / (p_entity1 * p_entity2))

    # Normalize to [0, 1]
    return sigmoid(pmi)
```

#### **Sentiment Score** (NEW - ADD THIS!)

```python
def sentiment_score(context_text):
    """
    Analyze context for positive/negative indicators
    """
    positive_patterns = [
        'optimal', 'best', 'efficient', 'guarantees', 'always',
        'works well', 'suitable for', 'ideal', 'recommended'
    ]

    negative_patterns = [
        'not optimal', 'inefficient', 'fails', 'cannot', 'never',
        'does not work', 'unsuitable', 'poor performance'
    ]

    conditional_patterns = [
        'if', 'when', 'only if', 'provided that', 'as long as'
    ]

    pos_count = count_patterns(context_text, positive_patterns)
    neg_count = count_patterns(context_text, negative_patterns)
    cond_count = count_patterns(context_text, conditional_patterns)

    # Positive = 1.0, Negative = 0.0, Conditional = 0.5
    if pos_count > neg_count:
        return 1.0 - (cond_count * 0.2)  # Reduce if conditional
    elif neg_count > pos_count:
        return 0.2  # Keep some score (still provides info)
    else:
        return 0.5
```

#### **Final Edge Score**

```python
def calculate_edge_score(entity1, entity2, mentions, doc_metadata):
    prox = proximity_score(mentions, doc_metadata['length'])
    freq = frequency_score(co_occ_count, mentions1_total, mentions2_total)
    sent = sentiment_score(context_texts)

    # Weighted average (tune these weights!)
    score = 0.35 * prox + 0.30 * freq + 0.35 * sent

    return score
```

---

### **2. Add Missing Entity Types & Attributes**

#### **Update `knowledge_graph.py`**

```python
@dataclass
class Node:
    id: str
    name: str
    type: str  # ADD: "time_complexity", "memory_complexity"
    properties: Dict[str, Any] = field(default_factory=dict)

    # NEW: Performance characteristics per instance type
    performance_profiles: Dict[str, Dict] = field(default_factory=dict)
    # Example:
    # {
    #   "small_instance": {"efficiency": 0.9, "avg_time": "fast"},
    #   "large_instance": {"efficiency": 0.3, "avg_time": "slow"}
    # }
```

#### **Update Algorithm Categories**

```python
# In document_processor_v5.py
def _init_algorithms(self):
    return {
        "a_star": {
            "name": "A*",
            "category": "informed",  # KEEP THIS
            "algorithm_type": "informed",  # ADD: For your proposed structure
            "constraint_based": False,  # ADD: Your proposed attribute
            "default_properties": {
                "complete": True,
                "optimal": True,
                "admissible": True
            },
            # ADD: Instance-specific performance
            "performance_by_instance": {
                "small": {"time": "O(b^d)", "space": "O(b^d)", "practical": "excellent"},
                "medium": {"time": "O(b^d)", "space": "O(b^d)", "practical": "good"},
                "large": {"time": "O(b^d)", "space": "O(b^d)", "practical": "depends_on_heuristic"}
            }
        },
        # Add constraint-based algorithms
        "backtracking": {
            "name": "Backtracking",
            "category": "constraint_based",  # NEW CATEGORY
            "algorithm_type": "constraint_based",
            "constraint_based": True,
            "default_properties": {
                "complete": True,
                "optimal": False  # Finds any solution, not necessarily optimal
            }
        },
        "forward_checking": {
            "name": "Forward Checking",
            "category": "constraint_based",
            "algorithm_type": "constraint_based",
            "constraint_based": True,
            # ...
        }
    }
```

#### **Add Complexity Nodes**

```python
def _preseed_knowledge_graph(self):
    # ... existing code ...

    # ADD: Time complexity nodes
    time_complexities = {
        "O(1)": "Constant Time",
        "O(log n)": "Logarithmic Time",
        "O(n)": "Linear Time",
        "O(n log n)": "Linearithmic Time",
        "O(n²)": "Quadratic Time",
        "O(n³)": "Cubic Time",
        "O(2^n)": "Exponential Time",
        "O(n!)": "Factorial Time",
        "O(b^d)": "Exponential in Depth"
    }

    for notation, description in time_complexities.items():
        node = Node(
            id=f"time_{notation.replace('(', '').replace(')', '').replace('^', '_')}",
            name=notation,
            type="time_complexity",
            properties={
                "description": description,
                # ADD: Performance ratings for different instance sizes
                "suitable_for": {
                    "small": notation in ["O(n²)", "O(n³)", "O(2^n)", "O(n!)"],
                    "medium": notation in ["O(n log n)", "O(n²)"],
                    "large": notation in ["O(1)", "O(log n)", "O(n)", "O(n log n)"]
                }
            }
        )
        self.knowledge_graph.add_node(node)

    # ADD: Memory complexity nodes (similar structure)
    memory_complexities = {
        "O(1)": "Constant Space",
        "O(log n)": "Logarithmic Space",
        "O(n)": "Linear Space",
        "O(n²)": "Quadratic Space",
        "O(b^d)": "Exponential Space"
    }

    for notation, description in memory_complexities.items():
        node = Node(
            id=f"mem_{notation.replace('(', '').replace(')', '').replace('^', '_')}",
            name=notation,
            type="memory_complexity",
            properties={
                "description": description,
                "suitable_for": {
                    "small": True,  # Any space OK for small instances
                    "medium": notation in ["O(1)", "O(log n)", "O(n)"],
                    "large": notation in ["O(1)", "O(log n)", "O(n)"]  # Large needs efficiency
                }
            }
        )
        self.knowledge_graph.add_node(node)
```

---

### **3. Enhanced Edge Structure**

```python
@dataclass
class Edge:
    source: str
    target: str
    relation_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    context: str = ""

    # ADD: Your distance-based scoring
    proximity_score: float = 0.0
    frequency_score: float = 0.0
    sentiment_score: float = 0.0

    # ADD: Multi-document support
    source_documents: List[str] = field(default_factory=list)
    per_document_scores: Dict[str, float] = field(default_factory=dict)

    # ADD: Instance-specific applicability
    instance_conditions: Dict[str, Any] = field(default_factory=dict)
    # Example: {"instance_size": "small", "constraint_density": "high"}
```

---

### **4. Instance-Aware Answer Generation**

**Current**: Your answer generator analyzes instance → scores algorithms

**Enhancement**: Add instance-specific edge filtering

```python
class AnswerGenerator:
    def generate_answer(self, problem_name: str, instance_data: dict):
        # Current: Analyze instance
        instance_analysis = self._analyze_instance(problem_name, instance_data)

        # NEW: Filter KG edges by instance characteristics
        relevant_edges = self._filter_edges_by_instance(
            problem_name,
            instance_analysis
        )

        # Score algorithms using filtered edges
        recommendations = self._score_algorithms_with_instance_edges(
            instance_analysis,
            relevant_edges
        )

        return {
            'recommendations': recommendations,
            'reasoning': self._generate_reasoning(recommendations, instance_analysis)
        }

    def _filter_edges_by_instance(self, problem_name, instance_analysis):
        """
        Filter KG edges based on instance characteristics
        """
        problem_node = self.find_problem_node(problem_name)
        all_edges = self.kg.get_outgoing_edges(problem_node['id'])

        relevant_edges = []
        for edge in all_edges:
            # Check if edge applies to this instance
            conditions = edge.instance_conditions

            if self._edge_matches_instance(conditions, instance_analysis):
                # Weight edge by how well it matches
                match_score = self._calculate_instance_match_score(
                    conditions,
                    instance_analysis
                )
                edge_with_score = {**edge, 'instance_match_score': match_score}
                relevant_edges.append(edge_with_score)

        return relevant_edges

    def _edge_matches_instance(self, conditions, instance):
        """Check if edge conditions match instance"""
        if not conditions:
            return True  # No conditions = applies to all

        # Check size condition
        if 'size' in conditions:
            if conditions['size'] != instance['complexity_level']:
                return False

        # Check other conditions...
        return True
```

---

### **5. Multi-Document Score Aggregation**

When merging KG from multiple documents:

```python
def merge_knowledge_graphs(kg1: KnowledgeGraph, kg2: KnowledgeGraph):
    merged = KnowledgeGraph()

    # ... merge nodes (existing code) ...

    # NEW: Smart edge merging
    edge_map = {}

    for kg, doc_name in [(kg1, "doc1"), (kg2, "doc2")]:
        for edge in kg.edges:
            key = (edge.source, edge.target, edge.relation_type)

            if key not in edge_map:
                edge_map[key] = edge
                edge.source_documents = [doc_name]
                edge.per_document_scores[doc_name] = edge.confidence
            else:
                existing = edge_map[key]
                existing.source_documents.append(doc_name)
                existing.per_document_scores[doc_name] = edge.confidence

                # Aggregate scores (multiple strategies)
                scores = list(existing.per_document_scores.values())

                # Strategy 1: Weighted average by document length
                # Strategy 2: Max score (most confident source wins)
                # Strategy 3: Bayesian combination

                # Example: Weighted average
                existing.confidence = sum(scores) / len(scores)

                # Merge proximity scores
                existing.proximity_score = max(
                    existing.proximity_score,
                    edge.proximity_score
                )

                # Aggregate frequency
                existing.frequency_score += edge.frequency_score

                # Sentiment: keep most positive (assuming correct source)
                existing.sentiment_score = max(
                    existing.sentiment_score,
                    edge.sentiment_score
                )

    # Add all merged edges
    for edge in edge_map.values():
        merged.add_edge(edge)

    return merged
```

---

## 🎯 **Final Recommendation: Implementation Plan**

### **Phase 1: Core Improvements** (Do First)

1. ✅ **Add explicit complexity nodes**

   - Time complexity nodes
   - Memory complexity nodes
   - Link algorithms → complexities

2. ✅ **Add constraint-based category**

   - Update algorithm definitions
   - Add backtracking, forward checking, etc.

3. ✅ **Implement proximity scoring** (Your idea!)

   - Calculate average distance between entities
   - Normalize by document length
   - Add to edge properties

4. ✅ **Implement sentiment analysis**
   - Detect positive/negative/conditional contexts
   - Weight edges accordingly

### **Phase 2: Enhanced Scoring** (Do Second)

5. ✅ **Combine multiple scoring signals**

   - Proximity (35%)
   - Frequency (30%)
   - Sentiment (35%)

6. ✅ **Multi-document aggregation**
   - Track per-document scores
   - Intelligent merging strategy

### **Phase 3: Instance-Specific** (Do Third)

7. ✅ **Add instance-specific edge properties**

   - `instance_conditions: {"size": "large", ...}`
   - Filter edges by instance match

8. ✅ **Performance profiles per instance type**
   - "A\* works well for small-medium 8-Puzzles"
   - "DFS best for large N-Queens"

---

## 🔍 **Validation: Will This Achieve Your Goal?**

### **Your Goal**: Generate questions → Determine best algorithm per instance → Provide reasoning

### **With Current System**:

- ❌ Edges are boolean (exists or not)
- ❌ No instance-specific information
- ❌ Answer generator **hardcodes** instance logic (not from KG)

### **With Recommended System**:

- ✅ Edges have rich scores (proximity, sentiment, frequency)
- ✅ Edges can be filtered by instance characteristics
- ✅ Answer generator **queries KG** for:
  - "Which algorithms solve 8-Puzzle with high confidence?"
  - "Which edges apply to large instances?"
  - "What's the sentiment for (A\*, 8-Puzzle, large)?"
- ✅ Reasoning is **derived from KG**, not hardcoded

### **Example Query Flow**:

```
Question: "8-Puzzle with 7 misplaced tiles (large instance)"

1. Analyze instance:
   - problem: "8-Puzzle"
   - complexity_level: "large"
   - misplaced_tiles: 7

2. Query KG:
   - Find problem node: "8-Puzzle"
   - Get all "solved_by" edges
   - Filter edges where:
     * instance_conditions["size"] in ["medium", "large"]
     * sentiment_score > 0.7 (positive relationship)

3. Rank algorithms by edge scores:
   - (A*, 8-Puzzle): proximity=0.85, freq=0.90, sent=1.0 → 0.92
   - (IDA*, 8-Puzzle): proximity=0.75, freq=0.65, sent=0.9 → 0.77
   - (BFS, 8-Puzzle): proximity=0.60, freq=0.80, sent=0.5 → 0.63

4. Generate reasoning from KG:
   - "A* selected because:
     * Mentioned 15 times near 8-Puzzle (avg 25 words apart)
     * Contexts: 'A* is optimal for sliding puzzles'
     * Time: O(b^d), Memory: O(b^d)
     * Suitable for large instances with good heuristic"
```

---

## ✅ **Conclusion**

### **Your Proposal**: ✅ **Good foundation, needs enhancements**

**Keep**:

- ✅ Distance-based scoring (proximity)
- ✅ Document normalization
- ✅ Adding explicit complexity nodes
- ✅ Constraint-based category

**Add**:

- ✅ Sentiment analysis (positive/negative/conditional)
- ✅ Multi-signal scoring (not just distance)
- ✅ Instance-specific edge filtering
- ✅ Smart multi-document merging

### **Will This Work?**

**YES** - with the recommended hybrid approach combining:

1. Your proximity scoring (distance-based)
2. Sentiment analysis (relationship quality)
3. Instance-specific filtering (applicability)

This will enable the KG to answer: **"What's the best algorithm for THIS specific instance?"**
and provide reasoning directly from the graph structure and edge properties.
