# Knowledge Graph Edge Calculation Explained

## Overview

The knowledge graph creates edges between entities based on **document analysis** with three main scoring components:

---

## 1. Proximity Score (Distance-Based)

**Location:** `CoOccurrenceAnalyzer.get_proximity_score()`

### How it works:

- Tracks the **word distance** between entity mentions in the document
- Uses a **sliding window** of 50 words to detect co-occurrences
- Calculates **average distance** across all co-occurrences
- **Normalizes** by document length

### Formula:

```python
proximity_score = 1.0 / (1.0 + (avg_distance / doc_length) * 10)
```

### Result:

- Score range: **0.0 to 1.0**
- **Higher score** = entities mentioned closer together
- **Lower score** = entities mentioned far apart

### Example:

```
Document: "A* algorithm uses Manhattan distance heuristic..."
- "A*" and "Manhattan distance" are 3 words apart
- High proximity score (~0.75+)
```

---

## 2. Frequency Score (Co-occurrence Strength)

**Location:** `CoOccurrenceAnalyzer.get_co_occurrence_strength()`

### How it works:

- Counts how many times two entities appear **within the same window**
- Normalizes by the **geometric mean** of individual entity counts
- Prevents bias toward frequently-mentioned entities

### Formula:

```python
frequency_score = co_occurrence_count / sqrt(count_entity1 * count_entity2)
```

### Result:

- Score range: **0.0 to ~1.0**
- **Higher score** = entities frequently mentioned together
- Normalized to avoid bias

### Example:

```
Document mentions:
- "A*" appears 10 times
- "8-Puzzle" appears 8 times
- They co-occur 6 times
→ frequency_score = 6 / sqrt(10 * 8) ≈ 0.67
```

---

## 3. Sentiment Score (Relationship Quality)

**Location:** `SentimentAnalyzer.analyze_sentiment()`

### How it works:

- Analyzes the **context** around entity co-occurrences
- Uses **regex patterns** to detect:
  - **Positive indicators**: "optimal", "best", "efficient", "works well"
  - **Negative indicators**: "fails", "inefficient", "poor", "cannot"
  - **Conditional indicators**: "if", "only when", "depends on"

### Formula:

```python
if positive_count > negative_count:
    sentiment = 0.7 + (min(positive_count, 3) * 0.1) - (conditional_count * 0.1)
elif negative_count > positive_count:
    sentiment = 0.3 - (min(negative_count, 3) * 0.1)
else:
    sentiment = 0.5  # Neutral
```

### Result:

- Score range: **0.0 to 1.0**
- **1.0** = very positive relationship
- **0.5** = neutral relationship
- **0.0** = negative relationship

### Example:

```
Context: "A* is optimal and efficient for 8-Puzzle"
→ Positive patterns: "optimal", "efficient"
→ sentiment_score ≈ 0.9

Context: "BFS is inefficient for large 8-Puzzle instances"
→ Negative pattern: "inefficient"
→ sentiment_score ≈ 0.2
```

---

## 4. Hybrid Confidence Score

**Location:** Edge creation in `_create_cooccurrence_edges()`

### How it works:

Combines all three scores with **weighted averaging**:

### Formula:

```python
confidence = 0.35 × proximity + 0.30 × frequency + 0.35 × sentiment
```

### Weights:

- **35%** Proximity (how close mentions are)
- **30%** Frequency (how often they co-occur)
- **35%** Sentiment (quality of relationship)

### Result:

- Final confidence score: **0.0 to 1.0**
- Used to rank edge importance

### Example:

```
proximity_score = 0.8 (close together)
frequency_score = 0.6 (mentioned together often)
sentiment_score = 0.9 (positive relationship)

confidence = 0.35 × 0.8 + 0.30 × 0.6 + 0.35 × 0.9
           = 0.28 + 0.18 + 0.315
           = 0.775
```

---

## 5. Multi-Document Aggregation

**Location:** `build_comprehensive_kg.py` - `merge_knowledge_graphs()`

### When merging edges from multiple documents:

- **Proximity**: Takes **maximum** (best case scenario)
- **Frequency**: **Sums with dampening** (more evidence, but with diminishing returns)
- **Sentiment**: **Weighted average** (balanced view across documents)

### Formula:

```python
merged_proximity = max(doc1_prox, doc2_prox, ...)
merged_frequency = sum(freq * (1 / sqrt(i))) for each document
merged_sentiment = weighted_avg by document confidence
```

---

## 6. Edge Creation Process

### Step-by-Step:

1. **Extract mentions**: Find all entities in document
2. **Build co-occurrence matrix**: Track which entities appear together
3. **Calculate scores**:
   - Proximity from average distance
   - Frequency from co-occurrence counts
   - Sentiment from context analysis
4. **Create edge** with all scores stored
5. **Filter**: Only edges with confidence > threshold

### Edge Data Structure:

```python
Edge(
    source="A*",
    target="8-Puzzle",
    relation_type="solves",
    confidence=0.775,          # Hybrid score
    proximity_score=0.8,       # Distance-based
    frequency_score=0.6,       # Co-occurrence
    sentiment_score=0.9,       # Context sentiment
    source_documents=["doc1.pdf"],
    per_document_scores={...}
)
```

---

## Summary

The edge calculation is **entirely data-driven**:

- ✅ **Proximity**: Calculated from word positions in document
- ✅ **Frequency**: Calculated from co-occurrence counts
- ✅ **Sentiment**: Calculated from context patterns
- ✅ **Confidence**: Weighted combination of above

**No hardcoded relationships** - all edges come from document analysis!
