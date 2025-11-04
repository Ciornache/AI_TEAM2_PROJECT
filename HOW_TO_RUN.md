# 🚀 How to Run the Project

## Quick Start (3 Steps)

### Step 1: Build the Knowledge Graph

```powershell
python build_comprehensive_kg.py
```

**What this does:**

- Processes all PDF documents in `Knowledge Source/` folder
- Extracts entities (algorithms, problems, heuristics)
- Calculates proximity, frequency, and sentiment scores for relationships
- Merges knowledge from multiple documents
- Outputs: `knowledge_graph.json`

**Expected output:**

```
Building Comprehensive Knowledge Graph from All Sources
...
Total Nodes: 58
Total Edges: 190
✓ Knowledge Graph Build Complete!
```

**Time:** ~30 seconds

---

### Step 2: Generate Questions

```powershell
python generate_questions.py
```

**What this does:**

- Reads the knowledge graph
- Generates questions for 5 problems (N-Queens, Hanoi, Graph Coloring, Knight's Tour, 8-Puzzle)
- For each problem, creates multiple instances with varying difficulty
- Determines best algorithm using KG-driven reasoning
- Outputs: PDF file `AI_Search_Questions_<timestamp>.pdf`

**Expected output:**

```
AI SEARCH STRATEGY QUESTION GENERATOR
Configuration:
  - Problems: 5
  - Instances per problem: 2-3 (random)
  - Total questions: 10-15

Generating PDF...
✓ SUCCESS!
PDF generated: AI_Search_Questions_20251104_143022.pdf
```

**Time:** ~10 seconds

---

### Step 3: View the Results

Open the generated PDF file to see:

- Problem instances with visualizations
- Questions about best solving strategy
- **KG-driven answers with reasoning**
- Complexity analysis
- Alternative strategies

---

## Advanced Usage

### Test Individual Components

#### Test the Enhanced Answer Generator

```powershell
python answer_generator.py
```

Tests the answer generation with 2 example problems.

#### Test All New Features

```powershell
python test_enhanced_features.py
```

Comprehensive test of:

- Complexity nodes
- Constraint-based algorithms
- Edge scoring components
- Sentiment analysis
- Multi-document tracking
- Instance-aware filtering
- Proximity scoring

#### Test KG Connections

```powershell
python check_kg_connections.py
```

Analyzes the knowledge graph structure and relationships.

---

## Project Structure

```
AI_TEAM2_PROJECT/
│
├── Knowledge Source/          # Input PDF documents
│   ├── ai-lecture03.pdf
│   ├── IA_2_SBM_I.pdf
│   ├── IA_3_SBM_II.pdf
│   ├── lecture2.pdf
│   └── Unit-3.pdf
│
├── Core Files (Run These):
│   ├── build_comprehensive_kg.py    # Step 1: Build KG
│   ├── generate_questions.py        # Step 2: Generate questions
│   └── answer_generator.py          # Answer generation logic
│
├── Core Libraries:
│   ├── knowledge_graph.py           # KG data structures (Node, Edge)
│   ├── document_processor_v5.py     # Entity extraction, scoring
│   ├── pdf_generator.py             # PDF output
│   └── problem_generators.py        # Problem instance generation
│
├── Testing:
│   ├── test_enhanced_features.py    # Feature tests
│   ├── test_v5_processor.py         # Processor tests
│   └── check_kg_connections.py      # KG analysis
│
├── Output:
│   ├── knowledge_graph.json         # Generated KG
│   └── AI_Search_Questions_*.pdf    # Generated questions
│
└── Documentation:
    ├── README.md
    ├── ANALYSIS_AND_RECOMMENDATIONS.md
    ├── IMPLEMENTATION_SUMMARY.md
    └── FINAL_STATUS.md
```

---

## Requirements

### Install Dependencies

```powershell
pip install -r requirements.txt
```

**Required packages:**

- PyPDF2 (PDF reading)
- reportlab (PDF generation)
- beautifulsoup4 (if processing web content)
- requests (if fetching web resources)

---

## Typical Workflow

### 1. **First Time Setup**

```powershell
# Install dependencies
pip install -r requirements.txt

# Build the knowledge graph
python build_comprehensive_kg.py
```

### 2. **Generate Questions (Daily Use)**

```powershell
# Generate new questions with random instances
python generate_questions.py

# Or specify number of instances per problem
python generate_questions.py 2
```

### 3. **Update Knowledge Graph (When PDFs Change)**

```powershell
# Rebuild KG from scratch
python build_comprehensive_kg.py
```

---

## Understanding the Output

### Knowledge Graph JSON Structure

```json
{
  "nodes": [
    {
      "id": "algo_a_star",
      "name": "A*",
      "type": "algorithm",
      "properties": { "optimal": true, "complete": true },
      "performance_profiles": {
        "small": { "efficiency": 0.9 },
        "large": { "efficiency": 0.7 }
      }
    }
  ],
  "edges": [
    {
      "source": "algo_a_star",
      "target": "prob_8_puzzle",
      "relation_type": "solves",
      "confidence": 0.92,
      "proximity_score": 0.85,
      "frequency_score": 0.9,
      "sentiment_score": 1.0,
      "source_documents": ["ai-lecture03.pdf", "Unit-3.pdf"]
    }
  ]
}
```

### Generated Question Format

Each question includes:

1. **Problem Instance** - Specific parameters (e.g., 8-Puzzle with 7 misplaced tiles)
2. **Visualization** - Graphical representation
3. **Question** - "What solving strategy works best?"
4. **Answer** - KG-driven recommendation with:
   - Best algorithm (e.g., A\*)
   - Score breakdown (proximity, sentiment, frequency)
   - Reasoning from knowledge graph
   - Complexity analysis
   - Alternative strategies

---

## Troubleshooting

### "Module not found" error

```powershell
pip install -r requirements.txt
```

### "File not found: knowledge_graph.json"

```powershell
# Build it first
python build_comprehensive_kg.py
```

### "No such file or directory: Knowledge Source/"

Make sure PDF files are in the `Knowledge Source/` folder.

### Generated PDF has no content

```powershell
# Rebuild the knowledge graph
python build_comprehensive_kg.py

# Then regenerate questions
python generate_questions.py
```

---

## What's New (Enhanced Features)

The system now includes:

✨ **Distance-based proximity scoring**

- Entities mentioned close together get higher scores

✨ **Sentiment analysis**

- Distinguishes positive ("A\* is optimal") from negative ("DFS fails for large instances")

✨ **Hybrid edge scoring**

- 35% proximity + 30% frequency + 35% sentiment

✨ **Multi-document evidence**

- Relationships mentioned in multiple PDFs get higher confidence

✨ **Instance-aware filtering**

- Recommends different algorithms for small vs large instances

✨ **Explicit complexity nodes**

- 9 time complexity nodes (O(1) through O(b^d))
- 5 memory complexity nodes

✨ **Constraint-based algorithms**

- Backtracking, Forward Checking, Arc Consistency, etc.

---

## Example Session

```powershell
# Start fresh
PS> python build_comprehensive_kg.py
Building Comprehensive Knowledge Graph...
[+] Pre-seeded: 58 nodes, 20 edges
Processing 5 documents...
✓ Merge Complete!
Total Nodes: 58
Total Edges: 190

# Generate questions
PS> python generate_questions.py
Configuration:
  - Problems: 5
  - Instances per problem: 2
  - Total questions: 10

Generating PDF...
✓ SUCCESS!
PDF generated: AI_Search_Questions_20251104_143022.pdf

# Test the system
PS> python test_enhanced_features.py
✅ Phase 1: Core Improvements - WORKING
✅ Phase 2: Enhanced Scoring - WORKING
✅ Phase 3: Instance-Specific - WORKING
System is ready for production use!
```

---

## Performance

- **Build KG**: ~30 seconds (5 PDFs)
- **Generate Questions**: ~10 seconds (5 problems × 2-3 instances)
- **Answer Generation**: <1 second per instance

---

## Next Steps

After running the project:

1. **Review the generated PDF** - See questions and KG-driven answers
2. **Inspect `knowledge_graph.json`** - Understand the graph structure
3. **Run tests** - Verify all features work: `python test_enhanced_features.py`
4. **Customize** - Modify problem generators in `problem_generators.py`
5. **Add PDFs** - Place new PDFs in `Knowledge Source/` and rebuild

---

## Support

For issues or questions:

1. Check `IMPLEMENTATION_SUMMARY.md` for detailed documentation
2. Run `python test_enhanced_features.py` to verify setup
3. Review `ANALYSIS_AND_RECOMMENDATIONS.md` for system design

---

## Quick Reference

| Task               | Command                            |
| ------------------ | ---------------------------------- |
| Build KG           | `python build_comprehensive_kg.py` |
| Generate Questions | `python generate_questions.py`     |
| Test Answer Gen    | `python answer_generator.py`       |
| Test All Features  | `python test_enhanced_features.py` |
| Check KG           | `python check_kg_connections.py`   |

---

**That's it! Start with Step 1 (build KG) and proceed to Step 2 (generate questions).** 🚀
