"""
Generate Questions - Simple Runner
===================================
Quick script to generate AI search strategy questions PDF.
"""

import sys
import random
from pdf_generator import PDFQuestionGenerator


def main():
    """Generate questions PDF with configurable instances."""
    print("=" * 80)
    print("AI SEARCH STRATEGY QUESTION GENERATOR")
    print("=" * 80)
    print()
    
    # Get number of instances from command line or use random
    if len(sys.argv) > 1:
        try:
            n_instances = int(sys.argv[1])
            if n_instances < 1 or n_instances > 3:
                print("⚠ Number of instances must be between 1 and 3")
                n_instances = random.randint(1, 3)
        except ValueError:
            print("⚠ Invalid number, using random")
            n_instances = random.randint(1, 3)
    else:
        n_instances = random.randint(1, 3)
    
    print(f"Configuration:")
    print(f"  - Problems: 5 (N-Queens, Hanoi, Graph Coloring, Knight's Tour, 8-Puzzle)")
    print(f"  - Instances per problem: {n_instances}")
    print(f"  - Total questions: {5 * n_instances}")
    print()
    
    # Generate PDF
    print("Generating PDF...")
    generator = PDFQuestionGenerator('knowledge_graph.json')
    
    # Use timestamp to avoid conflicts
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'AI_Search_Questions_{timestamp}.pdf'
    generator.generate_pdf(output_path, n_instances_per_problem=n_instances)
    
    print()
    print("=" * 80)
    print("✓ SUCCESS!")
    print("=" * 80)
    print(f"PDF generated: {output_path}")
    print(f"Total questions with answers: {5 * n_instances}")
    print()
    print("Each question includes:")
    print("  ✓ Problem instance visualization")
    print("  ✓ Detailed question about best solving strategy")
    print("  ✓ Knowledge graph-based answer with reasoning")
    print("  ✓ Complexity analysis")
    print("  ✓ Alternative strategies")
    print("  ✓ Recommended heuristics")
    print()


if __name__ == "__main__":
    main()
