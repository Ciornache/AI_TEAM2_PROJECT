"""
Generate Questions - Simple Runner
===================================
Quick script to generate AI search strategy questions PDF.
"""

import sys
import random
from pdf_generator import PDFQuestionGenerator


def main():
    """Generate questions PDF for ONE problem with 1-3 instances."""
    print("=" * 80)
    print("AI SEARCH STRATEGY QUESTION GENERATOR")
    print("=" * 80)
    print()
    
    # Available problems
    available_problems = ['N-Queens', 'Tower of Hanoi', 'Graph Coloring', 'Knight\'s Tour', '8-Puzzle']
    
    # Select one problem randomly
    selected_problem = random.choice(available_problems)
    
    # Get number of instances from command line or use random (1-3)
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
    print(f"  - Selected Problem: {selected_problem}")
    print(f"  - Number of instances: {n_instances}")
    print(f"  - Total questions: {n_instances}")
    print()
    
    # Generate PDF
    print("Generating PDF...")
    generator = PDFQuestionGenerator('knowledge_graph.json')
    
    # Use timestamp to avoid conflicts
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Fix f-string: avoid backslash in expression by pre-processing
    problem_name_clean = selected_problem.replace(" ", "_").replace("'", "")
    output_path = f'AI_Search_Questions_{problem_name_clean}_{timestamp}.pdf'
    generator.generate_pdf_single_problem(output_path, selected_problem, n_instances)
    
    print()
    print("=" * 80)
    print("✓ SUCCESS!")
    print("=" * 80)
    print(f"PDF generated: {output_path}")
    print(f"Problem: {selected_problem}")
    print(f"Total questions: {n_instances}")
    print()
    print("Each question includes:")
    print("  ✓ Problem instance visualization")
    print("  ✓ Detailed question about best solving strategy")
    print("  ✓ Knowledge graph-based answer with reasoning (NO hardcoded results)")
    print("  ✓ Complexity analysis")
    print("  ✓ Alternative strategies")
    print("  ✓ Recommended heuristics")
    print()


if __name__ == "__main__":
    main()
