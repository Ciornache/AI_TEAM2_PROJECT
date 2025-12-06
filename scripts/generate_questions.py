"""
Generate Questions - Simple Runner
===================================
Quick script to generate AI search strategy questions PDF.
"""

import sys
import os
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pdf_generator import PDFQuestionGenerator


def main():
    """Generate questions PDF for ALL problems with 1-3 instances."""
    print("=" * 80)
    print("AI SEARCH STRATEGY QUESTION GENERATOR (COMBINED)")
    print("=" * 80)
    print()
    
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
    print(f"  - Mode: Combined (All Problems)")
    print(f"  - Number of instances per problem: {n_instances}")
    print()
    
    # Generate PDF
    print("Generating PDF...")
    
    # Path to KG in data/ folder
    env_data_dir = os.getenv("DATA_DIR", "data")
    kg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), env_data_dir, "knowledge_graph.json")
    generator = PDFQuestionGenerator(kg_path)
    
    # Use timestamp to avoid conflicts
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Output to output/ folder
    env_output_dir = os.getenv("OUTPUT_DIR", "output")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), env_output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate PDF WITH answers
    output_path_answers = os.path.join(output_dir, f'AI_Search_Questions_With_Answers_{timestamp}.pdf')
    print(f"Generating PDF with answers: {output_path_answers}")
    generator.generate_pdf(output_path_answers, n_instances, include_answers=True)
    
    # 2. Generate PDF WITHOUT answers (Worksheet)
    output_path_worksheet = os.path.join(output_dir, f'AI_Search_Questions_Worksheet_{timestamp}.pdf')
    print(f"Generating Worksheet PDF: {output_path_worksheet}")
    generator.generate_pdf(output_path_worksheet, n_instances, include_answers=False)
    
    print()
    print("=" * 80)
    print("✓ SUCCESS!")
    print("=" * 80)
    print(f"1. With Answers: {output_path_answers}")
    print(f"2. Worksheet:    {output_path_worksheet}")
    print(f"Total questions: {6 * n_instances} (6 problems × {n_instances} instances)")
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
