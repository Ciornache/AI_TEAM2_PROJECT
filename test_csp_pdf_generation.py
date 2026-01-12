#!/usr/bin/env python
"""
Test CSP PDF Generation
======================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pdf_generator import PDFQuestionGenerator

def test_csp_pdf():
    """Test CSP PDF generation."""
    print("="*80)
    print("Testing CSP PDF Generation")
    print("="*80)

    try:
        # Initialize PDF generator
        kg_path = os.path.join(os.path.dirname(__file__), "data", "knowledge_graph.json")

        if not os.path.exists(kg_path):
            print(f"⚠ Warning: Knowledge graph not found at {kg_path}")
            print("  Creating test PDF anyway with limited features...")
            # Create dummy generator
            class DummyAnswerGen:
                def __init__(self, path):
                    pass
            PDFQuestionGenerator.answer_gen = DummyAnswerGen(kg_path)

        gen = PDFQuestionGenerator(kg_path)
        print("✓ PDF Generator initialized")

        # Test CSP PDF generation with answers
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)

        # Generate PDF with CSP
        pdf_path = os.path.join(output_dir, "CSP_Test_With_Answers.pdf")

        print(f"\nGenerating CSP PDF with answers...")
        problem_config = [
            {'name': 'CSP', 'count': 1}
        ]

        gen.generate_pdf(pdf_path, problem_config=problem_config, include_answers=True)

        if os.path.exists(pdf_path):
            file_size = os.path.getsize(pdf_path)
            print(f"✓ PDF generated successfully: {pdf_path}")
            print(f"  File size: {file_size} bytes")
        else:
            print(f"✗ PDF generation failed")
            return False

        # Generate worksheet (no answers)
        worksheet_path = os.path.join(output_dir, "CSP_Test_Worksheet.pdf")

        print(f"\nGenerating CSP worksheet (no answers)...")
        gen.generate_pdf(worksheet_path, problem_config=problem_config, include_answers=False)

        if os.path.exists(worksheet_path):
            file_size = os.path.getsize(worksheet_path)
            print(f"✓ Worksheet generated successfully: {worksheet_path}")
            print(f"  File size: {file_size} bytes")
        else:
            print(f"✗ Worksheet generation failed")
            return False

        # Generate combined test with all problems
        print(f"\nGenerating combined test with CSP + other problems...")
        combined_path = os.path.join(output_dir, "CSP_Combined_Test.pdf")

        combined_config = [
            {'name': 'N-Queens', 'count': 1},
            {'name': 'CSP', 'count': 1},
            {'name': 'MinMax', 'count': 1}
        ]

        gen.generate_pdf(combined_path, problem_config=combined_config, include_answers=True)

        if os.path.exists(combined_path):
            file_size = os.path.getsize(combined_path)
            print(f"✓ Combined PDF generated: {combined_path}")
            print(f"  File size: {file_size} bytes")
        else:
            print(f"✗ Combined PDF generation failed")
            return False

        print("\n" + "="*80)
        print("✅ CSP PDF Generation Test Successful!")
        print("="*80)
        print("\nGenerated PDFs:")
        print(f"  1. {pdf_path}")
        print(f"  2. {worksheet_path}")
        print(f"  3. {combined_path}")
        print("\nYou can now use the web interface to generate CSP PDFs!")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_csp_pdf()
    sys.exit(0 if success else 1)

