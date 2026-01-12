#!/usr/bin/env python
"""Test CSP PDF generation - Simple version"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pdf_generator import PDFQuestionGenerator

print("Testing CSP PDF Generation...")
print("=" * 60)

try:
    kg_path = os.path.join(os.path.dirname(__file__), "data", "knowledge_graph.json")
    print(f"KG Path: {kg_path}")
    print(f"KG exists: {os.path.exists(kg_path)}")

    print("Creating PDF Generator...")
    gen = PDFQuestionGenerator(kg_path)
    print("PDF Generator initialized")

    # Test 1: CSP Only
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir: {output_dir}")

    pdf_path = os.path.join(output_dir, "CSP_Test_With_Answers.pdf")
    print(f"Generating: {pdf_path}")
    print("Calling generate_pdf...")
    sys.stdout.flush()

    gen.generate_pdf(
        pdf_path,
        problem_config=[{'name': 'CSP', 'count': 1}],
        include_answers=True
    )
    print("generate_pdf completed")
    sys.stdout.flush()

    if os.path.exists(pdf_path):
        size = os.path.getsize(pdf_path)
        print(f"SUCCESS: Generated {size} bytes")
    else:
        print("FAILED: PDF not created")
        sys.exit(1)

    # Test 2: Worksheet
    worksheet_path = os.path.join(output_dir, "CSP_Test_Worksheet.pdf")
    print(f"\nGenerating: {worksheet_path}")

    gen.generate_pdf(
        worksheet_path,
        problem_config=[{'name': 'CSP', 'count': 1}],
        include_answers=False
    )

    if os.path.exists(worksheet_path):
        size = os.path.getsize(worksheet_path)
        print(f"SUCCESS: Generated {size} bytes")
    else:
        print("FAILED: Worksheet not created")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All tests passed!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

