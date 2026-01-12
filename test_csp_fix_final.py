#!/usr/bin/env python
"""Test that CSP PDF generation now works correctly"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pdf_generator import PDFQuestionGenerator

print("Testing CSP PDF Generation (Fixed)")
print("=" * 60)

try:
    kg_path = os.path.join(os.path.dirname(__file__), "data", "knowledge_graph.json")

    gen = PDFQuestionGenerator(kg_path)
    print("1. PDF Generator initialized - OK")

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Test 1: Using the full name from web interface
    pdf_path = os.path.join(output_dir, "CSP_TEST_FULL_NAME.pdf")
    print(f"2. Generating using full name: 'CSP - Graph Coloring & N-Queens'")

    gen.generate_pdf(
        pdf_path,
        problem_config=[{'name': 'CSP - Graph Coloring & N-Queens', 'count': 1}],
        include_answers=True
    )

    if os.path.exists(pdf_path):
        size = os.path.getsize(pdf_path)
        print(f"   SUCCESS - File created: {size} bytes")
        if size > 5000:
            print(f"   File size is large enough to contain CSP content!")
        else:
            print(f"   WARNING: File size might be too small")
    else:
        print("   FAILED - File not created")
        sys.exit(1)

    # Test 2: Using short name
    pdf_path2 = os.path.join(output_dir, "CSP_TEST_SHORT_NAME.pdf")
    print(f"3. Generating using short name: 'CSP'")

    gen.generate_pdf(
        pdf_path2,
        problem_config=[{'name': 'CSP', 'count': 1}],
        include_answers=True
    )

    if os.path.exists(pdf_path2):
        size = os.path.getsize(pdf_path2)
        print(f"   SUCCESS - File created: {size} bytes")
        if size > 5000:
            print(f"   File size is large enough to contain CSP content!")
    else:
        print("   FAILED - File not created")
        sys.exit(1)

    # Test 3: Worksheet format
    pdf_path3 = os.path.join(output_dir, "CSP_TEST_WORKSHEET.pdf")
    print(f"4. Generating worksheet (no answers)")

    gen.generate_pdf(
        pdf_path3,
        problem_config=[{'name': 'CSP - Graph Coloring & N-Queens', 'count': 1}],
        include_answers=False
    )

    if os.path.exists(pdf_path3):
        size = os.path.getsize(pdf_path3)
        print(f"   SUCCESS - File created: {size} bytes")
    else:
        print("   FAILED - File not created")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("\nGenerated PDFs:")
    print(f"  1. {pdf_path}")
    print(f"  2. {pdf_path2}")
    print(f"  3. {pdf_path3}")
    print("\nCSP PDFs should now include:")
    print("  - Graph Coloring problem")
    print("  - N-Queens problem")
    print("  - Strategy comparison")
    print("  - Questions and answers")
    print("  - Performance metrics")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

