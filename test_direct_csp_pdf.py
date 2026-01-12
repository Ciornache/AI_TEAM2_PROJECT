#!/usr/bin/env python
"""Direct test of CSP PDF generation bypassing AnswerGenerator"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test imports
print("1. Testing imports...")
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    print("   - reportlab OK")
except Exception as e:
    print(f"   - reportlab FAILED: {e}")
    sys.exit(1)

try:
    from src.csp_solver import GraphColoringCSP, NQueensCSP
    print("   - csp_solver OK")
except Exception as e:
    print(f"   - csp_solver FAILED: {e}")
    sys.exit(1)

# Test direct PDF generation without AnswerGenerator
print("\n2. Creating basic PDF with CSP content...")
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    pdf_path = os.path.join(output_dir, "CSP_Direct_Test.pdf")

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Add title
    story.append(Paragraph("CSP Test PDF", styles['Heading1']))
    story.append(Spacer(1, 0.2 * inch))

    # CSP 1: Graph Coloring
    story.append(Paragraph("CSP 1: Graph Coloring", styles['Heading2']))
    story.append(Paragraph("Test problem with graph coloring", styles['Normal']))
    story.append(Spacer(1, 0.1 * inch))

    # Solve
    print("   - Solving Graph Coloring...")
    csp = GraphColoringCSP.create_csp(5, [(0,1), (1,2), (2,3), (3,4), (4,0)], 3)
    solution = csp.solve_backtracking_basic()
    stats = csp.get_stats()

    print(f"   - Solution: {solution}")
    print(f"   - Stats: {stats}")

    result_text = f"Solution: {solution}<br/>Checks: {stats['constraint_checks']}<br/>Backtracks: {stats['backtracks']}"
    story.append(Paragraph(result_text, styles['Normal']))
    story.append(Spacer(1, 0.3 * inch))

    # CSP 2: N-Queens
    story.append(Paragraph("CSP 2: N-Queens", styles['Heading2']))
    story.append(Paragraph("N-Queens with 3 pre-placed", styles['Normal']))
    story.append(Spacer(1, 0.1 * inch))

    print("   - Solving N-Queens...")
    csp2 = NQueensCSP.create_csp(6, {0: 1, 1: 3, 2: 5})
    solution2 = csp2.solve_backtracking_basic()
    stats2 = csp2.get_stats()

    print(f"   - Solution: {solution2}")
    print(f"   - Stats: {stats2}")

    result_text2 = f"Solution: {solution2}<br/>Checks: {stats2['constraint_checks']}<br/>Backtracks: {stats2['backtracks']}"
    story.append(Paragraph(result_text2, styles['Normal']))

    # Build PDF
    print("   - Building PDF...")
    doc.build(story)

    if os.path.exists(pdf_path):
        size = os.path.getsize(pdf_path)
        print(f"\n3. SUCCESS! PDF created: {pdf_path} ({size} bytes)")
    else:
        print(f"\n3. FAILED: PDF not created at {pdf_path}")
        sys.exit(1)

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

