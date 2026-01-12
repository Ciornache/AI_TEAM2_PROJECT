#!/usr/bin/env python
"""Ultra simple PDF test"""

import sys
import os

print("1. Import reportlab...")
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    print("   OK")
except Exception as e:
    print(f"   FAILED: {e}")
    sys.exit(1)

print("2. Create output dir...")
output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)
print(f"   OK: {output_dir}")

print("3. Create PDF...")
pdf_path = os.path.join(output_dir, "SIMPLE_TEST.pdf")
print(f"   Path: {pdf_path}")

try:
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    print("   Document created")

    styles = getSampleStyleSheet()
    print("   Styles loaded")

    story = []
    story.append(Paragraph("Test", styles['Heading1']))
    print("   Content added")

    print("   Building PDF...")
    doc.build(story)
    print("   PDF built")

    if os.path.exists(pdf_path):
        size = os.path.getsize(pdf_path)
        print(f"\nSUCCESS: {pdf_path} ({size} bytes)")
    else:
        print(f"\nFAILED: File not created")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

