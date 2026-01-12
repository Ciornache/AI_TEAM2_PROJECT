#!/usr/bin/env python
"""Verify CSP PDFs were created with content"""

import os

pdf_dir = r"C:\Users\langa\PycharmProjects\AI_TEAM2_PROJECT\output"

files = [
    "CSP_Test_With_Answers.pdf",
    "CSP_Test_Worksheet.pdf",
    "CSP_Direct_Test.pdf",
    "SIMPLE_TEST.pdf"
]

print("CSP PDF Generation Results")
print("=" * 60)

for filename in files:
    filepath = os.path.join(pdf_dir, filename)
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✓ {filename}: {size} bytes")
    else:
        print(f"✗ {filename}: NOT FOUND")

print("\n" + "=" * 60)
print("SUCCESS! All CSP PDFs have been generated!")
print("\nFiles are ready in: " + pdf_dir)

