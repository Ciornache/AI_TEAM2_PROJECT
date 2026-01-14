#!/usr/bin/env python
"""
Flask Web Server Starter
Run this file to start the web interface on http://localhost:5000
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Change to web_interface directory
os.chdir(os.path.join(os.path.dirname(__file__), 'web_interface'))

# Now import and run Flask
from app import app

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 AI TEST GENERATOR - WEB SERVER")
    print("=" * 70)
    print("\n✅ Server starting...")
    print("📱 Access the application at: http://localhost:5000")
    print("\n🔗 Available Pages:")
    print("   - Main Generator:   http://localhost:5000/")
    print("   - CSP Solver:       http://localhost:5000/csp")
    print("   - Answer Validator: http://localhost:5000/validator")
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("=" * 70 + "\n")

    app.run(debug=True, host='127.0.0.1', port=5000)

