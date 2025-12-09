import sys
import os
import json
from datetime import datetime
from flask import Flask, render_template, request, send_file, jsonify

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pdf_generator import PDFQuestionGenerator

app = Flask(__name__)

# Available problems
PROBLEMS = [
    'N-Queens', 
    'Tower of Hanoi', 
    'Graph Coloring', 
    'Knight\'s Tour', 
    '8-Puzzle', 
    'MinMax'
]

@app.route('/')
def index():
    return render_template('index.html', problems=PROBLEMS)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        problem_config = data.get('config', [])
        
        if not problem_config:
            return jsonify({'error': 'No configuration provided'}), 400
            
        # Path to KG in data/ folder
        # Assuming we are running from project root or web_interface folder
        # We need to find the data folder relative to this script
        base_dir = os.path.dirname(os.path.dirname(__file__))
        kg_path = os.path.join(base_dir, "data", "knowledge_graph.json")
        
        if not os.path.exists(kg_path):
             return jsonify({'error': f'Knowledge Graph not found at {kg_path}'}), 500

        generator = PDFQuestionGenerator(kg_path)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate PDF with answers
        filename = f'Custom_Test_{timestamp}.pdf'
        output_path = os.path.join(output_dir, filename)
        
        generator.generate_pdf(output_path, problem_config=problem_config, include_answers=True)
        
        # Also generate worksheet (optional, but good to have)
        worksheet_filename = f'Custom_Worksheet_{timestamp}.pdf'
        worksheet_path = os.path.join(output_dir, worksheet_filename)
        generator.generate_pdf(worksheet_path, problem_config=problem_config, include_answers=False)
        
        return jsonify({
            'success': True, 
            'message': 'PDFs generated successfully',
            'files': {
                'test': filename,
                'worksheet': worksheet_filename
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download(filename):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(base_dir, "output")
    return send_file(os.path.join(output_dir, filename), as_attachment=True)

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=False, port=5000)
