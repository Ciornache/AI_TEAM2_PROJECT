import sys
import sys
import os
import json
from datetime import datetime
from flask import Flask, render_template, request, send_file, jsonify

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pdf_generator import PDFQuestionGenerator
from src.csp_solver import GraphColoringCSP, NQueensCSP

app = Flask(__name__)

# Available problems
PROBLEMS = [
    'N-Queens', 
    'Tower of Hanoi', 
    'Graph Coloring', 
    'Knight\'s Tour', 
    '8-Puzzle', 
    'MinMax',
    'CSP - Graph Coloring & N-Queens'
]

@app.route('/')
def index():
    return render_template('index.html', problems=PROBLEMS)

@app.route('/csp')
def csp_solver():
    """Serve the CSP solver interface"""
    return render_template('csp_solver.html')

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

@app.route('/solve-csp', methods=['POST'])
def solve_csp():
    """
    Endpoint to solve CSP problems with different optimization strategies.

    Request JSON format:
    {
        "problem_type": "Graph Coloring" or "N-Queens",
        "parameters": {
            "n_vertices": 5,      // for Graph Coloring
            "n_colors": 3,        // for Graph Coloring
            "edges": [[0,1], [1,2], ...],  // for Graph Coloring
            "n": 6,               // for N-Queens
            "n_prime": 2,         // for N-Queens (pre-placed)
            "placed_queens": {0: 1, 1: 3}  // for N-Queens
        },
        "strategies": ["backtracking", "fc", "mrv", "ac3", "fc_mrv"]  // which to run
    }
    """
    try:
        data = request.json
        problem_type = data.get('problem_type', '').lower()
        parameters = data.get('parameters', {})
        strategies = data.get('strategies', ['backtracking', 'fc', 'mrv', 'ac3', 'fc_mrv'])

        if not problem_type:
            return jsonify({'error': 'Problem type not specified'}), 400

        results = {
            'problem_type': problem_type,
            'parameters': parameters,
            'results': {},
            'comparison': {}
        }

        # Solve Graph Coloring CSP
        if 'coloring' in problem_type or 'graph' in problem_type:
            n_vertices = parameters.get('n_vertices', 5)
            n_colors = parameters.get('n_colors', 3)
            edges = parameters.get('edges', [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])

            strategy_map = {
                'backtracking': 'solve_backtracking_basic',
                'fc': 'solve_with_fc',
                'mrv': 'solve_with_mrv',
                'ac3': 'solve_with_ac3',
                'fc_mrv': 'solve_with_fc_and_mrv'
            }

            for strategy in strategies:
                if strategy not in strategy_map:
                    continue

                csp = GraphColoringCSP.create_csp(n_vertices, edges, n_colors)
                solve_method = getattr(csp, strategy_map[strategy])
                solution = solve_method()
                stats = csp.get_stats()

                results['results'][strategy] = {
                    'solution': solution,
                    'constraint_checks': stats['constraint_checks'],
                    'backtracks': stats['backtracks'],
                    'valid': stats['valid']
                }

                results['comparison'][strategy] = {
                    'checks': stats['constraint_checks'],
                    'backtracks': stats['backtracks']
                }

        # Solve N-Queens CSP
        elif 'queens' in problem_type or 'n-queens' in problem_type:
            n = parameters.get('n', 6)
            placed_queens = parameters.get('placed_queens', {})

            strategy_map = {
                'backtracking': 'solve_backtracking_basic',
                'fc': 'solve_with_fc',
                'mrv': 'solve_with_mrv',
                'ac3': 'solve_with_ac3',
                'fc_mrv': 'solve_with_fc_and_mrv'
            }

            for strategy in strategies:
                if strategy not in strategy_map:
                    continue

                csp = NQueensCSP.create_csp(n, placed_queens)
                solve_method = getattr(csp, strategy_map[strategy])
                solution = solve_method()
                stats = csp.get_stats()

                results['results'][strategy] = {
                    'solution': solution,
                    'constraint_checks': stats['constraint_checks'],
                    'backtracks': stats['backtracks'],
                    'valid': stats['valid']
                }

                results['comparison'][strategy] = {
                    'checks': stats['constraint_checks'],
                    'backtracks': stats['backtracks']
                }

        else:
            return jsonify({'error': f'Unknown problem type: {problem_type}'}), 400

        # Find best strategy
        if results['comparison']:
            best_strategy = min(results['comparison'].items(), key=lambda x: x[1]['checks'])
            results['best_strategy'] = {
                'name': best_strategy[0],
                'checks': best_strategy[1]['checks'],
                'backtracks': best_strategy[1]['backtracks']
            }

        return jsonify(results), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/csp-info', methods=['GET'])
def csp_info():
    """
    Get information about available CSP solving strategies.
    """
    strategies = {
        'backtracking': {
            'name': 'Basic Backtracking',
            'description': 'Sequential assignment with constraint checking',
            'complexity_time': 'O(d^n)',
            'complexity_space': 'O(n)',
            'best_for': 'Small instances (n < 5)',
            'pros': ['Simple', 'Low overhead', 'Guaranteed correct'],
            'cons': ['Explores many dead ends', 'Slow on large problems']
        },
        'fc': {
            'name': 'Forward Checking (FC)',
            'description': 'Backtracking + eliminate inconsistent values from neighbors',
            'complexity_time': 'O(e*d²) per node',
            'complexity_space': 'O(n*d)',
            'best_for': 'Medium instances (n ~ 5-8)',
            'pros': ['Detects failures early', 'Reduces search space'],
            'cons': ['Domain maintenance overhead']
        },
        'mrv': {
            'name': 'Minimum Remaining Values (MRV)',
            'description': 'Choose variable with smallest domain at each step',
            'complexity_time': '+O(n) per step',
            'complexity_space': 'O(n)',
            'best_for': 'Variable branching factor',
            'pros': ['Reduces branching factor', 'Good failure detection'],
            'cons': ['Overhead to find minimum domain']
        },
        'ac3': {
            'name': 'Arc Consistency 3 (AC-3)',
            'description': 'Remove values without support in connected variables',
            'complexity_time': 'O(e*d³)',
            'complexity_space': 'O(e)',
            'best_for': 'Dense graphs (n > 10)',
            'pros': ['Powerful propagation', 'Strong pruning'],
            'cons': ['Expensive for small instances']
        },
        'fc_mrv': {
            'name': 'FC + MRV Combined',
            'description': 'MRV for selection + FC for propagation',
            'complexity_time': 'O(e*d²) + O(n)',
            'complexity_space': 'O(n*d)',
            'best_for': 'General/default use',
            'pros': ['Optimal combination', 'Best overall performance'],
            'cons': ['Combined overhead']
        }
    }

    return jsonify(strategies), 200

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=False, port=5000)
