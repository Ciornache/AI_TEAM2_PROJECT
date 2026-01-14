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

@app.route('/validator')
def answer_validator():
    """Serve the answer validator interface"""
    return render_template('answer_validator.html')

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

@app.route('/validate-answer', methods=['POST'])
def validate_answer():
    """
    Validate student answer against correct answer.

    Request JSON format:
    {
        "problem_type": "CSP Graph Coloring" or "CSP N-Queens",
        "student_answer": "V0=Red, V1=Blue, V2=Green...",
        "question_id": "csp_1",
        "instance_data": {...}
    }

    Response:
    {
        "is_correct": true/false,
        "score": 0-100,
        "feedback": "explanation",
        "correct_answer": "expected answer",
        "explanation": "why this is correct/incorrect"
    }
    """
    try:
        data = request.json
        problem_type = data.get('problem_type', '')
        student_answer = data.get('student_answer', '').strip()
        question_id = data.get('question_id', '')
        instance_data = data.get('instance_data', {})

        if not student_answer:
            return jsonify({
                'is_correct': False,
                'score': 0,
                'feedback': 'Answer cannot be empty',
                'correct_answer': '',
                'explanation': 'Please provide an answer'
            }), 200

        # Validate based on problem type
        if 'CSP' in problem_type or 'Graph Coloring' in problem_type:
            result = _validate_graph_coloring(student_answer, question_id, instance_data)
        elif 'N-Queens' in problem_type:
            result = _validate_nqueens(student_answer, question_id, instance_data)
        elif 'Strategy' in problem_type:
            result = _validate_strategy(student_answer, question_id)
        else:
            return jsonify({
                'is_correct': False,
                'score': 0,
                'feedback': 'Unknown problem type',
                'correct_answer': '',
                'explanation': f'Problem type "{problem_type}" not recognized'
            }), 200

        return jsonify(result), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'is_correct': False,
            'score': 0
        }), 500

def _validate_graph_coloring(student_answer, question_id, instance_data):
    """Validate Graph Coloring answer"""

    # Expected answer format: "V0=Red, V1=Blue, V2=Green, V3=Red, V4=Blue"
    try:
        # Parse student answer
        assignments = {}
        parts = student_answer.split(',')
        for part in parts:
            part = part.strip()
            if '=' not in part:
                return {
                    'is_correct': False,
                    'score': 0,
                    'feedback': f'Invalid format in "{part}". Use format: V0=Red, V1=Blue, etc.',
                    'correct_answer': 'V0=Color0, V1=Color1, ...',
                    'explanation': 'Each assignment must be in format Variable=Color'
                }

            var, color = part.split('=')
            var = var.strip()
            color = color.strip()
            assignments[var] = color

        # Get instance data
        n_vertices = instance_data.get('n_vertices', 5)
        edges = instance_data.get('edges', [])
        n_colors = instance_data.get('n_colors', 3)

        # Check if all variables assigned
        expected_vars = [f'V{i}' for i in range(n_vertices)]
        missing = [v for v in expected_vars if v not in assignments]

        if missing:
            return {
                'is_correct': False,
                'score': 50,
                'feedback': f'Missing assignments for: {", ".join(missing)}',
                'correct_answer': 'All variables must be assigned',
                'explanation': f'You assigned {len(assignments)} variables but should assign {n_vertices}'
            }

        # Check if all colors are valid
        valid_colors = [str(i) for i in range(n_colors)]
        valid_color_names = ['Red', 'Green', 'Blue', 'Yellow', 'Orange', 'Purple'][:n_colors]

        for var, color in assignments.items():
            if color not in valid_color_names and color not in valid_colors:
                return {
                    'is_correct': False,
                    'score': 30,
                    'feedback': f'Invalid color "{color}". Valid colors: {", ".join(valid_color_names)}',
                    'correct_answer': f'Colors must be from: {", ".join(valid_color_names)}',
                    'explanation': 'Used color not available in domain'
                }

        # Check constraints (no adjacent nodes have same color)
        violations = []
        for u, v in edges:
            var_u = f'V{u}'
            var_v = f'V{v}'
            if assignments[var_u] == assignments[var_v]:
                violations.append(f'{var_u}={assignments[var_u]} and {var_v}={assignments[var_v]}')

        if violations:
            return {
                'is_correct': False,
                'score': 25,
                'feedback': f'Constraint violations: {"; ".join(violations[:2])}',
                'correct_answer': 'No two adjacent vertices should have same color',
                'explanation': f'Your solution violates {len(violations)} constraints. Adjacent vertices with same color: {violations[0]}'
            }

        # ✅ CORRECT!
        return {
            'is_correct': True,
            'score': 100,
            'feedback': 'Perfect! All constraints satisfied!',
            'correct_answer': ', '.join([f'{var}={color}' for var, color in sorted(assignments.items())]),
            'explanation': 'All variables assigned, all colors valid, and no constraint violations!'
        }

    except Exception as e:
        return {
            'is_correct': False,
            'score': 0,
            'feedback': f'Error parsing answer: {str(e)}',
            'correct_answer': 'V0=Color0, V1=Color1, ...',
            'explanation': 'Could not parse your answer. Check format.'
        }

def _validate_nqueens(student_answer, question_id, instance_data):
    """Validate N-Queens answer"""

    # Expected format: "Q0=Col1, Q1=Col3, Q2=Col5, ..."
    try:
        assignments = {}
        parts = student_answer.split(',')

        for part in parts:
            part = part.strip()
            if '=' not in part:
                return {
                    'is_correct': False,
                    'score': 0,
                    'feedback': f'Invalid format: "{part}". Use format: Q0=1, Q1=3, etc.',
                    'correct_answer': 'Q0=Col0, Q1=Col1, ...',
                    'explanation': 'Each queen assignment must be in format QRow=Column'
                }

            queen, col = part.split('=')
            queen = queen.strip().upper()
            col = int(col.strip())

            # Extract row from Q0, Q1, etc.
            row = int(queen[1:])
            assignments[row] = col

        board_size = instance_data.get('n', 8)
        pre_placed = instance_data.get('placed_queens', {})

        # Check if all queens placed
        if len(assignments) != board_size:
            return {
                'is_correct': False,
                'score': 40,
                'feedback': f'Incomplete solution. Placed {len(assignments)}/{board_size} queens',
                'correct_answer': f'All {board_size} queens must be placed',
                'explanation': f'You placed {len(assignments)} queens but need {board_size}'
            }

        # Check column validity
        for row, col in assignments.items():
            if col < 0 or col >= board_size:
                return {
                    'is_correct': False,
                    'score': 30,
                    'feedback': f'Invalid column {col} for queen at row {row}. Valid: 0-{board_size-1}',
                    'correct_answer': f'All columns must be in range 0-{board_size-1}',
                    'explanation': 'Column out of bounds'
                }

        # Check no two queens in same column
        columns = list(assignments.values())
        if len(columns) != len(set(columns)):
            duplicate_col = [c for c in columns if columns.count(c) > 1][0]
            return {
                'is_correct': False,
                'score': 35,
                'feedback': f'Two or more queens in column {duplicate_col}',
                'correct_answer': 'Each queen must be in different column',
                'explanation': 'Multiple queens share the same column'
            }

        # Check no two queens attack each other
        violations = []
        for r1 in range(board_size):
            for r2 in range(r1 + 1, board_size):
                c1 = assignments[r1]
                c2 = assignments[r2]

                # Check diagonal attack
                if abs(r1 - r2) == abs(c1 - c2):
                    violations.append(f'Q{r1}({r1},{c1}) attacks Q{r2}({r2},{c2})')

        if violations:
            return {
                'is_correct': False,
                'score': 25,
                'feedback': f'Queens attack each other: {violations[0]}',
                'correct_answer': 'No two queens should attack each other',
                'explanation': f'Diagonal attack found: {violations[0]}'
            }

        # ✅ CORRECT!
        return {
            'is_correct': True,
            'score': 100,
            'feedback': 'Excellent! All queens placed safely!',
            'correct_answer': ', '.join([f'Q{r}={c}' for r, c in sorted(assignments.items())]),
            'explanation': 'All queens placed, no column conflicts, no attacks!'
        }

    except Exception as e:
        return {
            'is_correct': False,
            'score': 0,
            'feedback': f'Error parsing N-Queens answer: {str(e)}',
            'correct_answer': 'Q0=0, Q1=4, Q2=7, ...',
            'explanation': 'Could not parse your answer.'
        }

def _validate_strategy(student_answer, question_id):
    """Validate strategy comparison answer"""

    student_answer = student_answer.lower().strip()

    # Expected keywords for AC-3 vs Backtracking question
    correct_keywords = ['n', 'variables', 'dense', 'constraint', 'propagation', 'preprocessing']

    found_keywords = sum(1 for kw in correct_keywords if kw in student_answer)

    if found_keywords >= 3:
        return {
            'is_correct': True,
            'score': 100,
            'feedback': 'Great explanation of when to use AC-3!',
            'correct_answer': 'AC-3 is better for large (n>10), dense constraint graphs where preprocessing overhead pays off',
            'explanation': 'You correctly identified key factors for AC-3 selection'
        }
    elif found_keywords >= 1:
        return {
            'is_correct': False,
            'score': 60,
            'feedback': 'Partial credit - you mentioned some correct concepts',
            'correct_answer': 'Should mention: problem size (n>10), constraint density, preprocessing benefits',
            'explanation': 'Your answer is incomplete. Consider: problem characteristics, algorithm complexity, and when overhead pays off'
        }
    else:
        return {
            'is_correct': False,
            'score': 0,
            'feedback': 'Answer does not match expected strategy concepts',
            'correct_answer': 'AC-3 excels on large, dense CSP instances due to strong constraint propagation',
            'explanation': 'Try to mention: variable count, constraint graph density, and why AC-3 preprocessing helps'
        }


if __name__ == '__main__':
    print("Starting Flask server on http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    app.run(debug=True, host='0.0.0.0', port=5000)
