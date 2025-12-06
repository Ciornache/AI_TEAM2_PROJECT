import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pdf_generator import PDFQuestionGenerator

def main():
    # Path to KG in data/ folder
    kg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "knowledge_graph.json")
    
    # Creează generatorul PDF
    generator = PDFQuestionGenerator(kg_path)
    
    # Output to output/ folder
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'AI_Search_Questions_MinMax.pdf')
    
    # Generează PDF pentru problema MinMax, o singură instanță
    generator.generate_pdf_single_problem(
        output_path=output_path,
        problem_name='MinMax',
        n_instances=1  # <- doar o instanță
    )

if __name__ == "__main__":
    main()
