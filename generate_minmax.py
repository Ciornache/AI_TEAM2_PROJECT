from pdf_generator import PDFQuestionGenerator

def main():
    # Creează generatorul PDF
    generator = PDFQuestionGenerator('knowledge_graph.json')
    
    # Generează PDF pentru problema MinMax, o singură instanță
    generator.generate_pdf_single_problem(
        output_path='AI_Search_Questions_MinMax.pdf',
        problem_name='MinMax',
        n_instances=1  # <- doar o instanță
    )

if __name__ == "__main__":
    main()
