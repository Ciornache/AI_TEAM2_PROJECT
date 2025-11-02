"""
PDF Question Generator
======================
Generates PDF with problem instances, questions, and answers.
"""

import random
from typing import List, Dict
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

from problem_generators import (
    NQueensGenerator, HanoiGenerator, GraphColoringGenerator,
    KnightTourGenerator, Puzzle8Generator, ProblemInstance
)
from answer_generator import AnswerGenerator


class PDFQuestionGenerator:
    """Generate PDF with questions and answers for AI search problems."""
    
    def __init__(self, knowledge_graph_path: str):
        self.answer_gen = AnswerGenerator(knowledge_graph_path)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Problem title
        self.styles.add(ParagraphStyle(
            name='ProblemTitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        ))
        
        # Instance title
        self.styles.add(ParagraphStyle(
            name='InstanceTitle',
            parent=self.styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=10,
            fontName='Helvetica-Bold'
        ))
        
        # Question style
        self.styles.add(ParagraphStyle(
            name='Question',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=15,
            fontName='Helvetica-Bold',
            leftIndent=20
        ))
        
        # Answer heading
        self.styles.add(ParagraphStyle(
            name='AnswerHeading',
            parent=self.styles['Heading4'],
            fontSize=11,
            textColor=colors.HexColor('#16a085'),
            spaceAfter=8,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        ))
        
        # Answer body
        self.styles.add(ParagraphStyle(
            name='AnswerBody',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            leftIndent=20
        ))
    
    def generate_pdf(self, output_path: str, n_instances_per_problem: int = 2):
        """Generate complete PDF with questions and answers."""
        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Container for all content
        story = []
        
        # Title page
        story.append(Paragraph("AI Search Strategy Questions", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(
            "Generated from Knowledge Graph Analysis",
            self.styles['Normal']
        ))
        story.append(Spacer(1, 0.5 * inch))
        
        # Introduction
        intro_text = """
        This document contains problem instances for various AI search problems and questions 
        about the most appropriate solving strategies. Each problem includes instance 
        visualizations and detailed answers based on knowledge graph analysis.
        """
        story.append(Paragraph(intro_text, self.styles['Normal']))
        story.append(PageBreak())
        
        # Generate instances for each problem
        problems = [
            ('N-Queens', self._generate_n_queens_instances, n_instances_per_problem),
            ('Tower of Hanoi', self._generate_hanoi_instances, n_instances_per_problem),
            ('Graph Coloring', self._generate_graph_coloring_instances, n_instances_per_problem),
            ('Knight\'s Tour', self._generate_knight_tour_instances, n_instances_per_problem),
            ('8-Puzzle', self._generate_8puzzle_instances, n_instances_per_problem)
        ]
        
        for i, (problem_name, generator_func, n_instances) in enumerate(problems, 1):
            # Problem section
            story.append(Paragraph(f"{i}. {problem_name}", self.styles['ProblemTitle']))
            story.append(Spacer(1, 0.2 * inch))
            
            # Generate instances
            instances = generator_func(n_instances)
            
            for j, instance in enumerate(instances, 1):
                # Instance heading
                story.append(Paragraph(f"Instance {j}:", self.styles['InstanceTitle']))
                
                # Visualize instance
                visualization = self._visualize_instance(instance)
                story.extend(visualization)
                story.append(Spacer(1, 0.15 * inch))
                
                # Question
                question = self._generate_question(problem_name)
                story.append(Paragraph(f"<b>Question:</b> {question}", self.styles['Question']))
                story.append(Spacer(1, 0.15 * inch))
                
                # Generate answer
                answer = self.answer_gen.generate_answer(problem_name, instance.instance_data)
                
                # Answer section
                story.append(Paragraph("Answer:", self.styles['AnswerHeading']))
                story.extend(self._format_answer(answer))
                
                story.append(Spacer(1, 0.3 * inch))
            
            # Page break between problems
            if i < len(problems):
                story.append(PageBreak())
        
        # Build PDF
        doc.build(story)
        print(f"PDF generated: {output_path}")
    
    def _generate_n_queens_instances(self, n: int) -> List[ProblemInstance]:
        """Generate N-Queens instances."""
        instances = []
        for _ in range(n):
            board_size = random.randint(4, 6)
            n_prime = random.randint(0, board_size - 2)
            instances.append(NQueensGenerator.generate(board_size, n_prime))
        return instances
    
    def _generate_hanoi_instances(self, n: int) -> List[ProblemInstance]:
        """Generate Tower of Hanoi instances."""
        instances = []
        for _ in range(n):
            n_disks = random.randint(3, 4)
            random_config = random.choice([False, True])
            instances.append(HanoiGenerator.generate(n_disks, random_config))
        return instances
    
    def _generate_graph_coloring_instances(self, n: int) -> List[ProblemInstance]:
        """Generate Graph Coloring instances."""
        instances = []
        for _ in range(n):
            n_vertices = random.randint(5, 7)
            n_colors = random.randint(3, 4)
            density = random.uniform(0.4, 0.6)
            instances.append(GraphColoringGenerator.generate(n_vertices, n_colors, density))
        return instances
    
    def _generate_knight_tour_instances(self, n: int) -> List[ProblemInstance]:
        """Generate Knight's Tour instances."""
        instances = []
        for _ in range(n):
            board_size = random.choice([5, 6])
            instances.append(KnightTourGenerator.generate(board_size))
        return instances
    
    def _generate_8puzzle_instances(self, n: int) -> List[ProblemInstance]:
        """Generate 8-Puzzle instances."""
        instances = []
        for _ in range(n):
            n_moves = random.randint(8, 15)
            instances.append(Puzzle8Generator.generate(n_moves))
        return instances
    
    def _generate_question(self, problem_name: str) -> str:
        """Generate question text."""
        return f"""For the {problem_name} problem and the given instance, 
        which is the most appropriate solving strategy among those mentioned 
        in the course (BFS, DFS, UCS, A*, GBFS, IDA*, Hill Climbing, Simulated Annealing)?"""
    
    def _visualize_instance(self, instance: ProblemInstance) -> List:
        """Visualize problem instance based on type."""
        elements = []
        data = instance.instance_data
        
        if instance.problem_type == "N-Queens":
            elements.extend(self._visualize_n_queens(data))
        elif instance.problem_type == "Tower of Hanoi":
            elements.extend(self._visualize_hanoi(data))
        elif instance.problem_type == "Graph Coloring":
            elements.extend(self._visualize_graph_coloring(data))
        elif instance.problem_type == "Knight's Tour":
            elements.extend(self._visualize_knight_tour(data))
        elif instance.problem_type == "8-Puzzle":
            elements.extend(self._visualize_8puzzle(data))
        
        return elements
    
    def _visualize_n_queens(self, data: Dict) -> List:
        """Visualize N-Queens board as table."""
        n = data['n']
        board = data['board']
        
        # Create table data
        table_data = []
        for i in range(n):
            row = []
            for j in range(n):
                if board[i][j] == 1:
                    row.append('Q')
                else:
                    row.append('·')
            table_data.append(row)
        
        # Create table
        t = Table(table_data, colWidths=[0.4 * inch] * n)
        t.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), 'Courier-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey)
        ]))
        
        info = Paragraph(
            f"<b>Board Size:</b> {n}×{n}, <b>Pre-placed Queens:</b> {data['n_prime']}",
            self.styles['Normal']
        )
        
        return [info, Spacer(1, 0.1 * inch), t]
    
    def _visualize_hanoi(self, data: Dict) -> List:
        """Visualize Tower of Hanoi configuration."""
        n_disks = data['n_disks']
        initial = data['initial']
        goal = data['goal']
        
        # Create text representation
        def format_peg(peg_name, disks):
            if disks:
                return f"{peg_name}: {disks}"
            else:
                return f"{peg_name}: []"
        
        initial_text = f"""
        <b>Initial Configuration:</b><br/>
        {format_peg('Peg A', initial['A'])}<br/>
        {format_peg('Peg B', initial['B'])}<br/>
        {format_peg('Peg C', initial['C'])}<br/>
        """
        
        goal_text = f"""
        <b>Goal:</b> All disks on Peg C
        """
        
        info = Paragraph(f"<b>Number of Disks:</b> {n_disks}", self.styles['Normal'])
        initial_p = Paragraph(initial_text, self.styles['Normal'])
        goal_p = Paragraph(goal_text, self.styles['Normal'])
        
        return [info, Spacer(1, 0.1 * inch), initial_p, Spacer(1, 0.05 * inch), goal_p]
    
    def _visualize_graph_coloring(self, data: Dict) -> List:
        """Visualize Graph Coloring problem."""
        n_vertices = data['n_vertices']
        n_colors = data['n_colors']
        edges = data['edges']
        
        info = Paragraph(
            f"<b>Vertices:</b> {n_vertices}, <b>Colors Available:</b> {n_colors}, <b>Edges:</b> {len(edges)}",
            self.styles['Normal']
        )
        
        # Create adjacency list text
        adj_text = "<b>Graph Edges:</b><br/>"
        edge_strings = [f"({u}, {v})" for u, v in edges[:15]]  # Show first 15 edges
        adj_text += ", ".join(edge_strings)
        if len(edges) > 15:
            adj_text += f", ... ({len(edges) - 15} more)"
        
        edges_p = Paragraph(adj_text, self.styles['Normal'])
        
        return [info, Spacer(1, 0.1 * inch), edges_p]
    
    def _visualize_knight_tour(self, data: Dict) -> List:
        """Visualize Knight's Tour board."""
        board_size = data['board_size']
        visited = data['visited']
        
        # Create board with visited squares marked
        board = [['·' for _ in range(board_size)] for _ in range(board_size)]
        for idx, (x, y) in enumerate(visited, 1):
            board[x][y] = str(idx)
        
        # Create table
        table_data = board
        t = Table(table_data, colWidths=[0.35 * inch] * board_size)
        t.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), 'Courier-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey)
        ]))
        
        info = Paragraph(
            f"<b>Board Size:</b> {board_size}×{board_size}, "
            f"<b>Starting Position:</b> {data['start_pos']}, "
            f"<b>Visited Squares:</b> {data['n_visited']}",
            self.styles['Normal']
        )
        
        return [info, Spacer(1, 0.1 * inch), t]
    
    def _visualize_8puzzle(self, data: Dict) -> List:
        """Visualize 8-Puzzle state."""
        initial = data['initial']
        
        # Create table for puzzle
        table_data = []
        for row in initial:
            formatted_row = []
            for val in row:
                if val == 0:
                    formatted_row.append(' ')
                else:
                    formatted_row.append(str(val))
            table_data.append(formatted_row)
        
        t = Table(table_data, colWidths=[0.5 * inch] * 3)
        t.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 14),
            ('GRID', (0, 0), (-1, -1), 2, colors.black),
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey)
        ]))
        
        info = Paragraph(
            f"<b>Initial State</b> (Goal: 1-2-3-4-5-6-7-8-blank)",
            self.styles['Normal']
        )
        
        return [info, Spacer(1, 0.1 * inch), t]
    
    def _format_answer(self, answer: Dict) -> List:
        """Format answer into PDF elements."""
        elements = []
        
        recommendations = answer['recommendations']
        if not recommendations:
            elements.append(Paragraph("No specific recommendations available.", self.styles['AnswerBody']))
            return elements
        
        # Best recommendation
        best = recommendations[0]
        
        # Main recommendation
        main_text = f"""
        <b>Best Strategy: {best['algorithm']}</b><br/>
        <i>{best['reason']}</i>
        """
        elements.append(Paragraph(main_text, self.styles['AnswerBody']))
        elements.append(Spacer(1, 0.1 * inch))
        
        # Complexity
        if best.get('complexity'):
            comp = best['complexity']
            comp_text = "<b>Complexity:</b> "
            if 'time' in comp:
                comp_text += f"Time: {comp['time']}"
            if 'space' in comp:
                comp_text += f", Space: {comp['space']}"
            elements.append(Paragraph(comp_text, self.styles['AnswerBody']))
            elements.append(Spacer(1, 0.05 * inch))
        
        # Properties
        if best.get('properties'):
            props = best['properties']
            prop_list = []
            if props.get('optimal'):
                prop_list.append("Optimal")
            if props.get('complete'):
                prop_list.append("Complete")
            if props.get('admissible'):
                prop_list.append("Admissible heuristic")
            
            if prop_list:
                prop_text = f"<b>Properties:</b> {', '.join(prop_list)}"
                elements.append(Paragraph(prop_text, self.styles['AnswerBody']))
                elements.append(Spacer(1, 0.05 * inch))
        
        # Alternative strategies
        if len(recommendations) > 1:
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph("<b>Alternative Strategies:</b>", self.styles['AnswerBody']))
            
            for alt in recommendations[1:]:
                alt_text = f"• <b>{alt['algorithm']}</b>: {alt['when_to_use']}"
                elements.append(Paragraph(alt_text, self.styles['AnswerBody']))
        
        # Heuristics
        if answer.get('heuristics'):
            elements.append(Spacer(1, 0.1 * inch))
            heur_text = f"<b>Recommended Heuristics:</b> {', '.join(answer['heuristics'])}"
            elements.append(Paragraph(heur_text, self.styles['AnswerBody']))
        
        return elements


def main():
    """Generate PDF with questions and answers."""
    print("Generating PDF with AI Search Strategy Questions...")
    print("=" * 80)
    
    generator = PDFQuestionGenerator('knowledge_graph.json')
    
    # Generate with 2 instances per problem (can be 1-3)
    n_instances = random.randint(1, 3)
    print(f"Generating {n_instances} instance(s) per problem...")
    
    output_path = 'AI_Search_Questions.pdf'
    generator.generate_pdf(output_path, n_instances_per_problem=n_instances)
    
    print(f"\n✓ PDF generated successfully: {output_path}")
    print(f"  - 5 problems covered")
    print(f"  - {n_instances} instance(s) per problem")
    print(f"  - Total: {5 * n_instances} questions with detailed answers")


if __name__ == "__main__":
    main()
