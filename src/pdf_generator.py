"""
PDF Question Generator
======================
Generates PDF with problem instances, questions, and answers.
"""

import random
from reportlab.graphics.shapes import Drawing, Circle, Line, String
from reportlab.platypus import Spacer
from typing import List, Dict
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Flowable
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfbase.pdfdoc import PDFDictionary, PDFName, PDFString, PDFArray

from .problem_generators import (
    NQueensGenerator, HanoiGenerator, GraphColoringGenerator,
    KnightTourGenerator, Puzzle8Generator, ProblemInstance
)
from .answer_generator import AnswerGenerator


class InteractiveTextField(Flowable):
    """A flowable that draws an interactive PDF text field."""
    def __init__(self, name, width=450, height=60, value="", read_only=False, multiline=True):
        Flowable.__init__(self)
        self.name = name
        self.width = width
        self.height = height
        self.value = value
        self.read_only = read_only
        self.multiline = multiline

    def wrap(self, availWidth, availHeight):
        return (self.width, self.height)

    def draw(self):
        self.canv.saveState()
        flags = ''
        if self.multiline:
            flags += ' multiline'
        if self.read_only:
            flags += ' readOnly'
            
        self.canv.acroForm.textfield(
            name=self.name,
            value=self.value,
            tooltip=f"Answer for {self.name}" if self.read_only else f"Enter answer for {self.name}",
            x=0, y=0, 
            width=self.width, 
            height=self.height,
            borderStyle='solid',
            borderColor=colors.black,
            # Force white background for editable fields, lightgrey for read-only
            fillColor=colors.lightgrey if self.read_only else colors.white,
            textColor=colors.black,
            forceBorder=True,
            fieldFlags=flags.strip(),
            relative=True
        )
        self.canv.restoreState()


class InteractiveDropdown(Flowable):
    """A flowable that draws an interactive PDF dropdown (combobox)."""
    def __init__(self, name, options, width=200, height=20, value=""):
        Flowable.__init__(self)
        self.name = name
        self.options = options
        self.width = width
        self.height = height
        # Workaround for ReportLab bug: value must not be empty for choice fields
        if not value and options:
            self.value = options[0]
        else:
            self.value = value

    def wrap(self, availWidth, availHeight):
        return (self.width, self.height)

    def draw(self):
        self.canv.saveState()
        
        self.canv.acroForm.choice(
            name=self.name,
            tooltip=f"Select answer for {self.name}",
            value=self.value,
            options=self.options,
            x=0, y=0,
            width=self.width,
            height=self.height,
            borderStyle='solid',
            borderColor=colors.black,
            fillColor=colors.white,
            textColor=colors.black,
            forceBorder=True,
            fieldFlags='combo',
            relative=True
        )
        self.canv.restoreState()


class CheckStrategyButton(Flowable):
    """A flowable that draws a button to check the strategy answer with JS."""
    def __init__(self, correct_strategy, field_name, status_field_name, width=120, height=30):
        Flowable.__init__(self)
        self.correct_strategy = correct_strategy
        self.field_name = field_name
        self.status_field_name = status_field_name
        self.width = width
        self.height = height

    def wrap(self, availWidth, availHeight):
        return (self.width, self.height)

    def draw(self):
        self.canv.saveState()
        
        # Draw button appearance
        self.canv.setFillColor(colors.lightgrey)
        self.canv.setStrokeColor(colors.black)
        self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=1)
        
        self.canv.setFillColor(colors.black)
        self.canv.setFont("Helvetica-Bold", 10)
        self.canv.drawCentredString(self.width/2, self.height/2 - 3, "Check Answer")
        
        btn_name = f"btn_check_{self.field_name}"
        
        js_code = """
        try {
            var fAnswer = this.getField("%s");
            var fStatus = this.getField("%s");
            var fBtn = this.getField("%s");
            
            if (!fAnswer) {
                app.alert("Error: Answer field not found.");
            } else {
                var valAnswer = fAnswer.value;
                if (valAnswer == null) valAnswer = "";
                
                var expected = "%s";
                
                // Simple string comparison
                if (valAnswer === expected) {
                    if (fStatus) { fStatus.value = "Correct!"; }
                    if (fBtn) { fBtn.buttonSetCaption("Correct!"); }
                    app.alert("Correct! " + expected + " is the best strategy.");
                } else {
                    if (fStatus) { fStatus.value = "Wrong"; }
                    if (fBtn) { fBtn.buttonSetCaption("Try Again"); }
                    app.alert("Incorrect. Try again!");
                }
            }
        } catch (e) {
            app.alert("JS Error: " + e);
        }
        """ % (self.field_name, self.status_field_name, btn_name, self.correct_strategy)
        
        # Create the widget annotation
        x, y = self.canv.absolutePosition(0, 0)
        rect = PDFArray([x, y, x + self.width, y + self.height])
        
        action = PDFDictionary({
            'S': PDFName('JavaScript'),
            'JS': PDFString(js_code)
        })
        
        annot = PDFDictionary({
            'Type': PDFName('Annot'),
            'Subtype': PDFName('Widget'),
            'FT': PDFName('Btn'),
            'Ff': 65536, # PushButton flag
            'Rect': rect,
            'A': action,
            'T': PDFString(btn_name),
            'MK': PDFDictionary({
                'BG': PDFArray([0.8, 0.8, 0.8]), 
                'BC': PDFArray([0, 0, 0]),       
                'CA': PDFString("Check Answer")  
            })
        })
        
        self.canv._addAnnotation(annot)
        self.canv.restoreState()


class CheckAnswerButton(Flowable):
    """A flowable that draws a button to check the answer with JS."""
    def __init__(self, correct_root, correct_leaves, field_root_name, field_leaves_name, status_field_name, width=120, height=30):
        Flowable.__init__(self)
        self.correct_root = correct_root
        self.correct_leaves = correct_leaves
        self.field_root_name = field_root_name
        self.field_leaves_name = field_leaves_name
        self.status_field_name = status_field_name
        self.width = width
        self.height = height

    def wrap(self, availWidth, availHeight):
        return (self.width, self.height)

    def draw(self):
        self.canv.saveState()
        
        # Draw button appearance
        self.canv.setFillColor(colors.lightgrey)
        self.canv.setStrokeColor(colors.black)
        self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=1)
        
        self.canv.setFillColor(colors.black)
        self.canv.setFont("Helvetica-Bold", 10)
        self.canv.drawCentredString(self.width/2, self.height/2 - 3, "Check Answer")
        
        # JavaScript for validation
        # We use try-catch and explicit null checks for robustness
        # We now update: 1. Status Field, 2. Button Caption, 3. Alert
        btn_name = f"btn_check_{self.field_root_name}"
        
        js_code = """
        try {
            var fRoot = this.getField("%s");
            var fLeaves = this.getField("%s");
            var fStatus = this.getField("%s");
            var fBtn = this.getField("%s");
            
            if (!fRoot || !fLeaves) {
                app.alert("Error: Fields not found. Please use Adobe Acrobat Reader.");
            } else {
                var valRoot = fRoot.value;
                var valLeaves = fLeaves.value;
                
                if (valRoot == null) valRoot = "";
                if (valLeaves == null) valLeaves = "";
                
                valRoot = String(valRoot).replace(/^\\s+|\\s+$/g, '');
                valLeaves = String(valLeaves).replace(/^\\s+|\\s+$/g, '');
                
                var expectedRoot = "%s";
                var expectedLeaves = "%s";
                
                if (valRoot == expectedRoot && valLeaves == expectedLeaves) {
                    // 1. Update Status Field
                    if (fStatus) { fStatus.value = "Correct!"; }
                    
                    // 2. Update Button Caption
                    if (fBtn) { fBtn.buttonSetCaption("Correct!"); }
                    
                    // 3. Show Alert
                    app.alert("Correct! Great job.");
                } else {
                    // 1. Update Status Field
                    if (fStatus) { fStatus.value = "Wrong"; }
                    
                    // 2. Update Button Caption
                    if (fBtn) { fBtn.buttonSetCaption("Try Again"); }
                    
                    // 3. Show Alert
                    app.alert("Incorrect.\\nYour answer: Root=" + valRoot + ", Leaves=" + valLeaves + "\\nExpected: Root=" + expectedRoot + ", Leaves=" + expectedLeaves);
                }
            }
        } catch (e) {
            app.alert("JS Error: " + e);
        }
        """ % (self.field_root_name, self.field_leaves_name, self.status_field_name, btn_name,
               self.correct_root, self.correct_leaves)
        
        # Create the widget annotation
        x, y = self.canv.absolutePosition(0, 0)
        rect = PDFArray([x, y, x + self.width, y + self.height])
        
        action = PDFDictionary({
            'S': PDFName('JavaScript'),
            'JS': PDFString(js_code)
        })
        
        annot = PDFDictionary({
            'Type': PDFName('Annot'),
            'Subtype': PDFName('Widget'),
            'FT': PDFName('Btn'),
            'Ff': 65536, # PushButton flag
            'Rect': rect,
            'A': action,
            'T': PDFString(btn_name),
            'MK': PDFDictionary({
                'BG': PDFArray([0.8, 0.8, 0.8]), 
                'BC': PDFArray([0, 0, 0]),       
                'CA': PDFString("Check Answer")  
            })
        })
        
        self.canv._addAnnotation(annot)
        self.canv.restoreState()


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
    
    def generate_pdf(self, output_path: str, n_instances_per_problem: int = 2, include_answers: bool = True, problem_config: List[Dict] = None):
        """
        Generate complete PDF with questions and optionally answers.
        
        Args:
            output_path: Path to save the PDF
            n_instances_per_problem: Default number of instances (used if problem_config is None)
            include_answers: Whether to include answers in the PDF
            problem_config: Optional list of dicts defining the problems to include.
                            Example: [{'name': 'N-Queens', 'count': 2}, {'name': 'MinMax', 'count': 1}]
        """
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
        title_text = "AI Search Strategy Questions" + (" (With Answers)" if include_answers else " (Worksheet)")
        story.append(Paragraph(title_text, self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(
            "Generated from Knowledge Graph Analysis",
            self.styles['Normal']
        ))
        story.append(Spacer(1, 0.5 * inch))
        
        # Introduction
        if include_answers:
            intro_text = """
            This document contains problem instances for various AI search problems and questions 
            about the most appropriate solving strategies. Each problem includes instance 
            visualizations and detailed answers based on knowledge graph analysis.
            """
        else:
            intro_text = """
            This document contains problem instances for various AI search problems. 
            Please analyze each instance and determine the most appropriate solving strategy.
            You can type your answers directly into the text boxes provided.
            """
        story.append(Paragraph(intro_text, self.styles['Normal']))
        story.append(PageBreak())
        
        # Map problem names to generator functions
        generator_map = {
            'N-Queens': self._generate_n_queens_instances,
            'Tower of Hanoi': self._generate_hanoi_instances,
            'Graph Coloring': self._generate_graph_coloring_instances,
            'Knight\'s Tour': self._generate_knight_tour_instances,
            '8-Puzzle': self._generate_8puzzle_instances,
            'MinMax': self._generate_minmax_instances
        }

        # Determine problems to generate
        problems_to_process = []
        if problem_config:
            for item in problem_config:
                name = item.get('name')
                count = item.get('count', 1)
                if name in generator_map:
                    problems_to_process.append((name, generator_map[name], count))
        else:
            # Default behavior
            default_problems = [
                'N-Queens', 'Tower of Hanoi', 'Graph Coloring', 
                'Knight\'s Tour', '8-Puzzle', 'MinMax'
            ]
            for name in default_problems:
                problems_to_process.append((name, generator_map[name], n_instances_per_problem))
        
        for i, (problem_name, generator_func, n_instances) in enumerate(problems_to_process, 1):
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
                
                if include_answers:
                    # Generate answer
                    story.append(Paragraph("Answer:", self.styles['AnswerHeading']))

                    if problem_name == "MinMax":
                        # doar calcul local, fără AnswerGenerator
                        tree = instance.instance_data['tree']
                        root_value, leaves_visited = self._minmax_alpha_beta(tree)
                        story.append(Paragraph(
                            f"<b>Root Value:</b> {root_value}<br/>"
                            f"<b>Leaves Visited with Alpha-Beta:</b> {leaves_visited}",
                            self.styles['AnswerBody']
                        ))
                        story.append(Spacer(1, 0.5 * inch))
                    else:
                        # restul problemelor folosesc AnswerGenerator
                        answer = self.answer_gen.generate_answer(problem_name, instance.instance_data)
                        story.extend(self._format_answer(answer))
                else:
                    if problem_name == "MinMax":
                        # Specific layout for MinMax: Split fields + Check Button
                        story.append(Paragraph("Your Answer:", self.styles['AnswerHeading']))
                        story.append(Spacer(1, 0.1 * inch))
                        
                        # Root Value
                        story.append(Paragraph("Root Value:", self.styles['Normal']))
                        root_field_name = f"minmax_root_{j}"
                        story.append(InteractiveTextField(root_field_name, width=150, height=20, multiline=False))
                        story.append(Spacer(1, 0.1 * inch))
                        
                        # Leaves Visited
                        story.append(Paragraph("Leaves Visited:", self.styles['Normal']))
                        leaves_field_name = f"minmax_leaves_{j}"
                        story.append(InteractiveTextField(leaves_field_name, width=150, height=20, multiline=False))
                        story.append(Spacer(1, 0.2 * inch))
                        
                        # Calculate correct values
                        tree = instance.instance_data['tree']
                        root_value, leaves_visited = self._minmax_alpha_beta(tree)
                        
                        # Check Button
                        status_field_name = f"minmax_status_{j}"
                        
                        # Add a label and the status field (initialized to "Ready")
                        story.append(Paragraph("Status:", self.styles['Normal']))
                        # IMPORTANT: read_only=False to ensure JS can update it in all viewers
                        story.append(InteractiveTextField(status_field_name, width=200, height=20, value="Ready...", multiline=False, read_only=False))
                        story.append(Spacer(1, 0.1 * inch))
                        
                        story.append(CheckAnswerButton(
                            correct_root=str(root_value),
                            correct_leaves=str(leaves_visited),
                            field_root_name=root_field_name,
                            field_leaves_name=leaves_field_name,
                            status_field_name=status_field_name
                        ))
                        
                        # Add a note about viewer compatibility
                        story.append(Spacer(1, 0.1 * inch))
                        story.append(Paragraph(
                            "<i>Note: This interactive check requires a PDF viewer with JavaScript support (e.g., Adobe Acrobat Reader, Foxit Reader, or Master PDF Editor). It may not work in browser-based viewers.</i>",
                            self.styles['AnswerBody']
                        ))
                    else:
                        # === Strategy Problems (N-Queens, Hanoi, etc.) ===
                        # Use Dropdown + Check Button
                        
                        # 1. Get correct answer from KG
                        answer_data = self.answer_gen.generate_answer(problem_name, instance.instance_data)
                        recommendations = answer_data.get('recommendations', [])
                        correct_strategy = recommendations[0]['algorithm'] if recommendations else "Unknown"
                        
                        story.append(Paragraph("Select Best Strategy:", self.styles['AnswerHeading']))
                        story.append(Spacer(1, 0.1 * inch))
                        
                        # 2. Dropdown with options
                        field_name = f"strategy_{problem_name.replace(' ', '_')}_{j}"
                        options = [
                            "Select Strategy...",
                            "BFS", "DFS", "UCS", "IDDFS", "A*", "IDA*", "GBFS", "RBFS",
                            "Hill Climbing", "Simulated Annealing", "Backtracking", 
                            "Min-Conflicts", "Forward Checking", "Arc Consistency"
                        ]
                        # Ensure correct strategy is in the list
                        if correct_strategy not in options:
                            options.append(correct_strategy)
                        options.sort()
                        
                        story.append(InteractiveDropdown(field_name, options, width=250))
                        story.append(Spacer(1, 0.1 * inch))
                        
                        # 3. Check Button
                        status_field_name = f"status_{problem_name.replace(' ', '_')}_{j}"
                        story.append(Paragraph("Status:", self.styles['Normal']))
                        story.append(InteractiveTextField(status_field_name, width=200, height=20, value="Ready...", multiline=False, read_only=False))
                        story.append(Spacer(1, 0.1 * inch))
                        
                        story.append(CheckStrategyButton(
                            correct_strategy=correct_strategy,
                            field_name=field_name,
                            status_field_name=status_field_name
                        ))
                        
                        story.append(Spacer(1, 0.1 * inch))
                        story.append(Paragraph(
                            "<i>Note: Requires PDF viewer with JavaScript support.</i>",
                            self.styles['AnswerBody']
                        ))
                    
                    story.append(Spacer(1, 0.3 * inch))
            
            # Separator between problems
            if i < len(problems_to_process):
                story.append(Spacer(1, 0.4 * inch))
        
        # Build PDF
        doc.build(story)
        print(f"PDF generated: {output_path}")
    
    def generate_pdf_single_problem(self, output_path: str, problem_name: str, n_instances: int = 2):
        """Generate PDF with questions for a SINGLE problem."""
        print(f"\n{'='*80}")
        print(f"GENERATING PDF FOR: {problem_name}")
        print(f"{'='*80}")
        
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
            f"Problem: {problem_name}",
            self.styles['ProblemTitle']
        ))
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph(
            "Generated from Knowledge Graph Analysis",
            self.styles['Normal']
        ))
        story.append(Spacer(1, 0.5 * inch))
        
        # Introduction
        intro_text = f"""
        This document contains {n_instances} instance(s) of the {problem_name} problem with questions 
        about the most appropriate solving strategies. Each instance includes visualizations 
        and detailed answers based on knowledge graph analysis.
        """
        story.append(Paragraph(intro_text, self.styles['Normal']))
        story.append(PageBreak())
        
        # Map problem name to generator function
        generator_map = {
            'N-Queens': self._generate_n_queens_instances,
            'Tower of Hanoi': self._generate_hanoi_instances,
            'Graph Coloring': self._generate_graph_coloring_instances,
            'Knight\'s Tour': self._generate_knight_tour_instances,
            '8-Puzzle': self._generate_8puzzle_instances,
            'MinMax': self._generate_minmax_instances  

        }
        
        if problem_name not in generator_map:
            raise ValueError(f"Unknown problem: {problem_name}. Must be one of {list(generator_map.keys())}")
        
        generator_func = generator_map[problem_name]
        
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
            
            # Generate answer from knowledge graph
            story.append(Paragraph("Answer:", self.styles['AnswerHeading']))

            if problem_name == "MinMax":
                # calcul local, fără AnswerGenerator
                tree = instance.instance_data['tree']
                root_value, leaves_visited = self._minmax_alpha_beta(tree)
                story.append(Paragraph(
                    f"<b>Root Value:</b> {root_value}<br/>"
                    f"<b>Leaves Visited with Alpha-Beta:</b> {leaves_visited}",
                    self.styles['AnswerBody']
                ))
                story.append(Spacer(1, 0.5 * inch))
            else:
                # restul problemelor folosesc AnswerGenerator
                answer = self.answer_gen.generate_answer(problem_name, instance.instance_data)
                story.extend(self._format_answer(answer))

            
            story.append(Spacer(1, 0.3 * inch))
        
        # Build PDF
        doc.build(story)
        print(f"PDF generated: {output_path}")
    def _generate_minmax_instances(self, n: int) -> List[ProblemInstance]:
        """Generate n random MinMax tree instances."""
        instances = []
        for _ in range(n):
            depth = random.randint(2, 4)  # nivele arbore
            max_children = random.randint(2, 3)  # copii pe nod
            # Folosim o funcție helper care creează arborele cu valori la frunze
            tree_data = self._create_random_minmax_tree(depth, max_children)
            instance = ProblemInstance(problem_type="MinMax", instance_data={"tree": tree_data})
            instances.append(instance)
        return instances
    def _create_random_minmax_tree(self, depth, max_children, max_nodes=20, current_count=None):
        """
        Returnează un arbore MinMax cu valori doar la frunze, limitat la max_nodes noduri.
        Folosim current_count ca listă pentru a putea modifica valoarea în recursie.
        """
        if current_count is None:
            current_count = [0]
            
        # Increment for the current node being created
        current_count[0] += 1
        
        # Check limits
        if current_count[0] > max_nodes or depth == 0:
            return {"value": random.randint(1, 20)}
            
        # Internal node
        n_children = random.randint(2, min(max_children, 3))
        children = []
        for _ in range(n_children):
            if current_count[0] >= max_nodes:
                break
            child = self._create_random_minmax_tree(depth-1, max_children, max_nodes, current_count)
            children.append(child)
            
        if not children:
            return {"value": random.randint(1, 20)}
            
        return {"children": children}

    def _draw_minmax_tree(self, tree, width=500, height=400, max_levels=4):
        """Desenează arbore MinMax cu spațiere mai mare între noduri."""
        drawing = Drawing(width, height)
        level_height = 100  # mai mult spațiu între nivele

        def count_leaves(node):
            if "value" in node:
                return 1
            elif "children" in node:
                return sum(count_leaves(c) for c in node["children"])
            return 0

        def draw_node(node, x, y, level, parent_coords=None):
            node_radius = 12
            if parent_coords:
                px, py = parent_coords
                drawing.add(Line(px, py - node_radius, x, y + node_radius))

            if "value" in node:
                drawing.add(Circle(x, y, node_radius, fillColor=None))
                drawing.add(String(x-4, y-4, str(node['value']), fontSize=8))
                return 1
            elif "children" in node:
                n_leaves = [count_leaves(c) for c in node["children"]]
                total_leaves = sum(n_leaves)
                spacing = 40  # spațiu mai mare între nodurile copii
                x_start = x - (total_leaves * spacing) / 2
                acc = 0
                for i, child in enumerate(node["children"]):
                    cx = x_start + acc*spacing + n_leaves[i]*spacing/2
                    cy = y - level_height
                    draw_node(child, cx, cy, level+1, (x, y))
                    acc += n_leaves[i]
                drawing.add(Circle(x, y, node_radius, fillColor=None))
                return total_leaves
            return 0

        draw_node(tree, width/2, height - 20, 0)
        return [drawing, Spacer(1, 0.3*inch)]  # mai mult spațiu după arbore

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
        if problem_name == "MinMax":
            return ("For the given MinMax tree, what is the value at the root "
                    "and how many leaf nodes are visited when applying MinMax with Alpha-Beta pruning?")
        else:
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
        elif instance.problem_type == "MinMax":  # ← nou
            elements.extend(self._visualize_minmax(data))
        
        return elements
    def _visualize_minmax(self, data) -> list:
        """Returnează o vizualizare grafică a arborelui MinMax."""
        tree = data.get("tree")
        return self._draw_minmax_tree(tree)

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
    def _minmax_alpha_beta(self, node, maximizing=True, alpha=float('-inf'), beta=float('inf'), counter=None):
        """
        Apply MinMax with Alpha-Beta pruning.
        Returns (value, leaves_visited)
        """
        if counter is None:
            counter = {"leaves": 0}

        # Leaf node check
        if "value" in node:
            counter["leaves"] += 1
            return node["value"], counter["leaves"]

        # Internal node
        if maximizing:
            value = float('-inf')
            for child in node.get("children", []):  # safer access
                child_value, _ = self._minmax_alpha_beta(child, False, alpha, beta, counter)
                value = max(value, child_value)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break  # prune
            return value, counter["leaves"]
        else:
            value = float('inf')
            for child in node.get("children", []):
                child_value, _ = self._minmax_alpha_beta(child, True, alpha, beta, counter)
                value = min(value, child_value)
                beta = min(beta, value)
                if beta <= alpha:
                    break  # prune
            return value, counter["leaves"]

    def _format_answer(self, answer: Dict) -> List:
        """Format answer into PDF elements with more spacing for readability."""
        elements = []

        # === Special handling for MinMax problem ===
        if answer.get("problem_type") == "MinMax":
            tree = answer['tree']  # arborele MinMax
            root_value, leaves_visited = self._minmax_alpha_beta(tree)
            elements.append(Paragraph(
                f"<b>Root Value:</b> {root_value}<br/>"
                f"<b>Leaves Visited with Alpha-Beta:</b> {leaves_visited}",
                self.styles['AnswerBody']
            ))
            elements.append(Spacer(1, 0.5 * inch))  # mai mult spațiu după MinMax
            return elements

        # === Other problems ===
        recommendations = answer.get('recommendations', [])
        if not recommendations:
            elements.append(Paragraph("No specific recommendations available.", self.styles['AnswerBody']))
            elements.append(Spacer(1, 0.3 * inch))
            return elements

        best = recommendations[0]
        
        # Best strategy
        main_text = f"""
        <b>Best Strategy: {best['algorithm']}</b><br/>
        <i>{best['reason']}</i>
        """
        elements.append(Paragraph(main_text, self.styles['AnswerBody']))
        elements.append(Spacer(1, 0.25 * inch))

        # Complexity
        if best.get('complexity'):
            comp = best['complexity']
            comp_text = "<b>Complexity:</b> "
            if 'time' in comp:
                comp_text += f"Time: {comp['time']}"
            if 'space' in comp:
                comp_text += f", Space: {comp['space']}"
            elements.append(Paragraph(comp_text, self.styles['AnswerBody']))
            elements.append(Spacer(1, 0.15 * inch))

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
                elements.append(Spacer(1, 0.15 * inch))

        # Alternative strategies
        if len(recommendations) > 1:
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(Paragraph("<b>Alternative Strategies:</b>", self.styles['AnswerBody']))
            elements.append(Spacer(1, 0.1 * inch))
            for alt in recommendations[1:]:
                alt_text = f"• <b>{alt['algorithm']}</b>: {alt['when_to_use']}"
                elements.append(Paragraph(alt_text, self.styles['AnswerBody']))
                elements.append(Spacer(1, 0.1 * inch))

        # Recommended heuristics
        if answer.get('heuristics'):
            elements.append(Spacer(1, 0.2 * inch))
            heur_text = f"<b>Recommended Heuristics:</b> {', '.join(answer['heuristics'])}"
            elements.append(Paragraph(heur_text, self.styles['AnswerBody']))
            elements.append(Spacer(1, 0.3 * inch))

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
    print(f"  - 6 problems covered")
    print(f"  - {n_instances} instance(s) per problem")
    print(f"  - Total: {6 * n_instances} questions with detailed answers")


if __name__ == "__main__":
    main()
