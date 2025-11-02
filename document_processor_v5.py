"""
Document Processor V5 - Pre-seeded Knowledge Graph with Enhanced Relationship Detection
=========================================================================================

Major Improvements:
1. Pre-seed KG with all known algorithms, problems, heuristics
2. Enhanced relationship detection with pattern matching
3. Post-processing phase for relationship inference
4. Better edge creation with semantic analysis
5. Richer graph connectivity
"""

import re
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from abc import ABC, abstractmethod
import requests
from bs4 import BeautifulSoup

from knowledge_graph import KnowledgeGraph, Node, Edge

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


@dataclass
class EntityMention:
    """Represents a single mention of an entity in the document."""
    entity_name: str
    entity_type: str
    start_pos: int
    end_pos: int
    word_position: int
    context_before: str
    context_after: str
    sentence: str = ""  # Full sentence containing the mention
    properties: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def full_context(self) -> str:
        """Get full context around mention."""
        return f"{self.context_before} {self.entity_name} {self.context_after}"
    
    def distance_to(self, other: 'EntityMention') -> int:
        """Calculate word distance to another mention."""
        return abs(self.word_position - other.word_position)


@dataclass
class EntityProfile:
    """Aggregated profile of an entity from all its mentions."""
    name: str
    entity_type: str
    mentions: List[EntityMention] = field(default_factory=list)
    all_properties: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    co_occurring_entities: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def add_mention(self, mention: EntityMention):
        """Add a new mention and aggregate properties."""
        self.mentions.append(mention)
        for prop, value in mention.properties.items():
            if value:
                self.all_properties[prop] += 1
    
    @property
    def mention_count(self) -> int:
        return len(self.mentions)
    
    def get_top_properties(self, n: int = 5) -> List[Tuple[str, int]]:
        """Get most frequently mentioned properties."""
        return sorted(self.all_properties.items(), key=lambda x: x[1], reverse=True)[:n]
    
    def get_contexts_with(self, other_entity: str, max_contexts: int = 5) -> List[str]:
        """Get contexts where this entity appears with another entity."""
        contexts = []
        for mention in self.mentions:
            if other_entity.lower() in mention.full_context.lower():
                contexts.append(mention.full_context[:300])
                if len(contexts) >= max_contexts:
                    break
        return contexts


class DocumentReader(ABC):
    """Abstract base class for document readers."""
    
    @abstractmethod
    def read(self, source: str) -> str:
        pass


class LocalFileReader(DocumentReader):
    """Reads local text and PDF files."""
    
    def read(self, source: str) -> str:
        if source.lower().endswith('.pdf'):
            return self._read_pdf(source)
        else:
            return self._read_text(source)
    
    def _read_text(self, filepath: str) -> str:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading text file: {e}")
            return ""
    
    def _read_pdf(self, filepath: str) -> str:
        if not PDF_SUPPORT:
            print("PyPDF2 not installed. Cannot read PDF files.")
            return ""
        
        try:
            with open(filepath, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text_parts = []
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    text_parts.append(page_text)
                
                return '\n'.join(text_parts)
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""


class WebResourceReader(DocumentReader):
    """Reads web resources."""
    
    def read(self, source: str) -> str:
        try:
            response = requests.get(source, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)
            
            return text
        except Exception as e:
            print(f"Error reading web resource: {e}")
            return ""


class EnhancedEntityExtractor:
    """Enhanced entity extractor with comprehensive algorithm and problem knowledge."""
    
    def __init__(self):
        # Initialize comprehensive knowledge bases
        self.algorithms = self._init_algorithms()
        self.problems = self._init_problems()
        self.heuristics = self._init_heuristics()
        self.optimizations = self._init_optimizations()
        
        # Compile patterns
        self._compile_patterns()
        
        # Enhanced property patterns
        self.property_patterns = {
            'optimal': r'\b(optimal|optimality|optim(?:al|ize)?)\b',
            'complete': r'\b(complete(?:ness)?)\b',
            'efficient': r'\b(efficient|efficiency)\b',
            'slow': r'\b(slow(?:er|ly)?|inefficient)\b',
            'fast': r'\b(fast(?:er)?|quick(?:ly)?|rapid)\b',
            'guaranteed': r'\b(guarante\w+)\b',
            'admissible': r'\b(admissible|admissibility)\b',
            'consistent': r'\b(consistent|consistency|monotonic)\b',
        }
        
        # Relationship patterns
        self.relationship_patterns = {
            'solves': [
                r'{algo}\s+(?:solves?|solve|solving|used for|applied to|tackles?)\s+(?:the\s+)?{prob}',
                r'{prob}\s+(?:is\s+)?(?:solved|tackled|addressed)\s+(?:using|with|by)\s+{algo}',
                r'(?:use|using|apply|applying)\s+{algo}\s+(?:to|for)\s+(?:solve|solving)?\s*{prob}',
            ],
            'uses': [
                r'{algo}\s+(?:uses?|employs?|utilizes?)\s+(?:the\s+)?{heur}',
                r'{heur}\s+(?:is\s+)?(?:used|employed|utilized)\s+(?:in|by|with)\s+{algo}',
            ],
            'has_property': [
                r'{algo}\s+(?:is|are)\s+{prop}',
                r'{prop}\s+(?:algorithm|search|strategy).*{algo}',
            ],
            'better_than': [
                r'{algo1}\s+(?:is\s+)?(?:better|superior|faster|more\s+efficient)\s+(?:than|to)\s+{algo2}',
                r'{algo1}\s+(?:outperforms?|beats?)\s+{algo2}',
            ],
            'worse_than': [
                r'{algo1}\s+(?:is\s+)?(?:worse|slower|less\s+efficient)\s+(?:than|to)\s+{algo2}',
            ],
            'related_to': [
                r'{entity1}\s+(?:and|or|with)\s+{entity2}',
                r'(?:both|either)\s+{entity1}\s+(?:and|or)\s+{entity2}',
            ],
        }
    
    def _init_algorithms(self) -> Dict[str, Dict]:
        """Initialize COMPREHENSIVE algorithm knowledge base."""
        return {
            # Uninformed Search
            "bfs": {
                "name": "BFS",
                "full_name": "Breadth-First Search",
                "aliases": ["breadth first", "breadth-first", "cautare in latime", "căutare în lățime", "bfs"],
                "category": "uninformed",
                "default_properties": {"complete": True, "optimal": True}
            },
            "dfs": {
                "name": "DFS",
                "full_name": "Depth-First Search",
                "aliases": ["depth first", "depth-first", "cautare in adancime", "căutare în adâncime", "dfs"],
                "category": "uninformed",
                "default_properties": {"complete": False, "optimal": False}
            },
            "ucs": {
                "name": "UCS",
                "full_name": "Uniform Cost Search",
                "aliases": ["uniform cost", "ucs", "uniform-cost", "cautare cost uniform", "căutare cost uniform"],
                "category": "uninformed",
                "default_properties": {"complete": True, "optimal": True}
            },
            "dls": {
                "name": "DLS",
                "full_name": "Depth-Limited Search",
                "aliases": ["depth limited", "dls", "depth-limited", "cautare la adancime limitata"],
                "category": "uninformed",
                "default_properties": {"complete": False, "optimal": False}
            },
            "iddfs": {
                "name": "IDDFS",
                "full_name": "Iterative Deepening DFS",
                "aliases": ["iddfs", "iterative deepening", "id-dfs", "adancire iterativa"],
                "category": "uninformed",
                "default_properties": {"complete": True, "optimal": True}
            },
            "bidirectional": {
                "name": "Bidirectional Search",
                "full_name": "Bidirectional Search",
                "aliases": ["bidirectional", "bidirectional search", "cautare bidirectionala"],
                "category": "uninformed",
                "default_properties": {"complete": True, "optimal": True}
            },
            
            # Informed Search
            "a_star": {
                "name": "A*",
                "full_name": "A-Star Search",
                "aliases": ["a*", "a star", "a-star", "algoritmul a*", "algoritm a*", "astar"],
                "category": "informed",
                "default_properties": {"complete": True, "optimal": True, "admissible": True}
            },
            "ida_star": {
                "name": "IDA*",
                "full_name": "Iterative Deepening A*",
                "aliases": ["ida*", "ida star", "iterative deepening a*", "adancire iterativa a*"],
                "category": "informed",
                "default_properties": {"complete": True, "optimal": True, "admissible": True}
            },
            "gbfs": {
                "name": "GBFS",
                "full_name": "Greedy Best-First Search",
                "aliases": ["greedy best first", "gbfs", "greedy best-first", "cautare lacom", "căutare lacomă", "greedy"],
                "category": "informed",
                "default_properties": {"complete": False, "optimal": False}
            },
            "rbfs": {
                "name": "RBFS",
                "full_name": "Recursive Best-First Search",
                "aliases": ["rbfs", "recursive best first", "recursive best-first"],
                "category": "informed",
                "default_properties": {"complete": True, "optimal": True}
            },
            "sma_star": {
                "name": "SMA*",
                "full_name": "Simplified Memory-Bounded A*",
                "aliases": ["sma*", "sma star", "simplified memory bounded"],
                "category": "informed",
                "default_properties": {"complete": True, "optimal": True}
            },
            
            # Local Search
            "hill_climbing": {
                "name": "Hill Climbing",
                "full_name": "Hill Climbing",
                "aliases": ["hill climbing", "hill-climbing", "urcare pe deal", "urcarea dealului"],
                "category": "local_search",
                "default_properties": {"complete": False, "optimal": False}
            },
            "simulated_annealing": {
                "name": "Simulated Annealing",
                "full_name": "Simulated Annealing",
                "aliases": ["simulated annealing", "annealing", "racire simulata", "răcire simulată"],
                "category": "local_search",
                "default_properties": {"complete": False, "optimal": False}
            },
            "genetic": {
                "name": "Genetic Algorithm",
                "full_name": "Genetic Algorithm",
                "aliases": ["genetic algorithm", "genetic", "algoritm genetic", "ga"],
                "category": "local_search",
                "default_properties": {"complete": False, "optimal": False}
            },
            "beam_search": {
                "name": "Beam Search",
                "full_name": "Beam Search",
                "aliases": ["beam search", "beam", "cautare cu fascicul"],
                "category": "local_search",
                "default_properties": {"complete": False, "optimal": False}
            },
        }
    
    def _init_problems(self) -> Dict[str, Dict]:
        """Initialize comprehensive problem knowledge base."""
        return {
            "8_puzzle": {
                "name": "8-Puzzle",
                "aliases": ["8 puzzle", "8-puzzle", "8puzzle", "puzzle cu 8 piese", "eight puzzle"],
                "category": "sliding_tile",
                "default_properties": {"state_space": "181440"}
            },
            "15_puzzle": {
                "name": "15-Puzzle",
                "aliases": ["15 puzzle", "15-puzzle", "15puzzle", "puzzle cu 15 piese", "fifteen puzzle"],
                "category": "sliding_tile",
                "default_properties": {}
            },
            "n_queens": {
                "name": "N-Queens",
                "aliases": ["n queens", "n-queens", "8 queens", "problema damelor", "problema reginelor", "queens problem"],
                "category": "constraint_satisfaction",
                "default_properties": {}
            },
            "tsp": {
                "name": "TSP",
                "aliases": ["tsp", "traveling salesman", "travelling salesman", "problema comis-voiajorului", "salesman problem"],
                "category": "optimization",
                "default_properties": {}
            },
            "route_finding": {
                "name": "Route Finding",
                "aliases": ["route finding", "path finding", "pathfinding", "gasirea drumului", "găsirea rutei", "navigation"],
                "category": "navigation",
                "default_properties": {}
            },
            "knight_tour": {
                "name": "Knight's Tour",
                "aliases": ["knight tour", "knight's tour", "knights tour", "plimbarea calului", "turul calului", "problema calului"],
                "category": "path_finding",
                "default_properties": {}
            },
            "hanoi": {
                "name": "Tower of Hanoi",
                "aliases": ["hanoi", "tower of hanoi", "towers of hanoi", "turnurile din hanoi", "turnul din hanoi", "problema hanoi"],
                "category": "puzzle",
                "default_properties": {}
            },
            "graph_coloring": {
                "name": "Graph Coloring",
                "aliases": ["graph coloring", "graph colouring", "colorare graf", "colorarea grafurilor", "vertex coloring", "colorare de noduri"],
                "category": "constraint_satisfaction",
                "default_properties": {}
            },
            "sudoku": {
                "name": "Sudoku",
                "aliases": ["sudoku"],
                "category": "constraint_satisfaction",
                "default_properties": {}
            },
            "missionaries_cannibals": {
                "name": "Missionaries and Cannibals",
                "aliases": ["missionaries cannibals", "missionaries and cannibals", "misionarii si canibalii", "misionarii și canibalii"],
                "category": "state_space_search",
                "default_properties": {}
            },
            "maze": {
                "name": "Maze Solving",
                "aliases": ["maze", "maze solving", "labirint", "rezolvare labirint"],
                "category": "path_finding",
                "default_properties": {}
            },
        }
    
    def _init_heuristics(self) -> Dict[str, Dict]:
        """Initialize heuristic knowledge base."""
        return {
            "manhattan": {
                "name": "Manhattan Distance",
                "aliases": ["manhattan", "manhattan distance", "distanta manhattan", "distanța manhattan", "taxicab"],
                "category": "distance_metric",
                "default_properties": {"admissible": True, "consistent": True}
            },
            "euclidean": {
                "name": "Euclidean Distance",
                "aliases": ["euclidean", "euclidean distance", "distanta euclidiana", "distanța euclidiană", "straight line"],
                "category": "distance_metric",
                "default_properties": {"admissible": True, "consistent": True}
            },
            "misplaced_tiles": {
                "name": "Misplaced Tiles",
                "aliases": ["misplaced tiles", "hamming distance", "piese deplasate", "hamming"],
                "category": "heuristic",
                "default_properties": {"admissible": True}
            },
            "linear_conflict": {
                "name": "Linear Conflict",
                "aliases": ["linear conflict", "conflict heuristic"],
                "category": "heuristic",
                "default_properties": {"admissible": True}
            },
        }
    
    def _init_optimizations(self) -> Dict[str, Dict]:
        """Initialize optimization techniques."""
        return {
            "memoization": {
                "name": "Memoization",
                "aliases": ["memoization", "memoizing", "memorizare", "memo"],
                "category": "optimization_technique",
                "default_properties": {}
            },
            "pruning": {
                "name": "Pruning",
                "aliases": ["pruning", "prune", "taiere", "tăiere"],
                "category": "optimization_technique",
                "default_properties": {}
            },
            "alpha_beta": {
                "name": "Alpha-Beta Pruning",
                "aliases": ["alpha beta", "alpha-beta pruning", "alpha beta pruning", "alpha-beta"],
                "category": "optimization_technique",
                "default_properties": {}
            },
            "caching": {
                "name": "Caching",
                "aliases": ["caching", "cache", "cached"],
                "category": "optimization_technique",
                "default_properties": {}
            },
            "parallel": {
                "name": "Parallel Processing",
                "aliases": ["parallel", "paralel", "concurrent", "parallelization"],
                "category": "optimization_technique",
                "default_properties": {}
            },
        }
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for faster matching."""
        self.entity_patterns = {}
        
        for entity_dict, entity_type in [
            (self.algorithms, "algorithm"),
            (self.problems, "problem"),
            (self.heuristics, "heuristic"),
            (self.optimizations, "optimization")
        ]:
            for entity_id, entity_info in entity_dict.items():
                for alias in entity_info["aliases"]:
                    pattern = re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE)
                    self.entity_patterns[pattern] = (entity_info["name"], entity_type, entity_info)
        
        # Complexity patterns
        self.complexity_pattern = re.compile(r'O\([^)]+\)')
        self.time_context_pattern = re.compile(r'\b(time|runtime|temporal|complexity)\b', re.IGNORECASE)
        self.space_context_pattern = re.compile(r'\b(space|memory|storage)\b', re.IGNORECASE)
    
    def extract_mentions(self, text: str) -> List[EntityMention]:
        """Extract all entity mentions with enhanced context."""
        mentions = []
        
        # Split into sentences for better context
        sentences = re.split(r'[.!?]+', text)
        
        # Tokenize text
        words = []
        for match in re.finditer(r'\b\w+\b', text):
            words.append((match.group(), match.start(), match.end()))
        
        # Extract entity mentions
        for pattern, (entity_name, entity_type, entity_info) in self.entity_patterns.items():
            for match in pattern.finditer(text):
                start_pos = match.start()
                end_pos = match.end()
                
                # Find word position
                word_pos = sum(1 for w, s, e in words if e <= start_pos)
                
                # Extract context windows
                context_before = self._get_context_before(words, word_pos, window=50)
                context_after = self._get_context_after(words, word_pos, window=50)
                
                # Find containing sentence
                sentence = self._find_containing_sentence(sentences, match.group())
                
                # Extract properties
                full_context = f"{context_before} {match.group()} {context_after}"
                properties = self._extract_properties(full_context, entity_info)
                
                mention = EntityMention(
                    entity_name=entity_name,
                    entity_type=entity_type,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    word_position=word_pos,
                    context_before=context_before,
                    context_after=context_after,
                    sentence=sentence,
                    properties=properties
                )
                mentions.append(mention)
        
        return mentions
    
    def _get_context_before(self, words: List[Tuple], pos: int, window: int) -> str:
        """Get context before position."""
        start_idx = max(0, pos - window)
        context_words = [w for w, s, e in words[start_idx:pos]]
        return ' '.join(context_words)
    
    def _get_context_after(self, words: List[Tuple], pos: int, window: int) -> str:
        """Get context after position."""
        end_idx = min(len(words), pos + window + 1)
        context_words = [w for w, s, e in words[pos+1:end_idx]]
        return ' '.join(context_words)
    
    def _find_containing_sentence(self, sentences: List[str], term: str) -> str:
        """Find the sentence containing the term."""
        for sent in sentences:
            if term.lower() in sent.lower():
                return sent.strip()
        return ""
    
    def _extract_properties(self, context: str, entity_info: Dict) -> Dict[str, Any]:
        """Extract properties from context."""
        properties = {}
        
        # Check for explicit properties
        for prop, pattern in self.property_patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                properties[prop] = True
        
        # Add default properties from entity info
        if 'default_properties' in entity_info:
            for prop, value in entity_info['default_properties'].items():
                if prop not in properties:
                    properties[prop] = value
        
        # Add category
        if 'category' in entity_info:
            properties['category'] = entity_info['category']
        
        return properties


class RelationshipDetector:
    """Detects relationships between entities using pattern matching and semantic analysis."""
    
    def __init__(self):
        self.relationship_patterns = self._init_relationship_patterns()
    
    def _init_relationship_patterns(self) -> Dict[str, List[str]]:
        """Initialize comprehensive relationship patterns."""
        return {
            'solves': [
                r'\b{algo}\s+(?:solves?|solving|solved)\s+(?:the\s+)?{prob}\b',
                r'\b{prob}\s+(?:can\s+be\s+)?(?:solved|tackled)\s+(?:using|with|by)\s+{algo}\b',
                r'\b(?:use|using|apply|applying)\s+{algo}\s+(?:to|for)\s+(?:solve|solving)?\s*{prob}\b',
                r'\b{algo}\s+(?:is\s+)?(?:suitable|good|ideal|best|effective)\s+(?:for|on)\s+{prob}\b',
            ],
            'uses': [
                r'\b{algo}\s+(?:uses?|employs?|utilizes?|applies)\s+(?:the\s+)?{heur}\b',
                r'\b{heur}\s+(?:is\s+)?(?:used|employed|utilized)\s+(?:in|by|with)\s+{algo}\b',
                r'\b{algo}\s+with\s+{heur}\b',
            ],
            'outperforms': [
                r'\b{algo1}\s+(?:is\s+)?(?:better|superior|faster|more\s+efficient)\s+(?:than|to)\s+{algo2}\b',
                r'\b{algo1}\s+(?:outperforms?|beats?|dominates?)\s+{algo2}\b',
            ],
            'variant_of': [
                r'\b{algo1}\s+(?:is\s+)?(?:a\s+)?(?:variant|variation|version)\s+(?:of|on)\s+{algo2}\b',
                r'\b{algo1}\s+(?:extends?|modifies?)\s+{algo2}\b',
            ],
            'requires': [
                r'\b{algo}\s+(?:requires?|needs?)\s+{prop}\s+(?:heuristic|function)\b',
            ],
            'applied_to': [
                r'\b{algo}\s+(?:applied|used)\s+(?:on|to|for)\s+{prob}\b',
            ],
        }
    
    def detect_relationships(self, text: str, entities: List[str]) -> List[Tuple[str, str, str, str]]:
        """
        Detect relationships in text.
        Returns list of (entity1, entity2, relation_type, context).
        """
        relationships = []
        
        # Create lowercase versions for matching
        entity_map = {e.lower(): e for e in entities}
        
        for relation_type, patterns in self.relationship_patterns.items():
            for pattern_template in patterns:
                # Try all entity combinations
                for e1_lower, e1 in entity_map.items():
                    for e2_lower, e2 in entity_map.items():
                        if e1 == e2:
                            continue
                        
                        # Fill in pattern
                        if '{algo}' in pattern_template and '{prob}' in pattern_template:
                            pattern = pattern_template.replace('{algo}', re.escape(e1_lower))
                            pattern = pattern.replace('{prob}', re.escape(e2_lower))
                        elif '{algo}' in pattern_template and '{heur}' in pattern_template:
                            pattern = pattern_template.replace('{algo}', re.escape(e1_lower))
                            pattern = pattern.replace('{heur}', re.escape(e2_lower))
                        elif '{algo1}' in pattern_template and '{algo2}' in pattern_template:
                            pattern = pattern_template.replace('{algo1}', re.escape(e1_lower))
                            pattern = pattern.replace('{algo2}', re.escape(e2_lower))
                        else:
                            continue
                        
                        # Search for matches
                        matches = re.finditer(pattern, text, re.IGNORECASE)
                        for match in matches:
                            context = text[max(0, match.start()-100):min(len(text), match.end()+100)]
                            relationships.append((e1, e2, relation_type, context))
        
        return relationships


class CoOccurrenceAnalyzer:
    """Analyzes entity co-occurrences within sliding windows."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.co_occurrence_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
        self.entity_counts: Dict[str, int] = defaultdict(int)
    
    def build_matrix(self, mentions: List[EntityMention]):
        """Build co-occurrence matrix from mentions."""
        # Sort mentions by position
        sorted_mentions = sorted(mentions, key=lambda m: m.word_position)
        
        # Sliding window approach
        for i, mention1 in enumerate(sorted_mentions):
            self.entity_counts[mention1.entity_name] += 1
            
            for j in range(i + 1, len(sorted_mentions)):
                mention2 = sorted_mentions[j]
                
                # Check if within window
                distance = mention2.word_position - mention1.word_position
                if distance > self.window_size:
                    break
                
                # Record co-occurrence (both directions)
                pair1 = (mention1.entity_name, mention2.entity_name)
                pair2 = (mention2.entity_name, mention1.entity_name)
                self.co_occurrence_matrix[pair1] += 1
                self.co_occurrence_matrix[pair2] += 1
    
    def get_co_occurrence_strength(self, entity1: str, entity2: str) -> float:
        """Calculate normalized co-occurrence strength."""
        co_count = self.co_occurrence_matrix.get((entity1, entity2), 0)
        if co_count == 0:
            return 0.0
        
        count1 = self.entity_counts.get(entity1, 1)
        count2 = self.entity_counts.get(entity2, 1)
        
        # Normalized by geometric mean of individual counts
        return co_count / (count1 * count2) ** 0.5
    
    def get_all_co_occurrences(self, min_strength: float = 0.1) -> List[Tuple[str, str, float]]:
        """Get all co-occurrences above threshold."""
        results = []
        seen_pairs = set()
        
        for (e1, e2), count in self.co_occurrence_matrix.items():
            if (e1, e2) in seen_pairs or (e2, e1) in seen_pairs:
                continue
            
            strength = self.get_co_occurrence_strength(e1, e2)
            if strength >= min_strength:
                results.append((e1, e2, strength))
                seen_pairs.add((e1, e2))
        
        return sorted(results, key=lambda x: x[2], reverse=True)


class DocumentProcessorV5:
    """Enhanced document processor with pre-seeded KG and better relationship detection."""
    
    def __init__(self, resource: str, resource_type: str = 'local'):
        self.resource = resource
        self.resource_type = resource_type
        
        if resource_type == 'local':
            self.reader = LocalFileReader()
        else:
            self.reader = WebResourceReader()
        
        self.extractor = EnhancedEntityExtractor()
        self.relationship_detector = RelationshipDetector()
        self.knowledge_graph = KnowledgeGraph()
        
        # Entity profiles
        self.entity_profiles: Dict[str, EntityProfile] = {}
        
        # Co-occurrence analyzer
        self.co_occurrence = CoOccurrenceAnalyzer(window_size=50)
        
        # Pre-seed the knowledge graph
        self._preseed_knowledge_graph()
    
    def _preseed_knowledge_graph(self):
        """Pre-seed KG with all known algorithms, problems, heuristics."""
        print("[*] Pre-seeding knowledge graph with known entities...")
        
        # Add all algorithms
        for algo_id, algo_info in self.extractor.algorithms.items():
            node_id = f"algo_{algo_id}"
            node = Node(
                id=node_id,
                name=algo_info["name"],
                type="algorithm",
                properties={
                    "full_name": algo_info.get("full_name", algo_info["name"]),
                    **algo_info.get("default_properties", {})
                }
            )
            self.knowledge_graph.add_node(node)
        
        # Add all problems
        for prob_id, prob_info in self.extractor.problems.items():
            node_id = f"prob_{prob_id}"
            node = Node(
                id=node_id,
                name=prob_info["name"],
                type="problem",
                properties={
                    "category": prob_info.get("category", ""),
                    **prob_info.get("default_properties", {})
                }
            )
            self.knowledge_graph.add_node(node)
        
        # Add all heuristics
        for heur_id, heur_info in self.extractor.heuristics.items():
            node_id = f"heur_{heur_id}"
            node = Node(
                id=node_id,
                name=heur_info["name"],
                type="heuristic",
                properties={
                    "category": heur_info.get("category", ""),
                    **heur_info.get("default_properties", {})
                }
            )
            self.knowledge_graph.add_node(node)
        
        # Add all optimizations
        for opt_id, opt_info in self.extractor.optimizations.items():
            node_id = f"opt_{opt_id}"
            node = Node(
                id=node_id,
                name=opt_info["name"],
                type="optimization",
                properties={
                    "category": opt_info.get("category", ""),
                    **opt_info.get("default_properties", {})
                }
            )
            self.knowledge_graph.add_node(node)
        
        # Add category nodes
        for category in ["informed", "uninformed", "local_search"]:
            node_id = f"cat_{category}"
            self.knowledge_graph.add_node(Node(
                id=node_id,
                name=category.replace('_', ' ').title(),
                type="category"
            ))
        
        # Add classification edges
        for algo_id, algo_info in self.extractor.algorithms.items():
            category = algo_info.get("category")
            if category:
                algo_node_id = f"algo_{algo_id}"
                cat_node_id = f"cat_{category}"
                if cat_node_id in self.knowledge_graph.nodes:
                    self.knowledge_graph.add_edge(Edge(
                        source=algo_node_id,
                        target=cat_node_id,
                        relation_type="classified_as",
                        confidence=1.0,
                        context="Pre-seeded classification"
                    ))
        
        node_count = len(self.knowledge_graph.nodes)
        edge_count = len(self.knowledge_graph.edges)
        print(f"[+] Pre-seeded: {node_count} nodes, {edge_count} edges")
    
    def process(self) -> Tuple[Dict[str, EntityProfile], KnowledgeGraph]:
        """Process document with enhanced relationship detection."""
        print(f"\n{'='*80}")
        print(f"Processing {self.resource_type} resource: {self.resource}")
        print(f"{'='*80}")
        
        # Step 1: Read document
        print("\n[1/7] Reading document...")
        raw_content = self.reader.read(self.resource)
        print(f"      Read {len(raw_content)} characters")
        
        # Step 2: Extract entity mentions
        print("\n[2/7] Extracting entity mentions...")
        mentions = self.extractor.extract_mentions(raw_content)
        print(f"      Found {len(mentions)} entity mentions")
        
        # Step 3: Build entity profiles
        print("\n[3/7] Building entity profiles...")
        self._build_entity_profiles(mentions)
        print(f"      Created {len(self.entity_profiles)} unique entity profiles")
        
        # Step 4: Update KG nodes with discovered properties
        print("\n[4/7] Updating node properties from document...")
        self._update_node_properties()
        
        # Step 5: Detect relationships using patterns
        print("\n[5/7] Detecting relationships with pattern matching...")
        entity_names = list(self.entity_profiles.keys())
        detected_relationships = self.relationship_detector.detect_relationships(raw_content, entity_names)
        self._add_detected_relationships(detected_relationships)
        print(f"      Found {len(detected_relationships)} explicit relationships")
        
        # Step 6: Build co-occurrence matrix and create edges
        print("\n[6/7] Analyzing co-occurrences...")
        self.co_occurrence.build_matrix(mentions)
        self._create_cooccurrence_edges()
        
        # Step 7: Post-process to infer additional relationships
        print("\n[7/7] Post-processing: Inferring additional relationships...")
        self._infer_relationships()
        
        print(f"\n{'='*80}")
        print(f"Processing Complete!")
        print(f"{'='*80}")
        print(f"  Entity mentions: {len(mentions)}")
        print(f"  Unique entities: {len(self.entity_profiles)}")
        print(f"  Knowledge graph: {len(self.knowledge_graph.nodes)} nodes, {len(self.knowledge_graph.edges)} edges")
        print(f"{'='*80}\n")
        
        return self.entity_profiles, self.knowledge_graph
    
    def _build_entity_profiles(self, mentions: List[EntityMention]):
        """Build aggregated profiles from all mentions."""
        for mention in mentions:
            if mention.entity_name not in self.entity_profiles:
                self.entity_profiles[mention.entity_name] = EntityProfile(
                    name=mention.entity_name,
                    entity_type=mention.entity_type
                )
            
            profile = self.entity_profiles[mention.entity_name]
            profile.add_mention(mention)
    
    def _update_node_properties(self):
        """Update pre-seeded nodes with properties discovered in document."""
        for entity_name, profile in self.entity_profiles.items():
            node_id = self._find_node_id(entity_name, profile.entity_type)
            if not node_id:
                continue
            
            node = self.knowledge_graph.get_node(node_id)
            if node:
                # Merge discovered properties (only if mentioned frequently)
                total_mentions = profile.mention_count
                for prop, count in profile.all_properties.items():
                    if count / total_mentions >= 0.3:  # 30% threshold
                        node.properties[prop] = True
    
    def _add_detected_relationships(self, relationships: List[Tuple[str, str, str, str]]):
        """Add explicitly detected relationships to KG."""
        for entity1, entity2, relation_type, context in relationships:
            profile1 = self.entity_profiles.get(entity1)
            profile2 = self.entity_profiles.get(entity2)
            
            if not profile1 or not profile2:
                continue
            
            source_id = self._find_node_id(entity1, profile1.entity_type)
            target_id = self._find_node_id(entity2, profile2.entity_type)
            
            if source_id and target_id:
                edge = Edge(
                    source=source_id,
                    target=target_id,
                    relation_type=relation_type,
                    properties={"explicit": True, "pattern_matched": True},
                    confidence=0.9,
                    context=context[:200]
                )
                self.knowledge_graph.add_edge(edge)
    
    def _create_cooccurrence_edges(self):
        """Create edges based on co-occurrence analysis."""
        co_occurrences = self.co_occurrence.get_all_co_occurrences(min_strength=0.2)
        
        for entity1, entity2, strength in co_occurrences:
            profile1 = self.entity_profiles.get(entity1)
            profile2 = self.entity_profiles.get(entity2)
            
            if not profile1 or not profile2:
                continue
            
            # Infer relationship type based on entity types
            relation_type = self._infer_relation_type(
                entity1, profile1.entity_type,
                entity2, profile2.entity_type
            )
            
            if relation_type:
                source_id = self._find_node_id(entity1, profile1.entity_type)
                target_id = self._find_node_id(entity2, profile2.entity_type)
                
                if source_id and target_id:
                    # Get context
                    contexts = profile1.get_contexts_with(entity2, max_contexts=1)
                    context = contexts[0] if contexts else ""
                    
                    edge = Edge(
                        source=source_id,
                        target=target_id,
                        relation_type=relation_type,
                        properties={
                            "co_occurrence_strength": strength,
                            "mention_count": self.co_occurrence.co_occurrence_matrix.get((entity1, entity2), 0)
                        },
                        confidence=min(0.8, strength),
                        context=context[:200]
                    )
                    self.knowledge_graph.add_edge(edge)
    
    def _infer_relationships(self):
        """Post-processing phase: Infer additional relationships."""
        # Transitivity: If A uses B and B solves C, then A can solve C
        self._infer_transitive_relationships()
        
        # Same category relationships
        self._infer_category_relationships()
        
        # Complexity relationships
        self._infer_complexity_relationships()
    
    def _infer_transitive_relationships(self):
        """Infer transitive relationships (A uses B, B solves C => A can solve C)."""
        algorithms = self.knowledge_graph.get_nodes_by_type("algorithm")
        
        for algo_node in algorithms:
            # Find what this algorithm uses
            uses_edges = self.knowledge_graph.get_outgoing_edges(algo_node.id, "uses")
            
            for uses_edge in uses_edges:
                # Find what the used entity solves
                solves_edges = self.knowledge_graph.get_outgoing_edges(uses_edge.target, "solves")
                
                for solves_edge in solves_edges:
                    # Create indirect "applicable_to" relationship
                    edge = Edge(
                        source=algo_node.id,
                        target=solves_edge.target,
                        relation_type="applicable_to",
                        properties={"inferred": True, "via": uses_edge.target},
                        confidence=0.6,
                        context="Inferred from transitive relationship"
                    )
                    self.knowledge_graph.add_edge(edge)
    
    def _infer_category_relationships(self):
        """Infer relationships between algorithms in same category."""
        categories = ["informed", "uninformed", "local_search"]
        
        for category in categories:
            cat_id = f"cat_{category}"
            if cat_id not in self.knowledge_graph.nodes:
                continue
            
            # Get all algorithms in this category
            incoming = self.knowledge_graph.get_incoming_edges(cat_id, "classified_as")
            algo_ids = [e.source for e in incoming]
            
            # Create "related_to" edges between algorithms in same category
            for i, algo1_id in enumerate(algo_ids):
                for algo2_id in algo_ids[i+1:]:
                    edge = Edge(
                        source=algo1_id,
                        target=algo2_id,
                        relation_type="related_to",
                        properties={"inferred": True, "reason": f"same_category_{category}"},
                        confidence=0.5,
                        context=f"Both belong to {category} category"
                    )
                    self.knowledge_graph.add_edge(edge)
    
    def _infer_complexity_relationships(self):
        """Infer relationships based on complexity properties."""
        # Find algorithms with optimality property
        optimal_algos = []
        for node in self.knowledge_graph.get_nodes_by_type("algorithm"):
            if node.properties.get("optimal"):
                optimal_algos.append(node.id)
        
        # Create "comparable_optimality" edges
        for i, algo1_id in enumerate(optimal_algos):
            for algo2_id in optimal_algos[i+1:]:
                edge = Edge(
                    source=algo1_id,
                    target=algo2_id,
                    relation_type="comparable_optimality",
                    properties={"inferred": True, "reason": "both_optimal"},
                    confidence=0.5,
                    context="Both algorithms guarantee optimal solutions"
                )
                self.knowledge_graph.add_edge(edge)
    
    def _infer_relation_type(self, entity1: str, type1: str, entity2: str, type2: str) -> Optional[str]:
        """Infer relationship type based on entity types."""
        if type1 == "algorithm" and type2 == "problem":
            return "solves"
        elif type1 == "problem" and type2 == "algorithm":
            return "solved_by"
        elif type1 == "algorithm" and type2 == "heuristic":
            return "uses"
        elif type1 == "heuristic" and type2 == "algorithm":
            return "used_by"
        elif type1 == "algorithm" and type2 == "algorithm":
            return "related_to"
        elif type1 == "problem" and type2 == "problem":
            return "related_to"
        elif type1 == "algorithm" and type2 == "optimization":
            return "can_use"
        else:
            return "related_to"
    
    def _find_node_id(self, entity_name: str, entity_type: str) -> Optional[str]:
        """Find node ID for entity name."""
        for node_id, node in self.knowledge_graph.nodes.items():
            if node.name == entity_name and node.type == entity_type:
                return node_id
        return None
    
    def _make_node_id(self, entity_name: str, entity_type: str) -> str:
        """Generate node ID from entity name and type."""
        prefix_map = {
            "algorithm": "algo",
            "problem": "prob",
            "heuristic": "heur",
            "optimization": "opt"
        }
        prefix = prefix_map.get(entity_type, entity_type[:4])
        clean_name = re.sub(r'[^a-z0-9]', '_', entity_name.lower())
        return f"{prefix}_{clean_name}"


# Export main class
__all__ = ['DocumentProcessorV5', 'EntityMention', 'EntityProfile']
