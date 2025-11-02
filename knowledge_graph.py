"""
Knowledge Graph Module
======================
Builds and manages a knowledge graph from extracted AI concepts.
Represents relationships between algorithms, problems, heuristics, and their properties.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Any, Optional
from collections import defaultdict


@dataclass
class Node:
    """Represents a concept in the knowledge graph."""
    id: str  # Unique identifier
    name: str  # Display name
    type: str  # "algorithm", "problem", "heuristic", "property", "time_complexity", "space_complexity", "time_issue", "memory_issue"
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id


@dataclass
class Edge:
    """Represents a relationship between two concepts."""
    source: str  # Node ID
    target: str  # Node ID
    relation_type: str  # "uses", "solves", "has_property", "classified_as", "has_time_complexity", "has_space_complexity", "suffers_from", etc.
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0  # Confidence score (0-1)
    context: str = ""  # Original text context where relationship was found
    
    def __hash__(self):
        return hash((self.source, self.target, self.relation_type))
    
    def __eq__(self, other):
        return (isinstance(other, Edge) and 
                self.source == other.source and 
                self.target == other.target and 
                self.relation_type == other.relation_type)


class KnowledgeGraph:
    """
    Knowledge graph for AI search strategies and related concepts.
    """
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}  # node_id -> Node
        self.edges: Set[Edge] = set()
        
        # Indices for fast lookups
        self.nodes_by_type: Dict[str, Set[str]] = defaultdict(set)
        self.outgoing_edges: Dict[str, Set[Edge]] = defaultdict(set)
        self.incoming_edges: Dict[str, Set[Edge]] = defaultdict(set)
    
    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        self.nodes_by_type[node.type].add(node.id)
    
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        # Ensure nodes exist
        if edge.source not in self.nodes or edge.target not in self.nodes:
            return
        
        self.edges.add(edge)
        self.outgoing_edges[edge.source].add(edge)
        self.incoming_edges[edge.target].add(edge)
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_nodes_by_type(self, node_type: str) -> List[Node]:
        """Get all nodes of a specific type."""
        return [self.nodes[nid] for nid in self.nodes_by_type.get(node_type, [])]
    
    def get_outgoing_edges(self, node_id: str, relation_type: Optional[str] = None) -> List[Edge]:
        """Get all outgoing edges from a node, optionally filtered by relation type."""
        edges = self.outgoing_edges.get(node_id, set())
        if relation_type:
            edges = {e for e in edges if e.relation_type == relation_type}
        return list(edges)
    
    def get_incoming_edges(self, node_id: str, relation_type: Optional[str] = None) -> List[Edge]:
        """Get all incoming edges to a node, optionally filtered by relation type."""
        edges = self.incoming_edges.get(node_id, set())
        if relation_type:
            edges = {e for e in edges if e.relation_type == relation_type}
        return list(edges)
    
    def get_neighbors(self, node_id: str, relation_type: Optional[str] = None) -> List[Node]:
        """Get all neighboring nodes connected by outgoing edges."""
        edges = self.get_outgoing_edges(node_id, relation_type)
        return [self.nodes[e.target] for e in edges if e.target in self.nodes]
    
    def get_path(self, source_id: str, target_id: str, max_depth: int = 3) -> Optional[List[str]]:
        """Find shortest path between two nodes using BFS."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        
        if source_id == target_id:
            return [source_id]
        
        visited = {source_id}
        queue = [(source_id, [source_id])]
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            for edge in self.outgoing_edges.get(current, []):
                if edge.target not in visited:
                    visited.add(edge.target)
                    new_path = path + [edge.target]
                    
                    if edge.target == target_id:
                        return new_path
                    
                    queue.append((edge.target, new_path))
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation."""
        return {
            "nodes": [
                {
                    "id": node.id,
                    "name": node.name,
                    "type": node.type,
                    "properties": node.properties
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation_type,
                    "properties": edge.properties,
                    "confidence": edge.confidence,
                    "context": edge.context[:100] if edge.context else ""
                }
                for edge in self.edges
            ]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "nodes_by_type": {
                node_type: len(node_ids) 
                for node_type, node_ids in self.nodes_by_type.items()
            },
            "avg_connections_per_node": len(self.edges) / len(self.nodes) if self.nodes else 0
        }
    
    def export_json(self, filepath: str) -> None:
        """Export graph to JSON file."""
        import json
        data = {
            "nodes": [
                {
                    "id": node.id,
                    "name": node.name,
                    "type": node.type,
                    "properties": node.properties
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relation_type": edge.relation_type,
                    "properties": edge.properties,
                    "confidence": edge.confidence,
                    "context": edge.context[:200] if edge.context else ""
                }
                for edge in self.edges
            ],
            "statistics": self.get_statistics()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def print_summary(self):
        """Print a summary of the knowledge graph."""
        stats = self.get_statistics()
        
        print("\n" + "="*80)
        print("KNOWLEDGE GRAPH SUMMARY")
        print("="*80)
        
        print(f"\nStatistics:")
        print(f"  * Total Nodes: {stats['total_nodes']}")
        print(f"  * Total Edges: {stats['total_edges']}")
        print(f"  * Avg Connections: {stats['avg_connections_per_node']:.2f}")
        
        print(f"\nNodes by Type:")
        for node_type, count in sorted(stats['nodes_by_type'].items()):
            print(f"  * {node_type.title()}: {count}")
        
        # Show some example relationships
        print(f"\nSample Relationships:")
        for edge in list(self.edges)[:10]:
            source = self.nodes.get(edge.source)
            target = self.nodes.get(edge.target)
            if source and target:
                props = ", ".join(f"{k}={v}" for k, v in list(edge.properties.items())[:2])
                props_str = f" ({props})" if props else ""
                print(f"  * {source.name} --[{edge.relation_type}]--> {target.name}{props_str}")
