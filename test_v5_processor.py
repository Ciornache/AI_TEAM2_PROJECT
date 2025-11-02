"""
Test script for Document Processor V5
Tests the new pre-seeded KG approach with enhanced relationship detection
"""

import json
from document_processor_v5 import DocumentProcessorV5

def test_v5_processor():
    """Test V5 processor on existing PDF."""
    
    print("\n" + "="*80)
    print(" Document Processor V5 - Testing")
    print("="*80)
    
    # Process the PDF file
    pdf_path = r"f:\General Info\Anul III\AI\Knowledge Source\ai-lecture03.pdf"
    
    processor = DocumentProcessorV5(pdf_path, resource_type='local')
    entity_profiles, knowledge_graph = processor.process()
    
    # Save the new knowledge graph
    print("\n[*] Saving knowledge graph to JSON...")
    graph_data = {
        "nodes": [
            {
                "id": node.id,
                "name": node.name,
                "type": node.type,
                "properties": node.properties
            }
            for node in knowledge_graph.nodes.values()
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "relation_type": edge.relation_type,
                "properties": edge.properties,
                "confidence": edge.confidence,
                "context": edge.context
            }
            for edge in knowledge_graph.edges
        ]
    }
    
    output_path = "knowledge_graph_v5.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    print(f"[+] Saved to {output_path}")
    
    # Print statistics
    print("\n" + "="*80)
    print(" Knowledge Graph Statistics")
    print("="*80)
    
    node_types = {}
    for node in knowledge_graph.nodes.values():
        node_types[node.type] = node_types.get(node.type, 0) + 1
    
    print("\nNodes by Type:")
    for node_type, count in sorted(node_types.items()):
        print(f"  {node_type:20s}: {count:3d}")
    
    print(f"\nTotal Nodes: {len(knowledge_graph.nodes)}")
    
    relation_types = {}
    for edge in knowledge_graph.edges:
        relation_types[edge.relation_type] = relation_types.get(edge.relation_type, 0) + 1
    
    print("\nEdges by Relation Type:")
    for relation_type, count in sorted(relation_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {relation_type:25s}: {count:3d}")
    
    print(f"\nTotal Edges: {len(knowledge_graph.edges)}")
    
    # Show some example relationships
    print("\n" + "="*80)
    print(" Sample Relationships")
    print("="*80)
    
    # Show algorithms solving problems
    print("\n[Algorithms → Problems (solves)]")
    solves_edges = [e for e in knowledge_graph.edges if e.relation_type == "solves"][:10]
    for edge in solves_edges:
        source_node = knowledge_graph.get_node(edge.source)
        target_node = knowledge_graph.get_node(edge.target)
        if source_node and target_node:
            print(f"  {source_node.name} → {target_node.name} (confidence: {edge.confidence:.2f})")
    
    # Show algorithms using heuristics
    print("\n[Algorithms → Heuristics (uses)]")
    uses_edges = [e for e in knowledge_graph.edges if e.relation_type == "uses"][:10]
    for edge in uses_edges:
        source_node = knowledge_graph.get_node(edge.source)
        target_node = knowledge_graph.get_node(edge.target)
        if source_node and target_node:
            print(f"  {source_node.name} → {target_node.name} (confidence: {edge.confidence:.2f})")
    
    # Show algorithm properties
    print("\n" + "="*80)
    print(" Algorithm Properties (from pre-seed + document)")
    print("="*80)
    
    algorithms = knowledge_graph.get_nodes_by_type("algorithm")
    for algo in sorted(algorithms, key=lambda x: x.name)[:15]:
        props = ", ".join([f"{k}={v}" for k, v in algo.properties.items() if k != "full_name"])
        print(f"  {algo.name:20s}: {props}")
    
    print("\n" + "="*80)
    print(" Testing Complete!")
    print("="*80)
    
    return knowledge_graph


if __name__ == "__main__":
    kg = test_v5_processor()
