"""
Build Comprehensive Knowledge Graph from All Sources
Process all PDFs and merge into single knowledge graph
"""

import json
from document_processor_v5 import DocumentProcessorV5
from knowledge_graph import KnowledgeGraph, Node, Edge

def process_all_sources():
    """Process all PDF sources and merge into comprehensive KG."""
    
    print("\n" + "="*100)
    print(" Building Comprehensive Knowledge Graph from All Sources")
    print("="*100)
    
    # List all PDF sources
    pdf_sources = [
        r"f:\General Info\Anul III\AI\Knowledge Source\ai-lecture03.pdf",
        r"f:\General Info\Anul III\AI\Knowledge Source\IA_2_SBM_I.pdf",
        r"f:\General Info\Anul III\AI\Knowledge Source\IA_3_SBM_II.pdf",
        r"f:\General Info\Anul III\AI\Knowledge Source\lecture2.pdf",
        r"f:\General Info\Anul III\AI\Knowledge Source\Unit-3.pdf",
    ]
    
    # Start with first processor (has pre-seeded KG)
    print(f"\n[*] Initializing with first source...")
    first_processor = DocumentProcessorV5(pdf_sources[0], resource_type='local')
    entity_profiles, merged_kg = first_processor.process()
    
    # Process remaining sources and merge
    for i, pdf_path in enumerate(pdf_sources[1:], 2):
        print(f"\n[*] Processing source {i}/{len(pdf_sources)}...")
        processor = DocumentProcessorV5(pdf_path, resource_type='local')
        profiles, kg = processor.process()
        
        # Merge knowledge graphs
        print(f"    Merging knowledge graph...")
        merged_kg = merge_knowledge_graphs(merged_kg, kg)
    
    print(f"\n{'='*100}")
    print(f" Merge Complete!")
    print(f"{'='*100}")
    
    # Save merged KG
    print("\n[*] Saving comprehensive knowledge graph...")
    graph_data = {
        "nodes": [
            {
                "id": node.id,
                "name": node.name,
                "type": node.type,
                "properties": node.properties
            }
            for node in merged_kg.nodes.values()
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
            for edge in merged_kg.edges
        ]
    }
    
    output_path = "knowledge_graph.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    print(f"[+] Saved to {output_path}")
    
    # Print comprehensive statistics
    print("\n" + "="*100)
    print(" Final Knowledge Graph Statistics")
    print("="*100)
    
    node_types = {}
    for node in merged_kg.nodes.values():
        node_types[node.type] = node_types.get(node.type, 0) + 1
    
    print("\nNodes by Type:")
    for node_type, count in sorted(node_types.items()):
        print(f"  {node_type:20s}: {count:4d}")
    
    print(f"\nTotal Nodes: {len(merged_kg.nodes)}")
    
    relation_types = {}
    for edge in merged_kg.edges:
        relation_types[edge.relation_type] = relation_types.get(edge.relation_type, 0) + 1
    
    print("\nEdges by Relation Type:")
    for relation_type, count in sorted(relation_types.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {relation_type:30s}: {count:4d}")
    
    print(f"\nTotal Edges: {len(merged_kg.edges)}")
    
    # Show algorithm coverage
    print("\n" + "="*100)
    print(" Algorithm Coverage")
    print("="*100)
    
    algorithms = merged_kg.get_nodes_by_type("algorithm")
    print(f"\nTotal Algorithms: {len(algorithms)}")
    
    # Count connections per algorithm
    algo_connections = {}
    for algo in algorithms:
        outgoing = len(merged_kg.get_outgoing_edges(algo.id))
        incoming = len(merged_kg.get_incoming_edges(algo.id))
        algo_connections[algo.name] = outgoing + incoming
    
    print("\nTop 10 Most Connected Algorithms:")
    for algo_name, conn_count in sorted(algo_connections.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {algo_name:25s}: {conn_count:3d} connections")
    
    # Show problem coverage
    print("\n" + "="*100)
    print(" Problem Coverage")
    print("="*100)
    
    problems = merged_kg.get_nodes_by_type("problem")
    print(f"\nTotal Problems: {len(problems)}")
    
    # Count which problems have solving algorithms
    for problem in problems:
        solving_algos = merged_kg.get_incoming_edges(problem.id, "solves")
        solved_by_algos = merged_kg.get_incoming_edges(problem.id, "solved_by")
        total_solvers = len(solving_algos) + len(solved_by_algos)
        
        if total_solvers > 0:
            print(f"  {problem.name:25s}: {total_solvers} solving algorithms")
    
    # Show sample relationships
    print("\n" + "="*100)
    print(" Sample High-Confidence Relationships")
    print("="*100)
    
    high_conf_edges = [e for e in merged_kg.edges if e.confidence >= 0.7]
    high_conf_edges.sort(key=lambda e: e.confidence, reverse=True)
    
    print("\n[Top 15 Highest Confidence Relationships]")
    for edge in high_conf_edges[:15]:
        source_node = merged_kg.get_node(edge.source)
        target_node = merged_kg.get_node(edge.target)
        if source_node and target_node:
            print(f"  {source_node.name:20s} --[{edge.relation_type}]--> {target_node.name:20s} (conf: {edge.confidence:.2f})")
    
    print("\n" + "="*100)
    print(" Knowledge Graph Build Complete!")
    print("="*100 + "\n")
    
    return merged_kg


def merge_knowledge_graphs(kg1: KnowledgeGraph, kg2: KnowledgeGraph) -> KnowledgeGraph:
    """Merge two knowledge graphs, combining nodes and edges."""
    merged = KnowledgeGraph()
    
    # Merge nodes (kg1 takes precedence, but merge properties)
    for node_id, node in kg1.nodes.items():
        merged.add_node(node)
    
    for node_id, node in kg2.nodes.items():
        if node_id in merged.nodes:
            # Merge properties
            existing_node = merged.nodes[node_id]
            for prop, value in node.properties.items():
                if prop not in existing_node.properties:
                    existing_node.properties[prop] = value
        else:
            merged.add_node(node)
    
    # Merge edges (keep all, update confidence if duplicate)
    edge_map = {}
    
    for edge in kg1.edges:
        key = (edge.source, edge.target, edge.relation_type)
        edge_map[key] = edge
    
    for edge in kg2.edges:
        key = (edge.source, edge.target, edge.relation_type)
        if key in edge_map:
            # Update confidence to max
            existing_edge = edge_map[key]
            existing_edge.confidence = max(existing_edge.confidence, edge.confidence)
            
            # Merge properties
            for prop, value in edge.properties.items():
                if prop == "mention_count":
                    existing_edge.properties[prop] = existing_edge.properties.get(prop, 0) + value
                elif prop == "co_occurrence_strength":
                    existing_edge.properties[prop] = max(existing_edge.properties.get(prop, 0), value)
                elif prop not in existing_edge.properties:
                    existing_edge.properties[prop] = value
        else:
            edge_map[key] = edge
    
    # Add all edges to merged graph
    for edge in edge_map.values():
        merged.add_edge(edge)
    
    return merged


if __name__ == "__main__":
    kg = process_all_sources()
